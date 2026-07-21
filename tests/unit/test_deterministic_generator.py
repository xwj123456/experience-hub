from __future__ import annotations

import tomllib
from datetime import UTC, datetime
from inspect import Parameter, signature
from pathlib import Path
from time import perf_counter
from uuid import UUID

import pytest

from experience_hub.canonical import canonical_json_bytes
from experience_hub.inspiration.generators.base import (
    GeneratorResult,
    OperatorFailureCode,
)
from experience_hub.inspiration.generators.deterministic import (
    DeterministicIdeaGenerator,
)
from experience_hub.inspiration.hashing import hash_snapshot, stable_evidence_key
from experience_hub.inspiration.models import (
    EvidenceSourceState,
    EvidenceSourceType,
    FrozenSnapshot,
    IdeaDraft,
    InspirationOperator,
    SnapshotEvidenceReference,
    SnapshotItem,
)
from experience_hub.inspiration.operators import (
    causal_gap_pairs,
    evidence_conflicts,
)

NOW = datetime(2026, 7, 18, 12, tzinfo=UTC)
RUN_ID = UUID("00000000-0000-0000-0000-000000000100")
GOAL = "Prevent externally visible partial state"
CONTEXT = "service=ledger; consistency=strict"


def _uuid(value: int) -> UUID:
    return UUID(f"00000000-0000-0000-0000-{value:012d}")


def _hash(value: int) -> str:
    return f"{value:064x}"


def _marked_text(
    start: str,
    end: str,
    *,
    length: int,
    fill: str,
) -> str:
    assert len(fill) == 1
    return start + fill * (length - len(start) - len(end)) + end


def _item(
    rank: int,
    *,
    summary: str,
    mechanism: str,
    applicability: tuple[str, ...],
    falsifiers: tuple[str, ...] = (),
    tags: tuple[str, ...] = (),
    excerpt: str = "Frozen supporting observation.",
) -> SnapshotItem:
    source_id = _uuid(1_000 + rank)
    version_id = _uuid(2_000 + rank)
    content_hash = _hash(3_000 + rank)
    return SnapshotItem(
        snapshot_item_id=_uuid(4_000 + rank),
        stable_evidence_key=stable_evidence_key(
            source_type=EvidenceSourceType.EXPERIENCE,
            source_id=source_id,
            source_version_id=version_id,
            content_hash=content_hash,
        ),
        run_id=RUN_ID,
        source_type=EvidenceSourceType.EXPERIENCE,
        source_id=source_id,
        source_version_id=version_id,
        source_state=EvidenceSourceState.WARM,
        source_trust=1.0,
        rank=rank,
        summary=summary,
        mechanism=mechanism,
        applicability=applicability,
        tags=tags,
        falsifiers=falsifiers,
        excerpt=excerpt,
        content_hash=content_hash,
        captured_at=NOW,
    )


def _snapshot(items: tuple[SnapshotItem, ...] | None = None) -> FrozenSnapshot:
    retained = (
        items
        if items is not None
        else (
            _item(
                1,
                summary="Cache commit barrier invalidation succeeds.",
                mechanism="commit barrier cache invalidation",
                applicability=(
                    "database commit succeeds",
                    "cache entry exists",
                ),
                falsifiers=("Cache readers still observe partial state.",),
                tags=("cache", "database"),
            ),
            _item(
                2,
                summary="Queue acknowledges durable work.",
                mechanism="durability gate queue release",
                applicability=("message processing is idempotent",),
                falsifiers=("Acknowledgement precedes durable processing.",),
                tags=("queue",),
            ),
            _item(
                3,
                summary="Cache commit barrier invalidation does not succeed.",
                mechanism="not commit barrier cache invalidation",
                applicability=("rollback remains possible",),
                falsifiers=("Cache commit barrier invalidation succeeds.",),
                tags=("cache", "rollback"),
            ),
            _item(
                4,
                summary="Immune cells release cytokines.",
                mechanism="durability gate immune release",
                applicability=("recognition precedes release",),
                falsifiers=("Release occurs without recognition.",),
                tags=("biology",),
            ),
        )
    )
    return FrozenSnapshot(
        run_id=RUN_ID,
        items=retained,
        snapshot_hash=hash_snapshot(retained),
        frozen_at=NOW,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("operator", "expected_count"),
    (
        (InspirationOperator.CAUSAL_GAP, 3),
        (InspirationOperator.COUNTERFACTUAL, 3),
        (InspirationOperator.DISTANT_ANALOGY, 2),
    ),
)
async def test_each_operator_is_complete_grounded_and_byte_stable(
    operator: InspirationOperator,
    expected_count: int,
) -> None:
    generator = DeterministicIdeaGenerator()
    snapshot = _snapshot()

    first = await generator.generate(
        goal=GOAL,
        context=CONTEXT,
        frozen_items=snapshot.items,
        operator=operator,
        branch_limit=3,
    )
    second = await generator.generate(
        goal=GOAL,
        context=CONTEXT,
        frozen_items=snapshot.items,
        operator=operator,
        branch_limit=3,
    )

    assert isinstance(first, GeneratorResult)
    assert first.error_code is None
    assert first.output_tokens_consumed == 0
    assert len(first.ideas) == expected_count
    assert canonical_json_bytes(first) == canonical_json_bytes(second)
    evidence = {
        (item.snapshot_item_id, item.stable_evidence_key) for item in snapshot.items
    }
    for idea in first.ideas:
        assert all(
            (
                idea.title,
                idea.hypothesis,
                idea.mechanism,
                idea.predictions,
                idea.falsifiers,
                idea.assumptions,
                idea.proposed_test,
                idea.evidence,
            )
        )
        assert {
            (reference.id, reference.stable_evidence_key) for reference in idea.evidence
        } <= evidence


@pytest.mark.asyncio
async def test_causal_gap_golden_includes_nonadjacent_negation_conflict() -> None:
    result = await DeterministicIdeaGenerator().generate(
        goal=GOAL,
        context=CONTEXT,
        frozen_items=_snapshot().items,
        operator=InspirationOperator.CAUSAL_GAP,
        branch_limit=3,
    )

    assert result.ideas[0].model_dump(mode="json") == {
        "title": (
            "Causal bridge: Cache commit barrier invalidation succeeds. -> "
            "Queue acknowledges durable work."
        ),
        "hypothesis": (
            'For goal "Prevent externally visible partial state", '
            '"Cache commit barrier invalidation succeeds." and '
            '"Queue acknowledges durable work." are linked by an unobserved '
            'transition controlled by the change from "commit barrier cache '
            'invalidation" to "durability gate queue release".'
        ),
        "mechanism": (
            'Bridge the transition between "commit barrier cache '
            'invalidation" and "durability gate queue release" through one '
            "measurable intermediate state."
        ),
        "predictions": [
            'The intermediate state changes after "Cache commit barrier '
            'invalidation succeeds." and before "Queue acknowledges durable '
            'work.".',
            "Intervening on the intermediate state changes the probability "
            'of "Queue acknowledges durable work.".',
        ],
        "falsifiers": [
            'No measurable intermediate improves prediction of "Queue '
            'acknowledges durable work." beyond either frozen observation '
            "alone."
        ],
        "assumptions": [
            "The two frozen observations describe comparable stages of the same goal.",
            "Context remains fixed: service=ledger; consistency=strict",
        ],
        "proposed_test": (
            "Measure candidate intermediate states between the two "
            "observations, then intervene on the best predictor and compare "
            "against both frozen-source baselines."
        ),
        "evidence": [
            {
                "type": "snapshot_item",
                "id": str(_uuid(4_001)),
                "stable_evidence_key": _snapshot().items[0].stable_evidence_key,
            },
            {
                "type": "snapshot_item",
                "id": str(_uuid(4_002)),
                "stable_evidence_key": _snapshot().items[1].stable_evidence_key,
            },
        ],
    }
    assert {reference.id for reference in result.ideas[1].evidence} == {
        _uuid(4_001),
        _uuid(4_003),
    }
    conflict = result.ideas[1]
    assert "explicit_negator_conflict" in conflict.assumptions
    assert _snapshot().items[0].summary in conflict.predictions[0]
    assert _snapshot().items[2].summary in conflict.predictions[1]
    assert conflict.predictions[0] != conflict.predictions[1]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "shared_applicability",
    (("shared boundary",), ()),
)
async def test_conflict_predictions_distinguish_equal_summaries_by_mechanism(
    shared_applicability: tuple[str, ...],
) -> None:
    shared_summary = "The observed outcome is unchanged."
    left = _item(
        1,
        summary=shared_summary,
        mechanism="cache release",
        applicability=shared_applicability,
    )
    middle = _item(
        2,
        summary="An unrelated observation.",
        mechanism="queue acknowledgement",
        applicability=("queue is enabled",),
    )
    right = _item(
        3,
        summary=shared_summary,
        mechanism="not cache release",
        applicability=shared_applicability,
    )

    result = await DeterministicIdeaGenerator().generate(
        goal=GOAL,
        context=CONTEXT,
        frozen_items=(left, middle, right),
        operator=InspirationOperator.CAUSAL_GAP,
        branch_limit=3,
    )

    conflict = next(
        idea
        for idea in result.ideas
        if "explicit_negator_conflict" in idea.assumptions
    )
    assert left.mechanism in conflict.predictions[0]
    assert right.mechanism in conflict.predictions[1]
    assert "activate" in conflict.predictions[0].casefold()
    assert "activate" in conflict.predictions[1].casefold()
    assert conflict.predictions[0] != conflict.predictions[1]
    assert "revers" in conflict.falsifiers[0].casefold()
    assert "signed contrast" in conflict.proposed_test.casefold()
    assert left.mechanism in conflict.proposed_test
    assert right.mechanism in conflict.proposed_test


@pytest.mark.asyncio
async def test_colliding_abbreviated_conditions_fall_back_to_interventions() -> None:
    shared_prefix = "condition-" + "a" * 500
    shared_suffix = "z" * 500 + "-boundary"
    left_condition = f"{shared_prefix}LEFT{shared_suffix}"
    right_condition = f"{shared_prefix}RIGHT{shared_suffix}"
    left = _item(
        1,
        summary="The same observed outcome.",
        mechanism="cache release",
        applicability=(left_condition,),
    )
    middle = _item(
        2,
        summary="An unrelated observation.",
        mechanism="queue acknowledgement",
        applicability=("queue enabled",),
    )
    right = _item(
        3,
        summary="The same observed outcome.",
        mechanism="not cache release",
        applicability=(right_condition,),
    )

    result = await DeterministicIdeaGenerator().generate(
        goal=GOAL,
        context=CONTEXT,
        frozen_items=(left, middle, right),
        operator=InspirationOperator.CAUSAL_GAP,
        branch_limit=3,
    )

    conflict = next(
        idea
        for idea in result.ideas
        if "explicit_negator_conflict" in idea.assumptions
    )
    assert "activate" in conflict.predictions[0].casefold()
    assert "activate" in conflict.predictions[1].casefold()
    assert left.mechanism in conflict.proposed_test
    assert right.mechanism in conflict.proposed_test


@pytest.mark.asyncio
async def test_counterfactual_golden_uses_earliest_canonical_assumption() -> None:
    result = await DeterministicIdeaGenerator().generate(
        goal=GOAL,
        context=CONTEXT,
        frozen_items=_snapshot().items,
        operator=InspirationOperator.COUNTERFACTUAL,
        branch_limit=1,
    )

    assert result.ideas[0].model_dump(mode="json") == {
        "title": "Counterfactual: cache entry exists",
        "hypothesis": (
            'For goal "Prevent externally visible partial state", if it is '
            'not true that cache entry exists, the outcome "Cache commit '
            'barrier invalidation succeeds." should change in a way not '
            "explained by the original assumption."
        ),
        "mechanism": (
            'Invert the applicability assumption "cache entry exists" while '
            'holding the frozen mechanism "commit barrier cache invalidation" '
            "fixed."
        ),
        "predictions": [
            "Under the inverted assumption, the observed outcome differs "
            'from "Cache commit barrier invalidation succeeds.".',
            "Restoring the original assumption restores the prior outcome.",
        ],
        "falsifiers": [
            "The outcome remains unchanged across the original and inverted assumption."
        ],
        "assumptions": [
            "it is not true that cache entry exists",
            "All non-target conditions in the frozen evidence remain fixed.",
        ],
        "proposed_test": (
            "Run matched trials with the applicability assumption present and "
            "inverted; compare the outcome and restore it in a crossover trial."
        ),
        "evidence": [
            {
                "type": "snapshot_item",
                "id": str(_uuid(4_001)),
                "stable_evidence_key": _snapshot().items[0].stable_evidence_key,
            }
        ],
    }


@pytest.mark.asyncio
async def test_distant_analogy_golden_prefers_lowest_lexical_overlap() -> None:
    result = await DeterministicIdeaGenerator().generate(
        goal=GOAL,
        context=CONTEXT,
        frozen_items=_snapshot().items,
        operator=InspirationOperator.DISTANT_ANALOGY,
        branch_limit=1,
    )

    assert result.ideas[0].model_dump(mode="json") == {
        "title": (
            "Distant analogy: Queue acknowledges durable work. <-> "
            "Immune cells release cytokines."
        ),
        "hypothesis": (
            'For goal "Prevent externally visible partial state", the shared '
            "mechanism terms [durability, gate, release] transfer from "
            '"Queue acknowledges durable work." to "Immune cells release '
            'cytokines." despite low lexical overlap.'
        ),
        "mechanism": (
            "Map the shared mechanism [durability, gate, release] from "
            '"durability gate queue release" onto "durability gate immune '
            'release" without assuming the surrounding domains are equivalent.'
        ),
        "predictions": [
            "A perturbation of [durability, gate, release] produces "
            "directionally similar changes in both evidence domains.",
            "Domain-specific variables explain residual differences after the "
            "shared mechanism is controlled.",
        ],
        "falsifiers": [
            "Perturbing [durability, gate, release] affects only one evidence "
            "domain or produces opposite effects."
        ],
        "assumptions": [
            "Mapping limit: only the shared mechanism terms transfer; actors, "
            "scale, and boundary conditions do not.",
            "Context remains fixed: service=ledger; consistency=strict",
        ],
        "proposed_test": (
            "Apply matched perturbations to the shared mechanism in both "
            "domains and compare normalized response shapes while recording "
            "domain-specific failures."
        ),
        "evidence": [
            {
                "type": "snapshot_item",
                "id": str(_uuid(4_002)),
                "stable_evidence_key": _snapshot().items[1].stable_evidence_key,
            },
            {
                "type": "snapshot_item",
                "id": str(_uuid(4_004)),
                "stable_evidence_key": _snapshot().items[3].stable_evidence_key,
            },
        ],
    }


def test_conflict_detection_supports_falsifier_similarity_and_cjk_negation() -> None:
    falsifier_left = _item(
        1,
        summary="Primary observation",
        mechanism="write after commit",
        applicability=("a",),
        falsifiers=("release before durable acknowledgement",),
    )
    middle = _item(
        2,
        summary="Middle observation",
        mechanism="unrelated process",
        applicability=("b",),
    )
    falsifier_right = _item(
        3,
        summary="release before durable acknowledgement",
        mechanism="another mechanism",
        applicability=("c",),
    )
    assert evidence_conflicts(falsifier_left, falsifier_right) is True
    assert evidence_conflicts(falsifier_left, middle) is False

    positive = _item(
        1,
        summary="正向观察",
        mechanism="缓存 提交 失效",
        applicability=("条件甲",),
    )
    negative = _item(
        2,
        summary="反向观察",
        mechanism="不缓存 提交 失效",
        applicability=("条件乙",),
    )
    assert evidence_conflicts(positive, negative) is True


@pytest.mark.parametrize(
    ("lexical_word", "stripped_word"),
    (
        ("未来计划", "来计划"),
        ("无锡部署", "锡部署"),
        ("不丹节点", "丹节点"),
        ("非洲市场扩张", "洲市场扩张"),
        ("非常重要", "常重要"),
        ("无限重试", "限重试"),
        ("无花果发酵", "花果发酵"),
        ("未名湖监测", "名湖监测"),
    ),
)
def test_cjk_lexical_words_are_not_treated_as_explicit_negation(
    lexical_word: str,
    stripped_word: str,
) -> None:
    assert (
        evidence_conflicts(
            _item(
                1,
                summary="词汇观察",
                mechanism=lexical_word,
                applicability=("条件甲",),
            ),
            _item(
                2,
                summary="错误剥离观察",
                mechanism=stripped_word,
                applicability=("条件乙",),
            ),
        )
        is False
    )


def test_cjk_prefix_rule_distinguishes_productive_non_negation() -> None:
    assert evidence_conflicts(
        _item(
            1,
            summary="常驻观察",
            mechanism="常驻进程",
            applicability=("条件甲",),
        ),
        _item(
            2,
            summary="非常驻观察",
            mechanism="非常驻进程",
            applicability=("条件乙",),
        ),
    )


@pytest.mark.parametrize(
    ("positive", "negative"),
    (
        ("提交后缓存", "提交后不缓存"),
        ("常数函数", "非常数函数"),
        ("线性模型", "非线性模型"),
        ("典型数据", "非典型数据"),
        ("对称结构", "非对称结构"),
        ("常态", "非常态"),
        ("要素模型", "非要素模型"),
        ("要求字段", "非要求字段"),
        ("得分事件", "非得分事件"),
    ),
)
def test_cjk_productive_negator_may_appear_inside_an_unspaced_phrase(
    positive: str,
    negative: str,
) -> None:
    assert evidence_conflicts(
        _item(
            1,
            summary="正向观察",
            mechanism=positive,
            applicability=("条件甲",),
        ),
        _item(
            2,
            summary="反向观察",
            mechanism=negative,
            applicability=("条件乙",),
        ),
    )


@pytest.mark.parametrize(
    ("positive", "negative"),
    (
        ("缓存 提交", "缓存 不 提交"),
        ("缓存", "不 缓存"),
        ("缓存-提交", "缓存-不-提交"),
    ),
)
def test_standalone_cjk_negator_removal_normalizes_separators(
    positive: str,
    negative: str,
) -> None:
    left = _item(
        1,
        summary="正向观察",
        mechanism=positive,
        applicability=("条件甲",),
    )
    right = _item(
        2,
        summary="反向观察",
        mechanism=negative,
        applicability=("条件乙",),
    )

    assert evidence_conflicts(left, right)
    assert evidence_conflicts(right, left)


def test_jieba_dictionary_version_is_exactly_pinned() -> None:
    project = tomllib.loads(
        (Path(__file__).parents[2] / "pyproject.toml").read_text(
            encoding="utf-8"
        )
    )

    assert "jieba==0.42.1" in project["project"]["dependencies"]


@pytest.mark.parametrize(
    ("left_mechanism", "right_mechanism"),
    (
        ("not not cache", "not cache"),
        ("不不缓存", "不缓存"),
        ("非非线性模型", "非线性模型"),
        ("不非线性模型", "非线性模型"),
    ),
)
def test_explicit_negator_conflict_requires_exactly_one_negated_side(
    left_mechanism: str,
    right_mechanism: str,
) -> None:
    assert (
        evidence_conflicts(
            _item(
                1,
                summary="左侧否定观察",
                mechanism=left_mechanism,
                applicability=("条件甲",),
            ),
            _item(
                2,
                summary="右侧否定观察",
                mechanism=right_mechanism,
                applicability=("条件乙",),
            ),
        )
        is False
    )


@pytest.mark.parametrize(
    ("lexical_word", "stripped_word"),
    (
        ("非典数据", "典数据"),
        ("非要缓存", "要缓存"),
        ("非得缓存", "得缓存"),
    ),
)
def test_fused_cjk_lexical_words_are_not_productive_negation(
    lexical_word: str,
    stripped_word: str,
) -> None:
    assert (
        evidence_conflicts(
            _item(
                1,
                summary="词汇观察",
                mechanism=lexical_word,
                applicability=("条件甲",),
            ),
            _item(
                2,
                summary="错误剥离观察",
                mechanism=stripped_word,
                applicability=("条件乙",),
            ),
        )
        is False
    )


def test_conflict_detection_locks_threshold_and_negator_boundaries() -> None:
    common = tuple(f"shared{index}" for index in range(13))
    at_threshold = _item(
        1,
        summary="left",
        mechanism="notice latency",
        applicability=("a",),
        falsifiers=(" ".join((*common, *(f"only{index}" for index in range(7)))),),
    )
    threshold_match = _item(
        2,
        summary=" ".join(common),
        mechanism="latency",
        applicability=("b",),
    )
    assert evidence_conflicts(at_threshold, threshold_match) is True

    below_common = common[:-1]
    below_threshold = _item(
        1,
        summary="left",
        mechanism="notice latency",
        applicability=("a",),
        falsifiers=(
            " ".join((*below_common, *(f"below-only{index}" for index in range(7)))),
        ),
    )
    below_match = _item(
        2,
        summary=" ".join(below_common),
        mechanism="latency",
        applicability=("b",),
    )
    assert evidence_conflicts(below_threshold, below_match) is False
    assert (
        evidence_conflicts(
            _item(
                1,
                summary="left",
                mechanism="notice latency",
                applicability=("a",),
            ),
            _item(
                2,
                summary="right",
                mechanism="latency",
                applicability=("b",),
            ),
        )
        is False
    )


def test_conflict_detection_is_bounded_at_maximum_mechanism_length() -> None:
    items = tuple(
        _item(
            rank,
            summary=f"最大长度观察 {rank}",
            mechanism=("不" * (1_990 - rank) + f"机制边界{rank:02d}")[:2_000],
            applicability=(f"条件 {rank}",),
        )
        for rank in range(1, 13)
    )

    started = perf_counter()
    pairs = causal_gap_pairs(items)
    elapsed = perf_counter() - started

    assert len(pairs) == 11
    assert elapsed < 10.0
    assert (
        evidence_conflicts(
            _item(
                1,
                summary="left",
                mechanism="no latency",
                applicability=("a",),
            ),
            _item(
                2,
                summary="right",
                mechanism="latency",
                applicability=("b",),
            ),
        )
        is True
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "operator",
    (
        InspirationOperator.CAUSAL_GAP,
        InspirationOperator.COUNTERFACTUAL,
        InspirationOperator.DISTANT_ANALOGY,
    ),
)
async def test_branch_limit_returns_a_stable_candidate_prefix(
    operator: InspirationOperator,
) -> None:
    generator = DeterministicIdeaGenerator()
    complete = await generator.generate(
        goal=GOAL,
        context=CONTEXT,
        frozen_items=_snapshot().items,
        operator=operator,
        branch_limit=3,
    )

    for branch_limit in (1, 2):
        limited = await generator.generate(
            goal=GOAL,
            context=CONTEXT,
            frozen_items=_snapshot().items,
            operator=operator,
            branch_limit=branch_limit,
        )
        assert limited.ideas == complete.ideas[:branch_limit]


@pytest.mark.asyncio
async def test_distant_analogy_supports_cjk_shared_mechanism_terms() -> None:
    items = (
        _item(
            1,
            summary="温控系统保持稳定",
            mechanism="反馈控制",
            applicability=("温度超过阈值",),
            tags=("工程",),
        ),
        _item(
            2,
            summary="团队根据复盘改变计划",
            mechanism="反馈调节",
            applicability=("迭代结束",),
            tags=("组织",),
        ),
    )

    result = await DeterministicIdeaGenerator().generate(
        goal="寻找跨领域反馈规律",
        context="",
        frozen_items=items,
        operator=InspirationOperator.DISTANT_ANALOGY,
        branch_limit=1,
    )

    assert result.error_code is None
    assert "反馈" in result.ideas[0].mechanism


@pytest.mark.asyncio
async def test_generated_ideas_remain_valid_at_input_length_limits() -> None:
    left_summary = _marked_text(
        "LEFT-SUMMARY-START",
        "LEFT-SUMMARY-END",
        length=1_000,
        fill="A",
    )
    right_summary = _marked_text(
        "RIGHT-SUMMARY-START",
        "RIGHT-SUMMARY-END",
        length=1_000,
        fill="B",
    )
    left_mechanism = _marked_text(
        "LEFT-MECHANISM-START",
        "LEFT-MECHANISM-END",
        length=2_000,
        fill="a",
    )
    right_mechanism = _marked_text(
        "RIGHT-MECHANISM-START",
        "RIGHT-MECHANISM-END",
        length=2_000,
        fill="b",
    )
    applicability = _marked_text(
        "APPLICABILITY-START",
        "APPLICABILITY-END",
        length=4_000,
        fill="c",
    )
    items = (
        _item(
            1,
            summary=left_summary,
            mechanism=left_mechanism,
            applicability=(applicability,),
        ),
        _item(
            2,
            summary=right_summary,
            mechanism=right_mechanism,
            applicability=("right condition",),
        ),
    )

    causal = await DeterministicIdeaGenerator().generate(
        goal="G" * 2_000,
        context="C" * 4_000,
        frozen_items=items,
        operator=InspirationOperator.CAUSAL_GAP,
        branch_limit=1,
    )

    assert causal.error_code is None
    causal_idea = causal.ideas[0]
    for marker in (
        "LEFT-SUMMARY-START",
        "LEFT-SUMMARY-END",
        "RIGHT-SUMMARY-START",
        "RIGHT-SUMMARY-END",
        "LEFT-MECHANISM-START",
        "LEFT-MECHANISM-END",
        "RIGHT-MECHANISM-START",
        "RIGHT-MECHANISM-END",
    ):
        assert marker in (
            causal_idea.title
            + causal_idea.hypothesis
            + causal_idea.mechanism
            + " ".join(causal_idea.predictions)
        )
    assert causal_idea.mechanism.endswith("measurable intermediate state.")

    counterfactual = await DeterministicIdeaGenerator().generate(
        goal="G" * 2_000,
        context="C" * 4_000,
        frozen_items=(items[0],),
        operator=InspirationOperator.COUNTERFACTUAL,
        branch_limit=1,
    )
    counterfactual_idea = counterfactual.ideas[0]
    for marker in (
        "APPLICABILITY-START",
        "APPLICABILITY-END",
        "LEFT-SUMMARY-START",
        "LEFT-SUMMARY-END",
        "LEFT-MECHANISM-START",
        "LEFT-MECHANISM-END",
    ):
        assert marker in (
            counterfactual_idea.title
            + counterfactual_idea.hypothesis
            + counterfactual_idea.mechanism
            + " ".join(counterfactual_idea.predictions)
            + " ".join(counterfactual_idea.assumptions)
        )
    assert counterfactual_idea.mechanism.endswith("fixed.")

    analogy_items = (
        items[0].model_copy(update={"mechanism": "shared " + left_mechanism[7:]}),
        items[1].model_copy(update={"mechanism": "shared " + right_mechanism[7:]}),
    )
    analogy = await DeterministicIdeaGenerator().generate(
        goal="G" * 2_000,
        context="C" * 4_000,
        frozen_items=analogy_items,
        operator=InspirationOperator.DISTANT_ANALOGY,
        branch_limit=1,
    )
    analogy_idea = analogy.ideas[0]
    for marker in (
        "LEFT-MECHANISM-END",
        "RIGHT-MECHANISM-END",
    ):
        assert marker in analogy_idea.mechanism
    assert analogy_idea.mechanism.endswith("domains are equivalent.")

    for idea in (causal_idea, counterfactual_idea, analogy_idea):
        assert len(idea.title) <= 1_000
        assert len(idea.mechanism) <= 2_000
        assert len(idea.hypothesis) <= 4_000


def test_generator_signature_exposes_only_the_frozen_generation_boundary() -> None:
    parameters = signature(DeterministicIdeaGenerator.generate).parameters

    assert tuple(parameters) == (
        "self",
        "goal",
        "context",
        "frozen_items",
        "operator",
        "branch_limit",
        "output_token_limit",
    )
    assert parameters["output_token_limit"].default == 1_200
    assert parameters["self"].kind is Parameter.POSITIONAL_OR_KEYWORD
    assert all(
        parameter.kind is Parameter.KEYWORD_ONLY
        for name, parameter in parameters.items()
        if name != "self"
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("operator", "items"),
    (
        (
            InspirationOperator.CAUSAL_GAP,
            (
                _item(
                    1,
                    summary="one",
                    mechanism="same mechanism",
                    applicability=(),
                ),
            ),
        ),
        (
            InspirationOperator.COUNTERFACTUAL,
            (
                _item(
                    1,
                    summary="one",
                    mechanism="same mechanism",
                    applicability=(),
                ),
            ),
        ),
        (
            InspirationOperator.DISTANT_ANALOGY,
            (
                _item(
                    1,
                    summary="one",
                    mechanism="first mechanism",
                    applicability=("a",),
                ),
                _item(
                    2,
                    summary="two",
                    mechanism="second process",
                    applicability=("b",),
                ),
            ),
        ),
    ),
)
async def test_structurally_missing_evidence_returns_fixed_failure(
    operator: InspirationOperator,
    items: tuple[SnapshotItem, ...],
) -> None:
    result = await DeterministicIdeaGenerator().generate(
        goal=GOAL,
        context="",
        frozen_items=_snapshot(items).items,
        operator=operator,
        branch_limit=3,
    )

    assert result == GeneratorResult(
        ideas=(),
        error_code=OperatorFailureCode.INSUFFICIENT_EVIDENCE,
        output_tokens_consumed=0,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("branch_limit", (0, 4, True, 1.0, "1"))
async def test_generator_rejects_non_strict_branch_limits(
    branch_limit: object,
) -> None:
    with pytest.raises(ValueError, match="branch_limit"):
        await DeterministicIdeaGenerator().generate(
            goal=GOAL,
            context=CONTEXT,
            frozen_items=_snapshot().items,
            operator=InspirationOperator.CAUSAL_GAP,
            branch_limit=branch_limit,  # type: ignore[arg-type]
        )


def test_generator_result_never_represents_an_empty_success() -> None:
    with pytest.raises(ValueError, match="successful generator result"):
        GeneratorResult(
            ideas=(),
            error_code=None,
            output_tokens_consumed=0,
        )

    with pytest.raises(ValueError, match="failed generator result"):
        GeneratorResult(
            ideas=(_expected_minimal_idea(),),
            error_code=OperatorFailureCode.INSUFFICIENT_EVIDENCE,
            output_tokens_consumed=0,
        )


def _expected_minimal_idea() -> IdeaDraft:
    item = _snapshot().items[0]
    return IdeaDraft(
        title="idea",
        hypothesis="hypothesis",
        mechanism="mechanism",
        predictions=("prediction",),
        falsifiers=("falsifier",),
        assumptions=("assumption",),
        proposed_test="test",
        evidence=(
            SnapshotEvidenceReference(
                id=item.snapshot_item_id,
                stable_evidence_key=item.stable_evidence_key,
            ),
        ),
    )
