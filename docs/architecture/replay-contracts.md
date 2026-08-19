# Replay contracts

Experience Replay Lab 是一个本地、离线、CLI-first 的实验边界。它从已经关闭且
校验过的 SQLite source 创建互相独立的 clone，在固定的 case、clock、seed 和
policy arm 下运行两次，然后发布 canonical evidence。它不会迁移或写回 source，
也不会把 replay 结果自动采纳为经验。

当前提交的 `smoke-replay` 只有两个合成 case 和两个 arm。它验证 isolation、
report validation 和 repeatability；它不是 ExperienceBench-S、human study，
也不是一般有效性或安全性的证据。

## Manifest and dataset closure

manifest 必须是 UTF-8 canonical JSON，严格匹配 `ReplayManifestV1`：

| 字段 | 契约 |
|---|---|
| `schema_version` | 当前必须是 `1` |
| `experiment_id` | 稳定的小写 replay label |
| `dataset` | versioned dataset descriptor；见下表 |
| `snapshot_binding` | 当前必须是 `validated_source` |
| `frozen_at` | canonical UTC timestamp；作为注入 clock |
| `seed` | 非负确定性 seed |
| `arms` | 按顺序列出 `no_memory`、`experience_hub`，两者都必须 `required=true` |
| `oracle` | 当前是 version `1` 的 `retrieval_labels` |
| `evidence_schema_version` | 当前必须是 `1` |
| `profile_schema_version` | 当前必须是 `1` |
| `deterministic_replay_runs` | 当前必须是 `2` |

`dataset` 包含：

| 字段 | 契约 |
|---|---|
| `schema_version` | 当前必须是 `1` |
| `dataset_id` | 稳定的小写 dataset label |
| `cases_file` | 与 manifest 同目录的一个普通文件名；不允许目录或 traversal |
| `cases_sha256` | 对 JSONL 精确 bytes 的 lowercase SHA-256 |

JSONL 必须以一个换行结尾，每个非空行都是一个 canonical `ReplayCaseV1`。
loader 先对完整 bytes 验证 `cases_sha256`，再解释 case；因此 manifest 与 case
文件构成 hash-closed input。每个 case 固定 owner、query、retrieval mode、tags、
mechanism cues、limit、content budget、cold expansion，以及 expected/forbidden
logical label 到 experience UUID 的映射。case ID、label 和映射都必须满足唯一性
与 closure 约束。

## Closed source and clone isolation

`--database` 是 caller 已经关闭、checkpoint 且不再写入的 file-backed SQLite
source。replay 会拒绝 symlink、非普通文件、非空 `-wal`、`-shm` 或 `-journal`、
不受支持的 schema、无效 authoritative source 或 projection mismatch。验证只在
disposable clone 上启动 runtime，且不会为旧 source 运行 migration。

source main-file bytes 在 freeze 时绑定 `snapshot_sha256`，在 clone 和 artifact
发布前重复核对。每个 case/arm/pass 都从该 frozen source 创建独立 clone；policy
只能得到自己的 clone path。source 发生变化、出现新 sidecar、clone identity
复用、clone hash 不符或 projection 校验失败都会 fail closed。

这个边界不承诺跨进程的任意敌对文件系统隔离。workspace 与 clone publication
使用 portable descriptor checks 和 advisory locks 协调遵守协议的 Experience Hub
进程；故意忽略 lock 并在 syscalls 之间交换名字的同 UID 恶意进程不在该 portable
threat model 内。

snapshot publication 的已知恢复限制也需由运行方保留：创建 private backup 后
若重新打开它失败，可能留下可识别的 backup 供人工恢复；descriptor close 错误
可能把本已成功的操作报告为失败；如果后续 rollback step 自身失败，严格 rollback
会停止并保留失败，而不是假装完成 best-effort recovery。

## Workspace ownership and replacement

replay workspace 由根目录中的
`.experience-hub-replay-workspace` marker 标记，marker bytes 是
`experience-hub replay workspace v1\n`。当前 policy 只拥有四个 top-level
entry：`snapshot`、`validation`、`arms` 和 `artifacts`。

首次运行只会采纳新目录或显式允许的空目录。已有有效 workspace 默认拒绝覆盖。
`replay run --replace-owned` 会在 marker、root identity 和全部 top-level entries
通过校验后，只替换上述 owned entries。marker 缺失或损坏、未知 entry、symlink
或 special node 都会保留原数据并拒绝运行；该 flag 不是递归删除任意目录的开关。

## Required arms and incomplete comparisons

当前两 arm 都是 required：

- `no_memory` 返回空 observation，且不打开 clone；
- `experience_hub` 在自己的 disposable clone 上执行 owner-scoped read-only
  retrieval，并只把 manifest 声明的 logical labels 交给 oracle。

policy exception 记录为 `arm_infrastructure_failure`；无效 oracle output 记录为
`oracle_validation_failure`。任一 required arm 失败时，case 为 `incomplete`，
`delta_utility_micros` 必须是 `null`。任一 case incomplete 时，整个
`comparison_complete=false`、`valid=false`，CLI 以非零状态退出；系统不会从剩余
arm 计算有利的 effect。

## Evidence, profiles, and verification

`artifacts/evidence.json` 是 bounded canonical JSON。它包含 resolved manifest
hashes、source schema revision、case/arm outcome、utility、delta、comparison
closure、source/clone checks 和两次 replay 的 byte-identity 结果。它排除 wall
duration、database path、credentials、raw UUID 和其他 runtime-only values。

standalone `replay verify` 能证明 evidence schema、canonical encoding 和报告内部
的 case/arm/completeness closure，也能核对报告中 hash 字段的形状。它不能从单个
artifact 证明未嵌入的 manifest、JSONL 或 SQLite preimage。`replay run` 才会在
加载时绑定 manifest/case 精确 bytes，在 freeze 时绑定 source bytes，并在发布前
再次核对这些 preimages。

`artifacts/profile.json` 单独保存 `wall_duration_ns` 和 `database_bytes`。profile
不参与 evidence score 或 byte-identical comparison；profile 收集或写入失败会让
process-facing run 不完整，但不会改写 canonical score evidence。

每个 artifact 都以同目录 temporary file 单独原子替换，但两个文件不是 cross-file
transaction。发布顺序是 profile first、evidence last：profile write 失败会保留
先前 evidence；evidence write 失败可能留下新的 profile 与先前 evidence 并存。
消费者应分别验证 evidence 和 profile，不应把“两个文件同时存在”当作事务证明。

## CLI

```bash
uv run experience-hub replay inspect \
  --manifest examples/replay/smoke-manifest.json \
  --database .data/demo.db

uv run experience-hub replay run \
  --manifest examples/replay/smoke-manifest.json \
  --database .data/demo.db \
  --workspace .data/replay-lab

uv run experience-hub replay verify \
  --report .data/replay-lab/artifacts/evidence.json
```

`inspect` 验证 manifest、dataset 和 closed source，但不保留 arm clones。`run`
创建 clones 和 artifacts。`verify` 只验证给定 evidence artifact。所有成功和失败
都输出一行 canonical JSON；稳定错误不会包含 source path、workspace path、SQL、
provider output 或 credentials。
