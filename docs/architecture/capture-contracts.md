# Capture and Candidate Contracts

Experience Hub 的第一阶段捕获流程把已清理的 agent 轨迹转换成候选经验，
但不会自动把候选写入普通经验。流程在本地 SQLite 中执行，默认不访问网络、
不读取模型密钥，也不调用外部工具；来源、事件、幂等 receipt 和可重建
projection 为后续审计提供依据。

## Trust boundary

输入方必须在边界外完成数据最小化和清理，并在 header 中显式声明
`sanitization.input_sanitized=true`。内置 scanner 会拒绝一组已知密钥、令牌、
认证头和私钥模式，并只报告 rule、step 和 field 位置，不回显匹配内容。

scanner 没有发现匹配项，不证明输入已经完成 sanitization，也不证明输入适合
公开、共享或长期保存。它是 fail-closed 的补充检查，不是隐私审计、恶意内容
检测或通用 DLP 系统。

## Generic JSONL v1

当前只接受 canonical UTF-8 Generic JSONL v1：

- 第一行是 header，包含 adapter 版本、owner UUID、轨迹标识、来源时间范围和
  sanitization 声明。
- 后续每行是按正整数 `ordinal` 排序的 step，包含 step 标识、带时区时间、
  status，以及必填的 observation、action、outcome；只有 `candidate_signal`
  可以省略。
- `candidate_signal` 必须给出严格的经验 kind、body、summary、mechanism、tags、
  applicability、falsifiers 和指向现有 step 字段的 evidence 引用。
- 重复键、非 canonical JSON、无效 Unicode、越界字段、错误 owner、歧义顺序或
  损坏 evidence 都会在写入前被拒绝。

已提交的合成示例位于
[`examples/trajectories/coding-agent-recovery.jsonl`](../../examples/trajectories/coding-agent-recovery.jsonl)。
它不包含真实用户、真实工作区或人类参与者数据。

## What is retained

成功 import 会保留 canonical manifest、轨迹与 step 的结构化元数据、候选内容、
候选内容哈希、extractor 配置哈希，以及候选明确引用的有界 evidence excerpt。
每条 evidence 分别保留完整来源字段哈希与实际 excerpt 哈希：前者锚定 manifest
中的完整字段，后者让截断后保留的 UTF-8 文本也可独立校验。
完整原始 observation、action、outcome 和原始 JSONL 字节不会作为轨迹副本保留。

因此，未被 evidence 引用的原始字段不应出现在任何 SQLite TEXT/BLOB 列中；
这条负向不变量由验收测试扫描全部 SQLite 表验证。被候选内容或 evidence 明确
引用的文本属于有意保留的数据，仍需由输入方在导入前完成清理。

## Candidate quarantine

新候选从 `pending` 开始。`pending` 和 `rejected` 候选不会产生普通
`experiences`、`experience_terms`、可发布 experience version 或 inspiration
experience snapshot item，也不会被普通 RetrievalService/search 返回。

候选列表和详情始终先按 owner 过滤。另一个 owner 不能通过详情、分页、筛选、
计数或错误差异获知候选是否存在。

## Explicit adoption and rejection

`candidates adopt` 是候选进入普通经验存储的唯一已交付路径。它创建或复用同一
owner 当前内容等价的经验，并写入恰好一条 candidate-to-experience-version
lineage。只有 resulting experience 会进入普通检索、experience terms、分享
发布前检查和 inspiration snapshot。

`candidates reject` 记录结构化原因和终态事件，但不创建经验。终态候选不能用
另一个 key 改写决定。相同 owner、操作、请求和 idempotency key 会逐字节重放
原响应，不重复 lineage、事件或 projection 变更。

## Owner isolation

capture header 的 owner 必须与命令 caller、路径 owner 和已存在 agent 一致。
候选 source、state、evidence、决策、lineage 和 resulting experience 的查询与
写入都使用 owner 约束。缺失、foreign owner 和伪造 caller 对外得到相同的稳定
not-found 结果。

SQLite 是本地权威存储；owner 隔离是应用与数据库契约，不是操作系统级多租户
边界。使用者仍需限制数据库文件访问权限，并且不能把没有认证的服务直接暴露
到不可信网络。

## Determinism and idempotency

`capture inspect` 不打开数据库。`capture import` 以 canonical manifest 和确定性
signal extractor 生成候选；同一 fixture 的 manifest hash 与 candidate content
hash 可重复得到相同结果。写命令要求显式 idempotency key，receipt 绑定 caller、
精确 operation scope 和 canonical request hash。

来源与事件不可变，candidate、experience terms 和其他 projection 可从 ledger
重建。`projections rebuild --verify` 只比较；`--repair` 在独占事务中原子替换。
重放或修复不得使 pending/rejected 候选进入普通路径，也不得复制 adoption
lineage。

## Known limitations

- 只实现 Generic JSONL v1，没有其他轨迹 adapter。
- 只实现显式 `candidate_signal` 的 deterministic signal extractor；没有模型
  extractor。
- 捕获和候选操作只提供本地 CLI；没有 capture/candidate HTTP routes。
- 没有 Replay Lab，也没有轨迹回放执行器。
- 没有使用人类参与者或真实用户数据进行有效性实验。
- scanner success 不证明数据已清理，也不构成安全或隐私保证。
- raw trajectory 不保留；该流程不是通用日志归档。
- 对外发布仍需刻意决定 license；当前实现不能替代该发布决策。
