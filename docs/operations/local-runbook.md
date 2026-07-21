# Experience Hub 本地运维手册

本文面向单机、文件型 SQLite 部署。所有命令都假定当前目录是
Experience Hub 仓库根目录，依赖已经安装到 `.venv`，并使用
`.venv/bin/experience-hub`。一次性 CLI 命令的标准输出是单行规范 JSON；
`serve` 的 access log 写到 stdout，startup/shutdown 与异常诊断写到 stderr。
程序化 migration 保留宿主 logging 配置，不保证输出 Alembic INFO；宿主如显式
配置 Alembic logger，其记录进入宿主指定的 handler。自动化应按入口采集，且不要
合并两条流。

## Operations

### 1. 数据文件与运行边界

默认数据库是仓库根目录下的：

```text
.data/experience_hub.db
```

运行时会创建父目录、把受支持的旧 schema 自动迁移到当前 Alembic head，并让
SQLite 使用 WAL。主库旁边可能出现以下 sidecar：

```text
.data/experience_hub.db-wal
.data/experience_hub.db-shm
.data/experience_hub.db-journal
```

这些文件是数据库状态的一部分。服务运行时绝不能只复制主 `.db` 文件作为
backup，因为已提交的数据可能仍只在 `-wal` 中。

不同入口的文件边界如下：

| 用途 | 默认位置 | 说明 |
| --- | --- | --- |
| HTTP 服务及普通运维 | `.data/experience_hub.db` | `serve`、生命周期、投影和 payload 命令的默认库 |
| 确定性演示 | `.data/demo.db` | 与默认库隔离；`demo --reset` 只重置这个库及其 sidecar |
| 效果基准工作区 | `.data/benchmark/` | 每次运行重建受控的 snapshot、`replay-a` 和 `replay-b` clones |
| 基准输入 | `benchmarks/seed.json`、`benchmarks/cases.jsonl` | 版本控制中的测试资产，不是可删除的工作 clone |

下列命令支持 `--database /absolute/or/relative/path.db`：

```text
experience-hub serve
experience-hub lifecycle run
experience-hub projections rebuild
experience-hub payloads reconcile
```

`--database` 接受文件路径，不接受 SQLite URL、URL query 或 `:memory:`。`demo`
和 `benchmark` 使用上述固定隔离位置，没有 `--database` 参数。

首次用新版本打开现有数据库之前应先做停机 backup。任何普通 CLI 命令也会先
迁移 schema，因此不要把“先运行 verify”误认为迁移前备份。

### 2. 启动、健康检查与停止

前台启动本地服务：

```bash
.venv/bin/experience-hub serve --host 127.0.0.1 --port 8765
```

指定独立数据库：

```bash
.venv/bin/experience-hub serve \
  --host 127.0.0.1 \
  --port 8765 \
  --database .data/experience_hub.db
```

默认只绑定 `127.0.0.1`。若要绑定其他接口，应先在外层配置访问控制、TLS 和
进程管理；应用自身不应被当作公网认证边界。

就绪探针：

```bash
curl --fail --silent --show-error http://127.0.0.1:8765/health
```

成功响应的 `data.status` 是 `ready`，并包含应用版本、当前 schema revision
和每个 projection reducer 的版本。服务只有在完成 schema 迁移、权威源校验、
中断 inspiration run 恢复并启动后台生命周期 worker 后才会 ready。未就绪时
返回 `503 service_not_ready`。

使用 `Ctrl-C` 或向 Uvicorn 进程发送 `SIGTERM` 做优雅停止。关闭过程先停止后台
生命周期 worker，再等待有界的 inspiration run 到各自 deadline；超期任务会被
取消，最后才释放 SQLite engine。做原始数据库复制、投影 repair 或恢复时，必须
先等待进程完全退出。

### 3. Provider：默认离线与显式 opt-in

默认 inspiration generator 是 `deterministic`，不需要网络、模型或密钥。
即使机器上存在任意 provider 环境变量，只要请求没有显式选择
`"generator": "openai_compatible"`，就不会创建 HTTP client。

当前 `experience-hub serve` 命令也不会自动读取 provider 环境变量，并且没有
provider CLI 参数。因此外部 provider 必须同时满足两个显式条件：

1. 嵌入应用时把完整配置传给 `Settings`；
2. 单次 inspiration 请求明确选择 `openai_compatible`。

可以建立一个不纳入版本控制的 `.local/provider_app.py`：

```python
import os

from experience_hub.api.app import create_app
from experience_hub.config import Settings

app = create_app(
    settings=Settings(
        openai_compatible_base_url=os.environ[
            "EXPERIENCE_HUB_OPENAI_BASE_URL"
        ],
        openai_compatible_model=os.environ[
            "EXPERIENCE_HUB_OPENAI_MODEL"
        ],
        openai_compatible_api_key=os.environ[
            "EXPERIENCE_HUB_OPENAI_API_KEY"
        ],
    )
)
```

再由受信任的 secret store 注入这三个变量并启动：

```bash
.venv/bin/python -m uvicorn \
  --app-dir .local \
  provider_app:app \
  --host 127.0.0.1 \
  --port 8765
```

这里的环境变量名是上述本地 wrapper 显式读取的名称，不是 `Settings` 自动发现
机制。base URL 必须是无 user-info、query、fragment 的绝对 HTTP(S) URL；
model 必须是规范、已 trim 的标识；API key 必须是非空可打印 ASCII。缺少或非法
配置时，显式选择 provider 的请求返回 `422 generator_not_configured`，而服务
启动和离线确定性请求仍然可用。

不要把 API key 写进数据库、请求体、仓库或命令行。持久化的 generator 配置只
包含非秘密的 base URL 和 model。`.env*`、`.secrets/`、`.local/` 等本地材料已
由 `.gitignore` 排除，但仍应限制文件权限和日志读取权限。

### 4. SQLite backup

#### 4.1 在线一致性 backup

服务不能停机时，优先使用 SQLite backup API，而不是 `cp` 主库。backup API
读取一个一致性快照并正确纳入 WAL 中已提交的页。备份期间不要同时部署新 schema
或执行 projection repair。

```bash
DB=.data/experience_hub.db
BACKUP_DIR=.data/backups
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
BACKUP="${BACKUP_DIR}/experience_hub-${STAMP}.db"

mkdir -p "${BACKUP_DIR}"
sqlite3 "${DB}" ".timeout 5000" ".backup '${BACKUP}'"
sqlite3 "${BACKUP}" "PRAGMA integrity_check;"
shasum -a 256 "${BACKUP}" > "${BACKUP}.sha256"
```

`PRAGMA integrity_check` 必须输出 `ok`。`.data/backups` 只是在本机的操作缓冲区；
还应把 backup 与校验文件复制到受控、加密、异机的存储。

#### 4.2 停机后的原始文件快照

需要字节级主库副本时：

1. 优雅停止所有 Experience Hub 进程和其他 SQLite writer；
2. 对 WAL 做 `TRUNCATE` checkpoint；
3. 确认 `-wal` 不存在或为空；
4. 才复制主库。

```bash
DB=.data/experience_hub.db
SNAPSHOT=.data/backups/experience_hub-offline.db

mkdir -p "$(dirname "${SNAPSHOT}")"
sqlite3 "${DB}" \
  "PRAGMA busy_timeout=5000; PRAGMA wal_checkpoint(TRUNCATE);"
test ! -s "${DB}-wal"
sqlite3 "${DB}" "PRAGMA integrity_check;"
cp -p "${DB}" "${SNAPSHOT}"
sqlite3 "${SNAPSHOT}" "PRAGMA integrity_check;"
```

完全 checkpoint 后通常输出 `0|0|0`，依次表示未 busy、WAL 剩余 frame 为零、
需 checkpoint 的 frame 为零。若第一列非零或 `-wal` 仍非空，说明仍有连接阻止
checkpoint；不要继续复制，先找出并停止连接后重试。不要用“同时复制 `.db` 和
正在变化的 `-wal`”替代一致性 backup。

### 5. Projection verify 与 repair

projection 是权威 event/source 的可重放读模型；event ledger、不可变 experience
version/payload 和 provenance source 不是 projection。repair 只能重建读模型，
不能修复损坏的权威源。

只读验证：

```bash
.venv/bin/experience-hub projections rebuild --verify \
  --database .data/experience_hub.db
```

成功时退出码为 0，`data.matches` 为 `true`，`differences` 为空。verify 在临时表
重放到固定 event head，比较 reducer version、checkpoint 和规范化行 hash，不会
替换在线 projection。发现差异时退出码为 1，并以
`projection_mismatch` 返回 projection 名称、online/rebuilt hash 和有限数量的
`differing_keys`。

修复前先停止 HTTP 服务和其他 writer，并完成 backup：

```bash
.venv/bin/experience-hub projections rebuild --repair \
  --database .data/experience_hub.db

.venv/bin/experience-hub projections rebuild --verify \
  --database .data/experience_hub.db
```

repair 使用独占事务，先校验全部权威源，再重放、原子替换 projection 并校验替换
后的 hash。下列情况会 fail closed：

- `source_integrity_error`：权威 ledger/source 不一致；不要 repair，保存现场并从
  已验证 backup 恢复；
- `maintenance_blocked_by_inflight`：存在 `in_progress` command receipt；等待
  有界任务结束或优雅停止服务后重试；
- `database_busy`：仍有连接持锁；响应建议 5 秒后重试，但计划维护应先停 writer；
- `event_head_changed`：维护期间 event head 发生变化；停止 writer 后重新执行；
- `reducer_version_mismatch` 或 `schema_version_unsupported`：运行代码与数据库不
  兼容；使用匹配版本或受支持的迁移，不要手改版本表。

服务启动只验证权威源与 reducer 兼容性，不做完整 projection equality replay；
因此应把 `--verify` 放入发布后和定期维护检查。

### 6. Payload reconcile

正常温度 transition 会在同一事务中把该经验所有版本的物理 payload codec
调整到当前偏好，同时保持语义 hash 不变：`hot`/`warm` 使用 plain，
`cold`/`archived` 使用 zlib。reconcile 是独立的审计/恢复入口，用于升级后的
历史数据或怀疑存在物理 codec 漂移时，不需要在每次正常 transition 后例行执行。

```bash
.venv/bin/experience-hub payloads reconcile \
  --database .data/experience_hub.db
```

命令在 immediate transaction 中先检查所有 experience version。成功响应包含：

- `changed_count`：实际转换 codec 的版本数；
- `skipped_count`：已经使用首选 codec 的版本数；
- `error_count` 与 `errors`：带 version identity 的安全诊断。

任何 preflight 或 rewrite validation 错误都会退出非零并避免部分提交。
`payloads reconcile` 只能对语义完好的 canonical payload 重新编码，不能猜测或
重造缺失/损坏的内容。`missing_identity`、`missing_payload`、`missing_state` 或
`semantic_validation_failed` 应按权威源损坏处理：停止写入、保留现场并恢复
backup，而不是直接修改表或 hash。

### 7. 手工 lifecycle

HTTP 服务启动后会每 15 分钟触发一次后台生命周期 worker。需要可审计的单次执行
时使用普通 command executor 的手工入口：

```bash
.venv/bin/experience-hub lifecycle run \
  --database .data/experience_hub.db \
  --idempotency-key ops-lifecycle-2026-07-19T020000Z
```

省略 `--evaluated-at` 时，评估时间固定为这条持久化 receipt 的创建时间。若需要
重放一个明确时点，传 RFC 3339、带时区且不晚于当前系统时钟的时间：

```bash
.venv/bin/experience-hub lifecycle run \
  --database .data/experience_hub.db \
  --evaluated-at 2026-07-19T02:00:00Z \
  --idempotency-key ops-lifecycle-2026-07-19T020000Z
```

idempotency key 必须是 1–128 个非空字符。同一个 key 与完全相同的请求会返回原
结果，不会重复产生 transition；不要拿同一个 key 改时间或请求内容。成功结果
包含 `cycle_id`、`evaluated_at`、`evaluated_count`、`transition_count`、
`archive_count` 和 `idea_archive_count`。

默认策略具有 15 分钟最小评估间隔，降温需要连续两个符合阈值的有效 cycle。
pin、重要性、置信度、访问强度和活动依赖会阻止部分降温/归档；强上下文 cue 可把
cold 经验重新激活为 warm。正常 lifecycle transition 已原子同步物理 codec；
只有升级或怀疑历史 codec 漂移时才需要额外执行 `payloads reconcile`。发布后的
完整一致性检查应再执行 projection verify。

若返回 `lifecycle_in_progress`，表示另一个 lifecycle lease 仍有效；等待当前
cycle 结束后重试。若返回 `database_busy`，不要并发堆叠手工 cycle。

### 8. 确定性 demo reset

演示完整覆盖双 agent 建立、经验写入、订阅/发布、显式采纳、warm→cold、
模糊召回、强 cue 再激活、确定性灵感、idea 隔离和 idea 显式采纳：

```bash
.venv/bin/experience-hub demo --reset
```

成功时 `data.all_invariants_hold` 为 `true`，并输出恰好 11 个 stage。`--reset`
会删除并重建 `.data/demo.db` 及它的 WAL/SHM/journal sidecar，不会触碰默认服务
数据库。省略 `--reset` 且 demo 库已存在时会返回 `demo_database_exists`，这是
为了避免意外覆盖保留的演示状态。

demo 是确定性验收，不是生产数据导入器。其固定时钟、固定 UUID 序列和隔离
数据库只服务于可重放演示。

### 9. Benchmark 指标与失败排障

执行离线效果基准：

```bash
.venv/bin/experience-hub benchmark
```

输入来自受版本控制的 `benchmarks/seed.json` 和 `benchmarks/cases.jsonl`。runner
先建立并 checkpoint 一个不可变 pre-run SQLite snapshot，再从完全相同的主库
字节分别创建 `replay-a` 和 `replay-b`。每个 case/baseline 使用独立 clone。
`.data/benchmark/` 是带 ownership marker 的受控工作区；runner 会拒绝删除含
未知文件的工作区。

命令总是把规范 report 写成一行 JSON。所有 gate 通过时退出 0；任一 gate 失败
时仍输出完整 report，但退出 1，失败名称列在 `data.failed_gates`。

| Gate | 要求 | 解释 |
| --- | ---: | --- |
| `focused_macro_recall_at_5` | `>= 0.90` | focused 普通检索的 case 宏平均 recall@5 |
| `cold_macro_recall_at_5` | `>= 0.85` | cold cue 检索与因果再激活的宏平均 recall@5 |
| `cold_recall_gain_over_hot_warm_baseline` | `>= 0.25` | 完整 cold 召回相对仅 hot/warm admission baseline 的增益 |
| `distractor_false_reactivations` | `= 0` | 无关 cue 不得扩展、再激活 cold 经验，也不得产生对应因果事件 |
| `pending_capsule_leakage` | `= 0` | quarantine 中 pending capsule 不得进入经验检索 |
| `adopted_provenance_completeness` | `= 1.0` | 显式采纳后 provenance chain 与 root fingerprint 完整 |
| `valid_idea_count` | `>= 12` | 持久化 idea 中 schema 与证据引用有效的数量 |
| `idea_schema_and_evidence_validity` | `= 1.0` | 全部 idea 通过 schema、frozen evidence 与 source 校验；任何 fixture evidence coverage failure 都使其失败 |
| `unique_mechanism_ratio` | `>= 0.70` | 有效 idea 的 distinct mechanism cluster 占比 |
| `same_snapshot_incubation_promotion` | `= 0` | 相同 snapshot 的重复 run 不能伪造独立复现并提升 maturity |
| `byte_identical_replay` | `= true` | 两套 fresh clones 的完整规范输出逐字节相同 |

排障顺序：

1. 保存 stdout，再确认 `data.failed_gates` 和对应 gate 的 `actual`/`required`；
2. 检查 `git diff -- benchmarks/seed.json benchmarks/cases.jsonl`，不要在不知情
   的情况下用改 fixture 或降阈值掩盖退化；
3. 检查 `.data/benchmark/` 是否含未知文件、非法 marker、残留进程或非空 WAL；
4. 根据 gate 定位机制，而不是只看总分：
   - focused/cold recall 下降：检查 tokenizer、ranking、温度迁移和 cold expansion；
   - cold gain 下降：检查 baseline 是否只禁用了 cold candidate admission；
   - distractor 非零：检查 expanded/reactivated hit 与
     `experience.reactivated` causation；
   - pending leakage/provenance 失败：检查 quarantine、显式 adoption 和
     provenance root；
   - idea validity/coverage 失败：检查 frozen snapshot item、stable evidence
     key、source version 和 operator 输出 schema；
   - unique mechanism 下降：检查 mechanism hash/cluster 与去重是否过度合并；
   - same-snapshot promotion 非零：incubation 必须按不同 snapshot 计数，不能按
     不同 run ID 计数；
   - byte replay 失败：检查未规范化的 UUID、时间、路径、集合顺序或 clone 间
     side effect 泄漏。

若 CLI 只返回安全的 `internal_error`，可在受信任的本机直接运行 runner 以取得
Python traceback；不要把 traceback 或含 provider 元数据的日志公开：

```bash
.venv/bin/python - <<'PY'
import asyncio

from experience_hub.benchmark.runner import run_benchmark

result = asyncio.run(run_benchmark())
print(result.body.decode("utf-8"))
PY
```

### 10. 故障恢复

先区分三类状态：

- **权威源**：event ledger、command receipt、experience version/payload、
  sharing provenance 和 inspiration source；损坏时必须从可信 backup 恢复；
- **projection**：可从已验证权威源重放；只有这一类适合 `--repair`；
- **物理 payload codec 漂移**：语义内容和 hash 完好时可用
  `payloads reconcile` 修正。

标准恢复流程：

1. 停止服务和所有 writer；
2. 把当前主库及 sidecar 移到独立取证目录，禁止把旧 sidecar 与恢复主库混用；
3. 复制已通过 integrity check 和 SHA-256 校验的 backup；
4. 检查 SQLite 完整性；
5. 启动当前 CLI 让受支持的 schema migration 完成；
6. verify projection；仅在权威源通过校验且确有 mismatch 时 repair；
7. reconcile payload；
8. 启动服务并检查 `/health`。

示例：

```bash
DB=.data/experience_hub.db
BACKUP=/secure/path/experience_hub-verified.db
RECOVERY=".data/recovery-$(date -u +%Y%m%dT%H%M%SZ)"

mkdir -p "${RECOVERY}"
for FILE in "${DB}" "${DB}-wal" "${DB}-shm" "${DB}-journal"; do
  if test -e "${FILE}"; then
    mv "${FILE}" "${RECOVERY}/"
  fi
done
cp -p "${BACKUP}" "${DB}"
sqlite3 "${DB}" "PRAGMA integrity_check;"

.venv/bin/experience-hub projections rebuild --verify --database "${DB}"
```

如果且仅如果上一步返回 `projection_mismatch`，再执行：

```bash
.venv/bin/experience-hub projections rebuild --repair --database "${DB}"
.venv/bin/experience-hub projections rebuild --verify --database "${DB}"
.venv/bin/experience-hub payloads reconcile --database "${DB}"
```

不要在 `source_integrity_error` 后直接执行 SQL“补行”、改 hash、删 event 或改
Alembic/projection version。这样会破坏因果闭包和后续可重放性。保留失败库、
sidecar、应用版本、命令输出与 hash，再从最后一个验证通过的 backup 恢复。

API 启动会把可识别的中断 inspiration run 恢复为一致的持久化终态，然后才进入
ready；它不会把任意损坏数据库“自动修好”。完整 projection equality 仍需运维
命令验证。

### 11. 日志、审计与日常检查

Uvicorn access log 写到 stdout；startup/shutdown 与意外异常诊断写到 stderr。
程序化 migration 不会重配宿主 logging，也不保证 Alembic INFO；若宿主显式配置
Alembic logger，其记录进入相应 handler。一次性 CLI 的规范 JSON 也写到 stdout，
因此服务日志与一次性命令输出应按入口分别采集，且 stdout/stderr 不要合并：

```bash
mkdir -p .logs
.venv/bin/experience-hub serve \
  --host 127.0.0.1 \
  --port 8765 \
  > .logs/serve.stdout.log \
  2> .logs/serve.stderr.log
```

应用对每个 HTTP error 返回 `X-Request-ID`；未处理异常的服务端日志也包含相同
request ID，可用于关联。领域错误和 provider 错误对外采用稳定、安全的 code，
不会把原始 provider 响应当作客户端诊断。日志仍可能包含路径、agent identity
或操作元数据，应按敏感运维数据保护，并由外部进程管理器负责轮转与留存。

自动化解析 CLI 输出时不要使用 `2>&1`。一次性维护命令可单独追加 stderr：

```bash
.venv/bin/experience-hub projections rebuild --verify \
  --database .data/experience_hub.db \
  > .data/projection-verify.json \
  2>> .logs/maintenance.log
```

建议的日常检查：

- 持续探测 `/health`，记录 status、schema revision 和 reducer versions；
- 每次发布后执行 projection verify；
- 定期执行经过校验的 SQLite backup，并做一次独立 restore 演练；
- 生命周期批次后检查 transition/archive 数量；升级或怀疑 codec 漂移时再
  reconcile payload；
- 代码、fixture 或依赖变化后执行 benchmark；
- 对 `database_busy`、`source_integrity_error`、projection mismatch、
  provider timeout 和连续 benchmark gate failure 设置告警。
