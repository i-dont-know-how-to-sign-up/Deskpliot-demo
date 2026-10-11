# 工程可靠性专项数据集

> P2 增补（2026-10-11）：现有 12 类 x 5 条 P0/P1 场景之外，新增 4 类 x 2 条 P2 场景，总计 68 条。新增场景均为合成故障注入；没有真实 SMTP 投递。

| P2 场景 | 条数 | 关键断言 |
|---|---:|---|
| `file_compensation` | 2 | 审批前无回滚，确认后恢复原文件并记为 compensated |
| `file_compensation_conflict` | 2 | 外部改动后拒绝回滚，不覆盖用户文件 |
| `mail_dispatch_reconcile` | 2 | 派发中断后禁止 SMTP 重发，只读查找已发送箱，找不到保留未知 |
| `attachment_renewal_guard` | 2 | 审批预览含附件摘要，附件变更使旧审批失效 |

运行全部场景：`python -m eval.run_reliability_eval --count 68`；单类可用 `--scenario mail_dispatch_reconcile --count 2`。下面原有的 60 条描述保留为 P0/P1 基线定义，不能把 68 条解释为 68 种独立故障。

## 定位与构成

`reliability_cases.jsonl` 是工程可靠性 P0/P1 的离线故障注入集，独立于原有 163 条 DeskPilotBench 和 22 条多模态集，保持功能基线可对照。它不评测 LLM 答案 Accuracy/F1，也不是公开 GAIA/BFCL 的官方榜单数据。

当前是 **12 类场景 × 5 次固定 seed 重复实验，共 60 条记录**。seed 0–4 用于复现；仅瞬时重试场景依据 seed 改变失败次数，其他场景重复检查同一个契约，不应描述为 60 种独立故障。测试临时生成文档、图片、PDF 和模拟邮箱数据，不联网、不下载大型权重、不读取私人邮箱。

| 场景 | 条数 | 注入方式与验收 |
|---|---:|---|
| nonzero_exit | 5 | 工具返回 executed=true、returncode=3；必须 failed |
| false_result | 5 | 工具正常返回 ok=false；不能误判成功 |
| transient_retry | 5 | 前 1–2 次 ConnectionError；有限重试后成功 |
| auth_no_retry | 5 | PermissionError 模拟认证/权限错误；只执行一次 |
| bounded_retry | 5 | 持续网络错误；最多 3 次节点尝试 |
| deadline | 5 | 派发前期限耗尽；handler 调用次数为 0 |
| cancel_process | 5 | 子进程启动后请求取消；实际停止而非只修改状态 |
| approval_restart | 5 | 重建审批 Store；会话绑定有效且不能消费两次 |
| attachment_changed | 5 | 预览后修改附件内容；拒绝旧审批且无提交 |
| checkpoint_process_restart | 5 | 真正子进程 os._exit(23)；重建后复用证据，上游只执行一次 |
| unknown_no_replay | 5 | 写节点结果 unknown；恢复拒绝自动重放 |
| pdf_pause_resume | 5 | 生成三页 PDF，首批后暂停；恢复跳过首批 OCR |

子进程树、CAS/lease 竞争、回调失败、预算计账、审批原任务回写和 UI 加载另由 `tests/` 的回归测试覆盖，不将没有明确观测分母的性质伪造为报告指标。

## 用例格式

```json
{
  "schema_version": "1.0",
  "id": "reliability_10_0",
  "module": "reliability",
  "capability": "checkpoint_process_restart",
  "scenario": "checkpoint_process_restart",
  "seed": 0,
  "source": {"type": "synthetic_fault_injection", "version": "p0p1-20261011"},
  "input": {"fixture": "generated_in_temporary_workspace", "fault": "checkpoint_process_restart"},
  "expected": {"invariant": "中断后复用检查点", "no_real_email": true},
  "metrics": ["task_state_consistency", "response_time", "bounded_attempts"]
}
```

`id` 必须唯一；`scenario` 映射到确定性执行器；`expected` 是契约描述，真正判定由场景中的状态、产物、调用次数断言实现，不由 LLM 主观评分。`metrics` 是用例的声明分类，runner 只汇总实际提供的观测字段，并非每个声明都自动输出独立百分比。

## 指标与理由

- 契约通过率：所有场景断言均满足的记录数/执行记录数，是当前故障样本的结果，不代表生产可用性。
- 平均/P95 实验耗时：测量完整故障脚本，便于回归监控；不是用户真实任务或模型响应延迟。P95 使用 nearest-rank。
- 检查点复用率：复用节点/符合复用条件节点，当前仅统计两类恢复实验；判断是否重复执行已完成工作。
- unknown 自动重放数：对应不确定副作用实验中额外派发次数；目标为 0，不声称普通 SMTP exactly-once。
- 已观测重复提交数：仅统计审批消费/检查点场景的明确断言；不替代真实 SMTP 投递计数。
- 取消响应 P95：从调用 cancel 到受控进程结束，不包含进程启动时间；验证状态与真实执行是否一致。
- 最大节点 attempt：瞬时/持续网络故障中的次数上限；验证重试没有无限放大。

报告保留 case_id、seed、平台、Python/项目版本、Git commit/dirty、数据 SHA-256、类别和具体错误。只聚合提供字段的样本；故障导致未获得测量的记录留在失败详情中，不默认为零。没有运行真实 API，因此 Token/API 效果不能从本报告推导。

## 执行与扩展

```powershell
.\.conda\deskpilot-py311\python.exe -m eval.run_reliability_eval --count 60
.\.conda\deskpilot-py311\python.exe -m eval.run_reliability_eval --scenario checkpoint_process_restart --count 5
```

产物为 `eval/reports/reliability/latest.json` 和 `latest.md`，均不提交 Git。`--count` 选择过滤后数据集的前 N 条。

新增场景时，在 `eval/reliability_scenarios.py` 的 `SCENARIOS` 中注册并实现断言，再追加唯一 ID 的 JSONL。参数化 pytest 自动纳入新类别；如引入不同故障位置，需明确读取用例输入并真正改变注入点，不只修改标签。P2 再追加文件原子提交、补偿冲突和 SMTP 副本核对等场景。

手工 fixture 可执行 `python -m eval.prepare_reliability_demo` 生成。已有同名文件不会覆盖，便于安全重复操作。
