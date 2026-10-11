# Durable Tasks

此模块为桌面 Agent 提供单机持久任务状态，不是分布式工作流引擎。

| 文件 | 职责 |
|---|---|
| store.py / models.py | SQLite WAL、短事务、状态/事件、检查点、attempt、CAS 与 lease fencing |
| service.py | 持久队列、注册领域 handler、任务控制、单 worker lease、恢复与真实调用统计 |
| control.py / retry.py | deadline、暂停/取消、调用及 Token 预算、分类有限重试和 unknown 副作用 |
| effects.py | 工具派发/返回日志，阻止不确定副作用自动重放，不提供通用回滚 |
| compensation.py / mail_reconcile.py | 经审批和版本校验的文件补偿，以及基于 Message-ID 的已发送副本核对 |
| worker.py | 可选 headless worker，与 UI 共用同一数据库和 worker 锁 |

任务 DB 默认位于文档索引旁的 `task_ledger.sqlite3`，旧 `task_ledger.json` 仅幂等迁移为不可恢复历史。新记录保存序列化计划与领域输入/输出，重建 handler 而不是序列化 Python closure。成功节点在校验输入、文件指纹及 handler 版本后复用；失败和等待人工节点不算完成依赖。

默认单 worker，队列上限 50、心跳 5 秒、lease 15 秒、最多 3 次手动恢复。聊天任务执行预算为 600 秒、40 次工具调用、80,000 Token、40 次 API attempt；已完成尝试的成本在恢复时保留。独立提交的多模态索引任务可使用 3,600 秒预算。缺少服务端 usage 显式标为 unknown。

UI 的 Tasks 面板提供暂停、取消、继续及重新预览。退出 UI 请求保存并暂停；暂停/取消是协作式请求，不保证撤回已经派发的网络操作。PDF 在批次提交后保存页检查点；本地代码/OCR/视觉编码子进程用 Windows Job Object 或 Linux process group 回收进程树。

```powershell
python -m deskpilot.tasks.worker --list
python -m deskpilot.tasks.worker --once
python -m deskpilot.tasks.worker
```

CLI 可用 `--index path/to/index.json` 隔离索引。后台持续执行必须由用户显式启动；没有实现操作系统级进程守护或多机高可用。已中断任务需在 UI 显式“继续”，`--once` 只领取排队任务，不自动重发历史失败任务。

审批保存在同一 DB，绑定会话、原 task/node、计划版本、参数和附件 SHA；确认消费与创建 attempt 在同一事务完成。正文或附件变化需要新预览，模型不能授予权限。审批后的原任务终态回写；若通用 DAG 尚有下游，当前需要显式继续调度。

文件写入或移动可在操作日志与目标版本匹配时申请人工确认的补偿；文件已被外部修改、备份缺失或状态竞争时拒绝自动处理。邮件核对只检查 Message-ID 与服务器已发送副本，找不到副本不代表 SMTP 未投递，不能自动重发。持久化对象可能含文档、报告或邮件正文，禁止上传任务 DB、对象目录及原始事件；凭证剔除不等于正文加密。尚未实现 DPAPI/额外 ACL 自动配置、通用 Saga 回滚、SMTP 完整核对或 exactly-once；取消不会撤销已产生副作用。

```powershell
python -m eval.run_reliability_eval --count 60
python -m pytest tests/test_task_store.py tests/test_task_execution_control.py tests/test_durable_supervisor.py tests/test_task_lifecycle.py -q
```
