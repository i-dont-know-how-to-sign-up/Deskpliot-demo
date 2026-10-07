# DeskPilot 测评资料

DeskPilot 是一个本地优先的个人办公助手 Agent。当前评测版本重点覆盖文档 RAG、工具路由、网页调研、文件写入、权限控制和会话记忆。

## RAG 设计

文档会被解析为 Document，随后按段落切分为 Chunk。每个 Chunk 保存文档 ID、来源标签、位置和 embedding。查询时先对问题生成 embedding，再按 cosine similarity 返回相关证据。

## 安全设计

工作区内的新文件可以低风险写入。工作区外的写入需要人工确认。删除、格式化和修改系统状态的命令会被阻断。工具执行失败时，Agent 不应声称任务已经成功。

## 记忆设计

会话消息保存在 SessionStore 中。系统可以抽取 fact、preference、decision、task 和 artifact 类型的记忆，并在后续问题中检索相关记忆。
