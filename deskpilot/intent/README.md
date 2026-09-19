# Intent System

DeskPilot 的意图系统负责把用户自然语言请求拆成两层决策：

1. 先判断是 `direct_answer`、`tool_call` 还是 `clarify`
2. 如果需要调用工具，再让 LLM 从注册表里选择具体工具并填充槽位

## 设计目标

- 减少纯关键词匹配带来的误判
- 让工具选择显式依赖 `ToolRegistry`
- 在真正执行前完成槽位校验和参数归一化
- 让澄清问题尽量在调用工具前发生

## 关键组件

- `router.py`
  - 输入用户问题、可用工具列表、会话上下文
  - 输出 `IntentDecision`
- `validator.py`
  - 按工具 schema 校验 required slots
  - 做基础类型归一化
- `schemas.py`
  - 定义路由结果与槽位校验结果的数据结构

## 槽位校验是什么

这里的“槽位校验”不是泛泛的参数校验，而是专门针对工具调用的结构化检查：

- 是否缺少必填参数
- 参数类型是否可转换
- 是否存在默认值可以补齐
- 是否需要追问用户补充信息

它发生在“LLM 选工具之后、工具执行之前”。

## 路由流程

```text
User Input
  -> IntentRouter
  -> SlotValidator
  -> Permission Precheck
  -> Tool Execution
```

## 扩展方式

新增工具时，只需要把工具注册到 `ToolRegistry`，并在路由提示中让 LLM 看到该工具的 schema。后续如果要增加多工具编排、子代理协同或计划执行，可以继续在这个目录下扩展新的路由器或策略模块。
