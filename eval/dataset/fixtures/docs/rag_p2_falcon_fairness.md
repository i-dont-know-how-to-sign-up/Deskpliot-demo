# Falcon Scheduler 公平性

Falcon Scheduler 为每个租户维护 deficit counter，并按权重补充时间片。小任务通过 aging 逐步提高优先级，避免持续被大批量任务饿死。

公平性的代价是高优先级任务无法无限抢占资源；当租户很多时，counter 扫描也会增加调度开销。
