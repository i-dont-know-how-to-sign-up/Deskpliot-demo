# Windows 桌面 Beta 发布说明

本模块实现 P0/P1 桌面发布基础。P2 的受限、仅本机访问的私有 API 见[私有 API 与 Docker 沙箱说明](../api/README.md)。桌面 EXE 不是 OS 沙箱，也不是公网 API 服务；发布模式的任意 Python/Shell 执行仍被禁用。桌面入口为 `run_demo.py`；源码开发模式只有设置 `DESKPILOT_RELEASE_MODE=1` 才启用发布策略。

## 构建

使用干净的 Windows Python 3.11 环境。`requirements-desktop.lock` 锁定构建环境观察到的直接与传递依赖，但不包含模型权重或 wheel 哈希。基础包不包含 PaddleOCR、Torch 和 Transformers；只有生成安装程序时才需要 Inno Setup 6。

```powershell
python -m pip install -r requirements-desktop.lock
.\packaging\build_windows.ps1 -Python (Get-Command python).Source
# 需要安装程序时再添加 -Installer
```

通常输出 `dist/DeskPilot/DeskPilot.exe`；安装包输出位于 `dist-installer/`。便携版是 onedir 构建，**必须分发完整的 `DeskPilot/` 目录及 `_internal/`**，不能只复制 EXE。分发前应签名。`-Full` 是实验性构建选项，需要先安装 `requirements-multimodal.txt`；本地 OCR/视觉运行时、权重下载和干净机器首轮推理尚未完成发布验收。基础包可使用云端 VLM 或受限本地图像回退。

2026-10-29 的 Docker 参数修复之后，单独重建的便携候选位于 `dist/deskpilot_sandboxfix_20261029/DeskPilot/DeskPilot.exe`，EXE SHA-256 为 `ae9eabd246302090634c119fa950132f2b4a0e9e17261df47e9d6e3a2589d39a`。原 `0.9.0-beta.3` EXE 打包于修复之前，不能用于证明 Docker 沙箱可运行；新便携候选也不等于安装程序或生产发布已验收。不要用旧包 SHA-256 代表新构建。

默认用户数据目录为 `%LOCALAPPDATA%\DeskPilot`；首次设置可选择其他绝对路径和授权工作区。发布模式的凭证通过 Windows CurrentUser DPAPI 存在 `secrets.dpapi`，与 `settings.json` 分开，不能直接迁移给另一 Windows 用户。发布模式不读取源码树中的 `.env`。基础安装程序的目标机器无需 Python/Conda。

## 维护与备份

备份或迁移前先关闭桌面主程序和后台 worker。通过开始菜单的 **DeskPilot 备份与迁移** 进入维护界面。备份使用 AES-256-GCM 和至少 12 字符的显式口令；SQLite 文件先生成一致性快照，恢复时核对哈希并只写入新目录，不覆盖已有数据。备份只包含 `data/`：profile 设置和 DPAPI 密钥必须在恢复后重新配置。`data/` 中遗留的 `secrets.dpapi` 会被跳过。操作时会在目标磁盘短暂出现明文临时 ZIP，必须使用可信本地磁盘并妥善保管口令。

开发者可使用源码命令：

```powershell
$env:DESKPILOT_RELEASE_MODE='1'
python -m deskpilot.release doctor
python -m deskpilot.release backup D:\backups\deskpilot.dpb
python -m deskpilot.release restore D:\backups\deskpilot.dpb D:\DeskPilot-Restored\data
python -m deskpilot.release migrate D:\broagent\data D:\DeskPilot-Migrated\data
python -m deskpilot.release candidate eval\reports\desktop_candidate.json
```

若用 `--distpath` 构建版本化目录，记录候选时须给 `candidate --executable` 传入**实际 EXE**；否则默认的 `dist/DeskPilot/DeskPilot.exe` 可能是旧包。`doctor` 和 UI 诊断导出仅包含汇总健康信息；原始 prompt、路径、邮件和密钥不会自动上传。反馈保存在本地 `feedback/`；登录自启 worker 需在设置中主动开启，关闭自启只移除本安装的快捷方式。

## EXE 沙箱验证

**双击桌面 EXE 并在聊天里要求执行代码，不是 Docker 沙箱测试。**桌面发布策略会拦截 Python/Shell。仅私有 API 的 `/v1/sandbox/tasks` 可选接入 Docker；其 API 和 worker 可以通过同一个便携 EXE 的 `-m deskpilot.api`、`-m deskpilot.tasks.worker --api` 参数分别启动。必须先启动 Docker Desktop 的 Linux engine，使用固定本地镜像 ID、独立测试 profile 和具有 `execute` 作用域的 Token。完整命令行、Python 调用、验证标准和安全边界见[私有 API 与 Docker 沙箱说明](../api/README.md)。

可复用的打包后冒烟命令：

```powershell
.\.conda\deskpilot-py311\python.exe packaging/smoke_sandbox_api.py `
  --exe D:\broagent\dist\deskpilot_sandboxfix_20261029\DeskPilot\DeskPilot.exe
```

冒烟脚本实际启动新 EXE 的 API 和 worker，并验证 `status=succeeded`、`sandbox.sandbox=docker`、容器内 `Linux`/UID 65534、输入可读且合成宿主文件不可见。它不证明桌面聊天进入沙箱，也不替代 7 项真实容器安全测试或独立目标主机验收。

本机验证记录：打包后端到端冒烟通过；`tests/test_p2_docker_sandbox.py`、`tests/test_p2_sandbox_api.py`、`tests/test_p2_api_contract.py` 和 `tests/test_p2_api_adversarial.py` 共 **71 passed**（含 7 项真实容器测试），且无残留 `deskpilot-sbx` 容器。此结果限定于当前 Windows + Docker Desktop 主机。

## 发布验收

1. 在没有 Python/Conda 的干净 Windows 10/11 用户环境安装，完成首次设置，确认不依赖 `D:\broagent` 等开发路径。
2. 运行 `DeskPilot.exe doctor`，检查发布模式、DPAPI 密钥状态，以及 Python/Shell 禁用。仅做打包 UI 冒烟时，可在全新 `DESKPILOT_HOME` 下设置 `QT_QPA_PLATFORM=offscreen` 并执行 `DeskPilot.exe --smoke-ui`；这不能代替真实显示器与已有 profile 验证。实际测试设置页连接和一条直接问答。
3. 用非私有 PDF/图片建立索引并进行带引用问答；生成报告、保存邮件草稿，确认发送须在聊天内人工审批。真实服务商用专用测试邮箱。
4. 关闭主程序与 worker 后创建加密备份，恢复到全新 profile，验证完整性并重新录入凭证；覆盖错误口令、已存在目标、损坏归档和旧数据迁移。检查恢复数据前不删除原始数据。
5. 在非私有数据副本上验证安装升级与旧版回滚；使用同一模型/配置运行固定离线、API、VLM 测试，对比模块质量、P95 与 Token。安全退化、沙箱缺口或邮件副作用结果不明时不得宣布生产发布。

建议源码检查：

```powershell
python -m pytest tests/test_desktop_release.py tests/test_task_qml.py -q
python -m eval.run_eval --count 163 --mode offline
python -m eval.run_baseline --project-version 0.9.0-beta.3 `
  --change-summary "Docker sandbox API candidate; live security acceptance pending"
```

基线命令在配置了 LLM/VLM 凭证时可能调用付费 API，不要无意运行。候选元数据**不是**验收报告；最终还需干净机器安装、真实 API/邮箱冒烟和人工安全审查。

## 安全边界

- 发布模式即使收到用户确认，也会阻断任意 Python/Shell。AST 静态检查、命令白名单和 Job Object 不能替代 OS 沙箱。Docker 当前只保护可选 API 的 Python 提交，不保护所有桌面/Agent 工具。
- 用户 profile 中的对话、任务正文和文档派生数据未整体加密；DPAPI 只保护凭证。应使用操作系统账户权限和磁盘加密保护数据，不要上传包含隐私的 `data/` 或支持包。
- 邮件 MCP stdio 服务不是带鉴权的 DeskPilot HTTP API。私有 API 只开放直接问答、受限审批和单独授权的可选沙箱任务；多租户、集中监控和 Bad Case 接入仍属后续阶段。
