# 私有 API 与 Docker 沙箱

本接口目前是**仅监听本机回环地址的单租户候选版**，不是公网服务。HTTP 进程负责认证、入队和查询；独立 worker 执行任务。默认只开放通用问题的直接问答。文件、网页、Shell、通用 Agent 工作流没有通过 HTTP 开放；邮件接口仅能审批服务端已经持久化、属于当前主体的待审批动作，不能用 HTTP 创建任意邮件任务。

Docker 沙箱只处理显式提交到 `POST /v1/sandbox/tasks` 的 Python 代码。**桌面聊天窗口不会自动转到这个沙箱**，发布模式下桌面的 Python/Shell 工具仍被禁用。独立 worker 是进程边界，不是宿主机安全边界；Docker Desktop/daemon 本身仍有宿主权限。沙箱通过本机指定测试，也不意味着可以向公网或不可信用户开放。

## 准备环境

- 使用 Python 3.11 安装 `requirements.txt` 和 `requirements-api.txt`。打包 EXE 的目标机器无需单独安装 Python，但生成测试 Token 的下例使用项目 Python。
- 启动 Docker Desktop，切换到 Linux containers。先运行 `docker info --format '{{.OSType}}'`，确认输出 `linux`。
- 在管理员控制下准备并审核 Python 镜像，记录本机不可变镜像 ID。运行时只接受 SHA-256 ID，不接受标签，也不会自动拉取镜像。不要把 Docker socket 挂进容器，或通过 TCP 暴露 daemon。
- API 和 worker 使用**相同**的 `DESKPILOT_HOME` 与索引路径；测试目录必须与日常桌面 profile 分开。模型凭证通过 DeskPilot 发布配置/系统凭证存储提供，不要在仓库写入密钥。单纯测试沙箱不需要模型 API Key。

```powershell
docker pull python:3.11-alpine
docker info --format '{{.OSType}}'
docker image inspect --format '{{.Id}}' python:3.11-alpine
```

下例用 `D:\DeskPilotPrivateApi` 作为**独立测试目录**。目录内的 `tokens.json` 保存 Token 的 SHA-256 摘要，命令只显示一次明文 Token；不要提交 Token、摘要文件或测试 profile。`--allow-sandbox-execute` 显式授予 `execute` 作用域，普通 Token 不具备此权限。文件已存在时该命令会拒绝覆盖；重复测试可继续使用已有 Token，或新建另一个独立目录。API 每次请求都会重新读取目录，因此删除记录即可撤销权限。

```powershell
$home = 'D:\DeskPilotPrivateApi'
$env:DESKPILOT_HOME = $home
.\.conda\deskpilot-py311\python.exe -m deskpilot.api.credentials `
  --output "$home\tokens.json" --principal sandbox_tester --allow-sandbox-execute
```

## 启动打包后的 EXE

仅分发 `DeskPilot.exe` 不够：便携版须保留整个 `DeskPilot` 目录，包括 `_internal`。以下使用**重新打包且包含最新 Docker 参数修复的 EXE**；旧 `0.9.0-beta.3` 候选包在修复前构建，不能作为沙箱验收对象。

在**两个 PowerShell 窗口**都先执行以下配置。把 `$exe` 改为当前实际路径；`$env:DESKPILOT_DOCKER_IMAGE` 必须是 `sha256:...`，不能填 `python:3.11-alpine`。不要将真实桌面数据目录用于测试。

```powershell
$exe = 'D:\broagent\dist\deskpilot_sandboxfix_20261029\DeskPilot\DeskPilot.exe'
$home = 'D:\DeskPilotPrivateApi'
$env:DESKPILOT_HOME = $home
$env:DESKPILOT_ENABLE_DOCKER_SANDBOX = '1'
$env:DESKPILOT_DOCKER_IMAGE = docker image inspect --format '{{.Id}}' python:3.11-alpine
```

窗口 1 启动 worker：

```powershell
& $exe -m deskpilot.tasks.worker --api --index "$home\index\index.json"
```

窗口 2 启动 API：

```powershell
& $exe -m deskpilot.api --index "$home\index\index.json" `
  --tokens "$home\tokens.json" --host 127.0.0.1 --port 8765
```

这两个进程必须持续运行。访问 `http://127.0.0.1:8765/health/live` 只说明 HTTP 进程存活；`/health/ready` 返回 `ready` 才说明同一数据库中有可用 worker。EXE 是窗口程序，启动后不一定在终端输出日志，可用健康检查和任务状态判断。沙箱需要 worker 能访问本机 Docker 命名管道；若只有 `dockerDesktopLinuxEngine` 管道，可在 **worker 所在窗口**设置 `$env:DESKPILOT_DOCKER_HOST='npipe:////./pipe/dockerDesktopLinuxEngine'` 后重启 worker。只允许本机这两个 Windows 管道或 Linux 的本机 Unix socket，拒绝远程 TCP Docker endpoint。

源码方式也可分别运行 `python -m deskpilot.tasks.worker --api ...` 和 `python -m deskpilot.api ...`，但这**不能替代对打包 EXE 的验证**。两个进程都要设置启用开关；只有 worker 端还必须有固定镜像 ID。

## 命令行调用

先在**第三个 PowerShell 窗口**设置 `$token` 为生成目录时显示的明文 Token，不要把它写入脚本、提交仓库或发到聊天。以下请求会返回 `202` 和 `task_id`；任务异步运行，随后查询结果。`Idempotency-Key` 每个新任务都用新的随机值；相同键加相同请求可重放到原任务，内容变更会返回 `409`。

```powershell
$token = [System.Net.NetworkCredential]::new('', (Read-Host '输入测试 Token' -AsSecureString)).Password
$base = 'http://127.0.0.1:8765'
$headers = @{ Authorization = "Bearer $token"; 'Idempotency-Key' = [guid]::NewGuid().ToString() }
$body = @{ code = "import os, platform; print(platform.system(), os.getuid())"; inputs = @{} } |
  ConvertTo-Json -Compress
$task = Invoke-RestMethod "$base/v1/sandbox/tasks" -Method Post -Headers $headers `
  -ContentType 'application/json' -Body $body
$task.task_id
$result = Invoke-RestMethod "$base/v1/tasks/$($task.task_id)" `
  -Headers @{ Authorization = "Bearer $token" }
$result | ConvertTo-Json -Depth 6
```

若仍是 `queued` 或 `running`，稍后再次执行查询 `$result = ...` 与显示 `$result | ConvertTo-Json -Depth 6` 两行。成功时应看到 `status=succeeded`、`answer` 中的 `Linux 65534`，以及 `sandbox.sandbox=docker`、`sandbox.ok=true`、`sandbox.returncode=0`。元数据由 worker/Runner 产生，不能只凭模型文字声称“已在沙箱运行”。短任务可能来不及通过 `docker ps --filter name=deskpilot-sbx` 观察到容器；长一点的无害任务可配合该命令查看。验证完执行 `Remove-Variable token` 清除当前终端变量，并关闭测试 API/worker。

若要从代码调用，标准库即可，无需使用 DeskPilot 内部模块：

```python
import json
import os
import secrets
import time
from urllib.request import Request, urlopen

base = "http://127.0.0.1:8765"
token = os.environ["DESKPILOT_API_TOKEN"]  # 只放在客户端环境中，不写入源码
payload = json.dumps({"code": "print(1 + 1)", "inputs": {}}).encode()
headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json",
           "Idempotency-Key": secrets.token_hex(16)}
with urlopen(Request(base + "/v1/sandbox/tasks", data=payload, headers=headers), timeout=5) as response:
    task_id = json.load(response)["task_id"]
for _ in range(60):
    with urlopen(Request(base + "/v1/tasks/" + task_id,
                         headers={"Authorization": f"Bearer {token}"}), timeout=5) as response:
        result = json.load(response)
    if result["status"] in {"succeeded", "failed", "cancelled", "partial"}:
        print(result)
        break
    time.sleep(0.5)
else:
    raise TimeoutError(task_id)
```

不要把 `execute` Token 交给不可信客户端。当前沙箱接口**没有逐任务人工审批**：授予 `execute` 本身就是管理员授权。

## 其他接口与限制

| 接口 | 作用域 | 当前行为 |
|---|---|---|
| `POST /v1/tasks` | `submit` | 提交通用直接问答；JSON 含 `question`、可选 `session_id`，返回 `202` |
| `GET /v1/tasks/{id}` | `read` | 查询本人任务、脱敏错误、审批预览和结果；他人任务返回 `404` |
| `GET /v1/tasks/{id}/events` | `read` | 有界 SSE 事件，可用 `Last-Event-ID` 续读；不返回原始 prompt/工具载荷 |
| `POST /v1/tasks/{id}/cancel`、`/resume` | `control` | 持久化控制请求 |
| `POST /v1/tasks/{id}/approve`、`/reject` | `approve` | 只接受服务端提供的 `action_id` 和 `preview_digest`，不接受客户端自带工具参数或 `confirm=true` |
| `GET /v1/artifacts/{id}` | `read` | 目前只提供已完成的直接问答 `answer.txt`；未开放本地文件/报告交付 |
| `POST /v1/sandbox/tasks` | `execute` | 可选的受限 Python 容器执行；需要 Docker 和双进程启用 |

请求正文最多 16 KiB，直接问题最多 4000 字符，沙箱代码最多 8192 UTF-8 字节。入口按主体限制活动任务、分钟频率和每日预留 Token；超限返回 `429` 与 `Retry-After`，这不是计费级精确计量。任务由 SQLite 持久化后才派发；30 秒后可回收卡在 `creating` 的预留。Docker 失败时任务失败，**绝不退回宿主 Python 执行**。

容器默认以 UID/GID 65534 运行，根文件系统只读、网络关闭、无 capabilities、禁止提权，限制 256 MiB 内存/交换、1 CPU、32 PID、32 MiB `/tmp`、10 秒墙钟时间和 64 KiB 合并输出。输入仅是经 `SandboxBroker` 校验并复制到一次性目录的内联文本，以只读方式挂在 `/inputs`；请求不能指定宿主路径、卷、镜像或 Docker 参数。这是纵深防御，不能宣称具有 VM 级隔离或防容器逃逸保证。

## 验证与发布边界

打包后可运行本仓库的端到端冒烟脚本。脚本使用临时测试 profile、临时 Token 和合成宿主文件，启动**传入的 EXE** 的 API/worker，不调用模型或真实邮箱，结束时关闭进程；需要 Docker Linux engine 与已拉取的 `python:3.11-alpine`：

```powershell
.\.conda\deskpilot-py311\python.exe packaging/smoke_sandbox_api.py `
  --exe D:\broagent\dist\deskpilot_sandboxfix_20261029\DeskPilot\DeskPilot.exe
```

另需运行离线 API/权限测试，以及显式开启的 7 项真实容器安全测试；真实测试使用合成宿主秘密，不自动拉取镜像。未启用真实测试时 pytest 会**跳过**，不能计作通过；Docker 不可用时显式测试会失败。

```powershell
.\.conda\deskpilot-py311\python.exe -m pytest -q -p no:cacheprovider `
  tests/test_p2_api_contract.py tests/test_p2_api_adversarial.py
$env:DESKPILOT_RUN_DOCKER_SECURITY_TESTS = '1'
$env:DESKPILOT_TEST_DOCKER_IMAGE = 'python:3.11-alpine'
.\.conda\deskpilot-py311\python.exe -m pytest -q -rs -p no:cacheprovider `
  tests/test_p2_docker_sandbox.py tests/test_p2_sandbox_api.py
```

上述测试仍不能替代目标部署主机的授权文件读写 broker、symlink/junction 越界、容器逃逸、TLS/网关、监控和人工操作验收。桌面 Python/Shell 工具及通用 Agent HTTP 工具在这些边界完成前保持关闭。
