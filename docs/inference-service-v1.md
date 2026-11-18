# FormaSpace Python HTTP / WebSocket v1

本轮只实现 Python。新服务默认 `http://127.0.0.1:8001` / `ws://127.0.0.1:8001`，与已有 Gradio、旧 api_server.py 是不同入口。现有 Gradio 源码保留，但不要同时运行多个入口占用同一 GPU。所有本文 Path 均相对于 **Python 服务**，不相对于 Java 应用，跨机器不共享本地文件路径。

推理与真实预览已提取到 `hy3dgen/shapegen/preview.py`，验证脚本与常驻服务共享实现。`ServiceJournal` 将原验证脚本的 preview_saved 转为 v1 的 preview_ready，去掉本机路径、补充完整 positions/UTC 时间/protocolVersion；task_started/preprocessing 由 worker 在预处理前写入。两种入口均补充最后采样后的 finalizing，Gradio 默认链路不变。

## 启动

复用现有 `.venv`。唯一新增依赖为 `websockets==15.0.1`（Uvicorn 原生 WebSocket transport 与验收客户端）；本机已仅安装此包，没有升级其他依赖。FastAPI、Uvicorn、Pydantic、python-multipart、httpx 等均使用已有版本。

```sh
uv pip install --python .venv/bin/python --no-deps websockets==15.0.1
```

先在部署环境设置独立随机的 `INFERENCE_API_TOKEN`，Python 和 Java 使用同一值。不要将令牌提交仓库或写日志。未配置令牌时入口立即失败，不加载模型。本机联调令牌已保存于仓库外、权限 0600 的 `~/.config/formaspace/inference.env`，不在本文展示。

Mac 单行启动命令（已 export 令牌的 shell）：

```sh
PYTORCH_ENABLE_MPS_FALLBACK=1 HF_HUB_OFFLINE=1 .venv/bin/python scripts/inference_server.py --host 127.0.0.1 --port 8001
```

本机使用已有令牌文件时可先执行 `set -a`、`source ~/.config/formaspace/inference.env`、`set +a`，再运行上面命令。`HF_HUB_OFFLINE=1` 要求权重已缓存，不触发 Hugging Face 下载；没有删除模型或去背景缓存。MPS fallback 在导入 torch 前通过环境变量开启。

入口固定 `workers=1, reload=False`，模型及 BackgroundRemover 在唯一推理线程中各初始化一次，后续任务复用；预处理、采样、同步预览、最终解码都在该线程中。不同机器可修改 host/port、Java baseUrl 和服务凭证，不依赖共享目录。部署监听非回环地址时，应在受控网络/反向代理后提供 HTTPS/WSS。

配置：

| 配置 | 默认 | 说明 |
| --- | --- | --- |
| INFERENCE_API_TOKEN | 无，必填 | HTTP / WS 共享 Bearer 令牌 |
| INFERENCE_STORAGE_ROOT | ~/.formaspace/inference-tasks | 请求元数据、源图、事件、快照、点云、GLB、内部日志 |
| INFERENCE_DEVICE | mps | 当前实测 MPS；不切换模型 |
| INFERENCE_QUEUE_CAPACITY | 4 | 等待队列容量，不含正在执行的一项 |
| INFERENCE_SUBSCRIBER_CAPACITY | 64 | 每订阅者有界缓冲，含线程到 asyncio 的桥接 |
| INFERENCE_SUBSCRIBER_LIMIT | 32 | 服务总订阅连接上限 |
| --host / --port | 127.0.0.1 / 8001 | 命令行监听配置 |

默认图片上限 10 MiB、1600 万像素、JPEG/PNG；总 multipart 体积额外允许 64 KiB 字段/头部，流式读取时强制限制，不能绕过 Content-Length 限制。接收文件、图片 verify/load、磁盘操作在 HTTP 线程池执行，模型工作不占用 FastAPI 事件循环。发送单条 WS 消息的超时为 10 秒。

## HTTP

健康检查 `/healthz` 与 `/readyz` 无需令牌。其他 HTTP 和 WS 都必须包含：

```text
Authorization: Bearer <INFERENCE_API_TOKEN>
```

`GET /healthz`：200 `{ "alive": true }`。不加载新模型、不执行推理。

`GET /readyz`：模型加载完成、worker 存活且队列未满时 200，否则 503。返回 ready、modelId、device、queueDepth。模型加载中仍可访问 health、已持久化的快照与结果；设备错误使 worker 不可用时，需要重启服务，不会继续不安全推理。

`POST /api/v1/tasks`：multipart 的 taskId 为 Java 生成 UUID，image 为图片内容，parameters 为可选 JSON 字符串。新任务立即返回 202 和 queued；重复任务返回 202 及当前状态。不会等待生成结束。

默认参数：

```json
{"seed":1234,"steps":30,"guidanceScale":5.0,"finalResolution":256,"previewEnabled":true,"previewStep":24,"previewResolution":128,"pointCount":5000,"numChunks":4000}
```

范围：seed=0～10000000、steps=2～50、guidanceScale=0～10（有限数值）、finalResolution=128/256、previewResolution=64/128、pointCount=1～5000、numChunks=1000～8000。未传阶段计划且开启预览时必须 `1 <= previewStep < steps`，未传 previewStages 时每个任务最多一次预览；显式阶段计划支持最多六版，见 [多阶段真实预览](multistage-preview.md)。未知字段、非布尔 previewEnabled、非整数整数字段、NaN/Infinity 均拒绝。

固定模型 `tencent/Hunyuan3D-2mini/hunyuan3d-dit-v2-mini`，无纹理，使用与 Gradio 默认路径相同的 RGBA 转换、去背景开启、pipeline ImageProcessorV2。新协议没有 removeBackground/generateTexture 字段，不要直接透传旧 Gradio 参数。quality 由 Java 自己映射，不能作为 Python 参数发送。

幂等指纹为原始图片字节 SHA256 与填充默认值后的规范化参数；文件名、multipart MIME 声明、JSON 键顺序不影响指纹。并发重复提交受同一锁保护，只写一份任务、只入队一次。相同 UUID 改图或改参数为 409；原图字节不同，即使视觉一样，也视为不同请求。队列满 429，尚未接受新任务，客户端可保留同一 taskId 后续重试。已接收任务不随 HTTP 或 WS 断开取消。

新任务响应：

```json
{"taskId":"<UUID>","status":"queued","snapshotPath":"/api/v1/tasks/<UUID>","eventsPath":"/api/v1/tasks/<UUID>/events"}
```

`GET /api/v1/tasks/{taskId}` 返回快照：

```json
{
  "taskId":"<UUID>","status":"running","sequence":28,"stage":"sampling",
  "completedSteps":24,"totalSteps":30,"previewVersion":1,
  "latestPreview":{"version":1,"sourceStep":24,"pointCount":5000,"positionsPath":"/api/v1/tasks/<UUID>/previews/1"},
  "result":null,"error":null
}
```

status 为 queued/running/completed/failed。latestPreview 没有时为 null；完成 result 为 `{ "modelPath":"/api/v1/tasks/<UUID>/result", "format":"glb" }`；失败 error 为 `{ "code":"INFERENCE_FAILED", "message":"简短业务提示" }`。不会返回本机路径或 traceback。

`GET /api/v1/tasks/{taskId}/previews/{version}`：返回 taskId、previewVersion、sourceStep、pointCount、positions。坐标有限，长度等于 pointCount；与最终模型使用相同坐标系，无逐预览居中缩放。未知版本或未就绪 404。

`GET /api/v1/tasks/{taskId}/result`：完成后返回 GLB，`Content-Type: model/gltf-binary`。任务不存在 404，任务存在但结果未就绪 409；仅读取受控 UUID 任务目录，不接受任意文件路径。

curl 示例（令牌已导入环境；不要把真实令牌写入命令历史）：

```sh
TASK_ID=$(uuidgen | tr '[:upper:]' '[:lower:]')
curl -sS -X POST http://127.0.0.1:8001/api/v1/tasks -H "Authorization: Bearer $INFERENCE_API_TOKEN" -F "taskId=$TASK_ID" -F 'image=@assets/example_images/004.png;type=image/png' -F 'parameters={"seed":1234,"steps":30,"previewStep":24}'
curl -sS "http://127.0.0.1:8001/api/v1/tasks/$TASK_ID" -H "Authorization: Bearer $INFERENCE_API_TOKEN"
curl -sS "http://127.0.0.1:8001/api/v1/tasks/$TASK_ID/previews/1" -H "Authorization: Bearer $INFERENCE_API_TOKEN" -o preview.json
curl -f "http://127.0.0.1:8001/api/v1/tasks/$TASK_ID/result" -H "Authorization: Bearer $INFERENCE_API_TOKEN" -o final.glb
```

## WebSocket

原生 JSON 文本帧：`ws://127.0.0.1:8001/api/v1/tasks/{taskId}/events`，握手携带相同 Authorization。非 STOMP/SockJS，客户端不需要发启动消息。鉴权无效、taskId 无效/不存在或超过订阅总上限时，在 accept 前拒绝握手，Uvicorn 对应 HTTP 403；可先用 HTTP 查询区分 401/404。不会通过 WS 启动或取消模型。

每个连接第一条都是 task_snapshot。控制消息不占新 sequence，payload 是该时刻完整 HTTP 快照。获取快照和登记后续订阅在同一锁下完成，后续事件 sequence 必须大于快照 sequence。终态任务也会先发最终快照，然后以 1000 关闭；完成/失败事件发出后正常连接同样以 1000 关闭。

统一帧：

```json
{"protocolVersion":1,"taskId":"<UUID>","sequence":28,"type":"preview_ready","stage":"sampling","completedSteps":24,"totalSteps":30,"previewVersion":1,"timestamp":"2026-11-17T08:24:25Z","payload":{}}
```

| type | stage | payload |
| --- | --- | --- |
| task_queued | queued | {} |
| task_started | preprocessing | {} |
| sampling_step | sampling | timestep 数值 |
| preview_started | preview_decoding | {} |
| preview_ready | sampling | sourceStep、pointCount、完整 positions、positionsPath、decodeSeconds |
| preview_failed | sampling | code=PREVIEW_DECODE_FAILED、简短 message、fatal |
| finalizing | final_decoding | {} |
| task_completed | completed | modelPath、format=glb |
| task_failed | failed | code、简短 message |

第 30 步完成后立即产生 finalizing，任务仍为 running；最终解码/GLB 原子保存成功后才产生 task_completed。不能将 30/30 当作文件已完成。

预览仍从所选步骤的 `outputs.prev_sample.detach().clone()` 经完整 `_export` 解码（含 VAE scale_factor），同步暂停后续采样约数秒。使用独立 RNG 按面积采点，不用 latent 直接伪造 XYZ，也不拿最终模型代替中间结果。点云 PLY/JSON 成功保存后才递增 previewVersion，首次为 1。

普通预览失败发送 fatal=false 的 preview_failed，version 保持不变，继续最终生成。GPU 内存/设备或状态错误先记录 preview_failed，再 task_failed，并停止接收新任务；等待任务标记 WORKER_UNAVAILABLE。完整堆栈保存在 service.log，内部 summary 另存诊断状态，不通过接口发送。

推理线程先 fsync events.jsonl，再更新并原子保存 snapshot.json，最后通知异步发送层。线程到异步桥接是有界 deque，合并唤醒通知，不为每条事件无限积累 asyncio 回调。GPU 回调不等待网络；慢连接缓冲溢出或发送超时以 **1013** 关闭，客户端必须通过快照恢复，不能当作收到完整终态。正常断线仅注销订阅，不取消任务。

WebSocket 心跳使用 `IsolatedHeartbeatProtocol`：继承当前 Uvicorn transport 的超时、Ping/Pong 和关闭行为，但用 `secrets.token_bytes` 生成 Ping 内容。安装版本的 websockets legacy 默认心跳调用进程全局 `random.getrandbits`；它可能与预览并发，导致 RNG 审计误判。必须使用本文启动脚本，不能改回默认 `ws="websockets"`。没有关闭心跳、禁用 RNG 审计或回滚全局随机状态；内部 summary 现额外记录 changedRandomGenerators，真实推理污染 RNG 时仍失败。

2026-11-18 已补心跳回归：直接调用安装版本的默认 ping 可重现 Python 全局 RNG 变化；新 transport 连续 ping 与显式 payload/重复保护测试通过。另以真实 Uvicorn + WebSocket、10 ms 心跳和受控慢预览验证心跳实际落在预览窗口内，任务仍完成，RNG 审计保持通过；人为注入 NumPy 全局随机消费仍被拒绝，并记录 changedRandomGenerators=[numpy]。共 29 项测试通过，无依赖升级。

还使用失败任务 8 的原图片字节和完整参数（seed=5920516、30 步、24 步预览、256/128、5000 点、numChunks=4000）执行真实 MPS 复验，新 Python 任务 `4117b700-120c-41ea-a936-60532ae5876c` 成功生成点云并完成 GLB 下载（6,137,504 字节）；预览解码 5.71 秒，RNG 状态未变。该长连接收到 3 个服务端心跳，但本次真实 MPS 的预览窗口恰好未与心跳重叠；强制窗口重叠由前述网络回归覆盖。报告位于 `~/.formaspace/rng-heartbeat-check-20261118/summary.json`，不改写 Java 原失败任务状态。

## 重连、持久化与 Java 接入

Java 端本轮未修改、未运行本协议联调；当前旧 Gradio 客户端不会自动切换到新服务。这是新增协议适配，不是对 Gradio 的路径别名。

Java 开发者需要：

1. 在现有数据库中先建立任务并生成 UUID；以同一 taskId、同一图片字节和规范化参数 POST。超时后可重复 POST 或 GET 协调，禁止生成另一个 taskId 自动重做。
2. 收到 202 后以后台原生 WebSocket 客户端连接 eventsPath，所有 HTTP/握手使用服务令牌；baseUrl、token、连接/读取/任务截止时间配置化，不能长期占用 Servlet 请求线程。
3. 首先处理 task_snapshot；保存 sequence 与 previewVersion。增量事件按 sequence 去重；发现跳跃或 1013 时关闭重连，使用有上限的指数退避。无需等待完整历史重放，服务首版只保证当前快照加后续事件。
4. 恢复快照若有新 previewVersion，用 positionsPath 拉取完整点云；preview_ready 直接带完整 5000 点，不是增量。任务跑到 final_decoding 时继续等待。
5. completed 后使用相对 Python 的 modelPath 下载 GLB，验证并保存到 Java 自有存储；浏览器不访问 Python 内部地址或服务令牌。重连即 completed 时也补做结果下载；failed 时同步错误。
6. 断线显示恢复中，不能立即把业务任务判失败；完成或失败后释放 WS。保留前端转发扩展点，本轮不要求前端改变。

每个任务目录包含 request.json（图片指纹/参数）、input.image、events.jsonl、snapshot.json、点云、final.glb、内部 summary.json。结果保存成功先于完成事件；事件日志是重启恢复的依据，不依赖内存 Map。事件 sequence 跨重启延续，截断进程崩溃留下的不完整末行；完整中间事件损坏则启动失败，不猜测或伪造恢复。

重启不恢复 GPU 计算：遗留 queued/running 任务追加 SERVICE_RESTARTED 的 task_failed，保留成功预览和原 sequence；completed 任务保持可查询和下载。终态不被后续事件回退。相同 taskId 重交只返回原终态，不重新计算。普通优雅停止会等待当前任务结束，尚未执行的 queued 任务在下次启动失败；突然进程退出也按此持久记录恢复。存储根目录使用排他文件锁，拒绝多个服务进程共同运行。

错误体统一为 `{"error":{"code":"...","message":"..."}}`。常见 HTTP 错误：401 UNAUTHORIZED、404 TASK_NOT_FOUND/PREVIEW_NOT_FOUND、409 IDEMPOTENCY_CONFLICT/RESULT_NOT_READY、413 IMAGE_TOO_LARGE、422 INVALID_REQUEST/INVALID_PARAMETERS/INVALID_IMAGE、429 QUEUE_FULL、503 INFERENCE_UNAVAILABLE。任务错误另有 INFERENCE_FAILED、SERVICE_RESTARTED、WORKER_UNAVAILABLE；预览错误 PREVIEW_DECODE_FAILED。

## 验证命令

离线协议与共享逻辑测试，不加载真实模型：

```sh
.venv/bin/python -m unittest discover -s tests -v
```

真实网络验收客户端（服务必须已经启动，环境包含同一令牌）：

```sh
.venv/bin/python scripts/check_inference_service.py --base-url http://127.0.0.1:8001 --image assets/example_images/004.png --output-dir /tmp/formaspace-service-check
```

此命令真实创建两个串行任务、在第 20 步断开/重连、接收 5000 点、预览后再次重连并经 HTTP 恢复点云、检查 final_decoding、完成后再订阅、重复/冲突提交，并跨 HTTP 下载两份 GLB。输出目录必须不存在。它是 **Python 客户端验收，不是 Java 联调**。

## 本机执行记录（2026-11-17）

已执行 18 项 unittest，全部通过：原始预览/scale_factor/RNG 测试、鉴权/参数/图片和总上传限制、并发重复提交、队列满 429、串行执行、20 步断线/预览快照恢复、终态订阅、final_decoding、普通预览异常、设备异常使 worker 不可用、慢 WS 1013、有界桥接缓冲、初始化 readiness、排他进程锁、重启恢复和日志不完整尾行处理。测试使用假推理器和真实共享采点代码，不启动模型；保留 Gradio 6 已有的资源/参数弃用警告，没有为消除警告升级依赖。

另已启动真实 Mac/MPS 常驻服务，使用缓存 mini 非 turbo、004.png、seed=1234、30 步、24 步预览、final=256、preview=128、5000 点、numChunks=4000：

| 验收 | 实际结果 |
| --- | --- |
| 首次提交立即返回 | 两个新任务分别约 87 ms / 18 ms，202 |
| 两任务队列 | 第二项最初 queued；第一项完成后才开始 preprocessing |
| 初次 WS | 首帧 task_snapshot，随后持久 sequence 单调递增 |
| 20 步断开 | 推理继续；重连快照反映当前实际步骤 |
| 第 24 步预览 | 收到完整 5000 点，decodeSeconds≈6.74，RNG 状态指纹未改变 |
| 预览后重连 | snapshot.latestPreview.version=1，HTTP positions 与 WS 完全一致 |
| 30/30 | final_decoding、status=running，结果下载返回 409 |
| 最终完成 | 两份真实 GLB 经 HTTP 下载成功，Content-Type 正确，GLB header/长度检查通过 |
| 关闭/开启预览对照 | 下载后的两份模型均 198,828 顶点/397,656 面，顶点/面数组完全相同 |
| 完成后再连接 | 首帧直接给 completed 快照与 result 路径 |
| 重复与冲突提交 | 同 taskId 同内容 202 原终态；改 seed 返回 409，未重复运行 |

真实任务 ID 为 `06358ae1-3e43-49c1-933c-725628361eb4`、`70250995-d900-4c08-9d28-08c90afc286d`。两者内部完整推理/导出约 70.74 秒（有预览）与 67.10 秒（无预览），这是单图功能验证，不能将差值全部归因于预览或宣称性能基准。

还做了真实服务崩溃恢复：新任务 `77ae7551-33e0-49c6-a719-84a5b4f00a00` 在第 24 步成功持久化预览后，向本轮启动的服务进程发送 SIGKILL；重启后 status=failed、error.code=SERVICE_RESTARTED，sequence 28→29，previewVersion=1、5000 点仍可下载。加载模型期间 readyz 为 503，但已有完成任务快照/两份 GLB 可正常读取。重启后模型重新就绪。

真实网络产物：`~/.formaspace/inference-communication-check-20261117/` 下 summary.json、received-events.jsonl、ws-preview.json、http-preview.json、两份下载 GLB 和 restart-check.json。服务自己的可恢复任务目录位于 `~/.formaspace/inference-tasks/`；令牌不在这些报告中。

未执行：Java 原生 WS 客户端联调、前端点云转发、远端 GPU/TLS/代理部署、多机并发或真实 GPU OOM。普通预览失败、队列压力和慢订阅者使用受控假推理器验证，没有将其冒充真实 GPU 故障测试。本轮没有修改 Java 或前端，没有自动替换 Java 的旧 Gradio 连接配置。

文件职责：`scripts/inference_server.py` 启动；`inference_service/app.py` HTTP/WS；config/parameters 配置校验；store 持久化/快照；subscriptions 有界通知；worker 单线程队列；engine 真实推理与协议适配；`hy3dgen/shapegen/preview.py` 共享真实预览；`scripts/check_inference_service.py` 显式真实网络验收；`tests/test_inference_service.py` 离线协议验收。`scripts/preview_probe.py` 保留对照命令，requirements.txt 仅增加 websockets 固定版本。

## No surface found 与 Gradio 的 None.faces

`marching_cubes` 的 `No surface found at the given iso value` 表示当前标量场在指定等值面下未能生成表面。仅凭该异常不能确定是图片、参数还是数值异常，也不能据此自动更换等值面。稀疏解码器可能有意保留未评估区域的 NaN，不能对所有非有限区域直接填零。

MCSurfaceExtractor 现于失败时记录 level、resolution、网格 shape、有限值数量、有限值 min/max，并保留原始异常链。外层仍保留批量失败项为 None 的已有约定；Gradio 在读取 faces/vertices 前复用 require_mesh 校验，空容器、None 或无效网格会显示明确的“未能提取有效模型表面”提示，不再触发二次 AttributeError。每次 Gradio 推理还会记录实际固定 seed、模型和解析后的参数，便于复现。

排查时保存输入图片、实际 seed（随机模式应看日志中的最终值）、步数、guidance、分辨率、num_chunks、去背景设置以及完整解码日志。有限值数量为 0 时重点排查数值异常；范围不包含目标 level 时说明当前场无法在该 level 提取表面；范围包含目标值也不保证一定有可用表面。不会自动调整阈值、换模型或返回伪造 GLB。

堆栈含 gradio/queueing.py、gradio_app.py 时走的是 Gradio 入口，通常为 7860，并非本文 8001 常驻服务。修改 Python 后需要重启对应入口才能加载新诊断；两个入口不要同时占用 GPU。

2026-11-18 修复验证：24 项离线测试通过，含无表面/全 NaN 标量场、正常平面坐标、批处理 None 约定，以及 Gradio 无效网格提示和有效网格回归。重启原 mini/MPS/disable_tex Gradio 并加载本次修复后，004.png、seed=1234、30 步、guidance=5、resolution=256、num_chunks=4000、去背景开启的真实对照成功：198,828 顶点、397,656 面，GLB 7,159,044 字节。对照不代表原失败图片已验证；原图片与实际参数尚未提供，不能宣称原始无表面问题的根因已定位。当前日志 `/tmp/formaspace-gradio-surface.log`，客户端对照日志 `/tmp/formaspace-gradio-surface-recheck.log`。
