# 真实中间点云最小验证

入口为 `scripts/preview_probe.py`，只在 Python 内运行，不接 Java、前端或 WebSocket，不改变 Gradio 默认行为。

后续常驻服务已复用同一推理实现，见 [HTTP/WebSocket v1 服务](inference-service-v1.md)。共享模块位于 `hy3dgen/shapegen/preview.py`；验证入口保留 `--compare` / `--no-preview`，任务结束前新增 finalizing 事件。下文为最初独立脚本实测记录，当前服务验收记录见服务文档。

## Mac 单行命令

在 `formaspace-inference` 根目录，使用已有 `.venv` 和缓存权重：

```sh
HF_HUB_OFFLINE=1 .venv/bin/python scripts/preview_probe.py --image assets/example_images/004.png --output-dir /tmp/hunyuan-preview-24 --seed 1234 --steps 30 --preview-step 24
```

替换 `--image` 即可使用自己的图片。输出目录必须新建或为空，避免覆盖上一任务；本阶段不支持断线/进程恢复。`HF_HUB_OFFLINE=1` 禁止 Hugging Face 联网下载，当前模型权重须已缓存，去背景所用 rembg 权重也须已存在。脚本不安装依赖、不清理缓存、不加载纹理模型。运行前释放其他推理进程占用的 GPU 资源。

默认配置：`tencent/Hunyuan3D-2mini` / `hunyuan3d-dit-v2-mini`（非 turbo）、MPS、fp16、30 步、guidance=5、固定 seed=1234、去背景开启、最终分辨率 256、预览分辨率 128、num_chunks=4000、预览点数 5000。可显式指定 `--device cpu/cuda`，本轮仅验证当前 Mac MPS 环境。

同图同 seed 的对照命令：

```sh
HF_HUB_OFFLINE=1 .venv/bin/python scripts/preview_probe.py --image assets/example_images/004.png --output-dir /tmp/hunyuan-preview-compare --seed 1234 --steps 30 --preview-step 24 --compare
```

`--compare` 只加载一次 pipeline、预处理一次图片，在同一线程依次执行 baseline（关闭预览）和 with_preview（开启预览）；每次采样创建相同 seed 的独立 CPU `torch.Generator`，不设置全局 seed。单独关闭预览可使用 `--no-preview`。

## 回调与预览语义

已核对本地 `Hunyuan3DDiTFlowMatchingPipeline.__call__`：`scheduler.step` 后得到 `outputs.prev_sample`，然后调用 `callback(i // scheduler.order, t, outputs)`。当前 FlowMatchEulerDiscreteScheduler 的 order=1，所以回调索引 23 对应 completedSteps=24。脚本拒绝 order 不为 1 的 scheduler，并检查连续回调与最终实际步数，避免把配置步数冒充实际步数。

每次回调写 `sampling_step`。只在指定步骤执行 `outputs.prev_sample.detach().clone()`；该张量仍是带噪中间 latent，绝不直接解释为 XYZ，也不使用预测终态替代它。

在原推理线程同步调用 `pipeline._export`，以关键字指定 output_type=trimesh、box_v=1.01、mc_level=0、num_chunks=4000、octree_resolution=128、mc_algo=mc。该本地实现依次执行 `latents / vae.scale_factor`、VAE 前向、latents2mesh、trimesh 转换。没有手工遗漏或重复应用 scale_factor，也不重置 scheduler、不修改模型参数或 latent 原件。

**同步预览会暂停后续采样，直到预览解码与文件保存结束，因此增加总耗时。** 第 24 步解码完成后继续第 25～30 步，最后正常执行 256 分辨率解码并导出 final.glb。没有新增线程并发访问模型。

## 点云与失败处理

解码后检查单图容器、非 None mesh、非空 XYZ 顶点/三角面、有效索引和有限坐标。从网格表面按三角形面积选择面，再按重心坐标均匀采样 5000 点；使用独立 `np.random.default_rng(seed)`。保留原始模型坐标，不为预览居中或缩放。

预览前后观察 Python、NumPy、PyTorch CPU 与当前设备 RNG 的状态指纹，不调用全局 seed 或恢复函数；同时检查 latent 副本未被解码修改。发现随机状态改变时明确失败，不用恢复状态掩盖问题。

普通预览解码/采点/保存异常写完整 traceback 到 probe.log，产生 preview_failed，继续最终采样。SurfaceExtractor 内部可能捕获异常后返回 None，脚本也保存其 stderr traceback，不只记录二次校验错误。无效预览不产生成功事件、不增加 previewVersion，不用最终模型填充预览。

GPU OOM、设备/Metal 错误或状态不再可信时产生 fatal 的 preview_failed 和 task_failed，停止后续采样；不假定清空缓存后可以安全继续。

## 文件与事件

普通模式的输出：

```text
events.jsonl
preview_step_024.ply
preview_step_024.json
final.glb
summary.json
probe.log
```

点云 JSON 包含 `positions: [[x,y,z], ...]`，PLY 为仅含顶点的 ASCII 点云。先成功保存两种文件，再增加 previewVersion 并产生 preview_saved。

每条事件包含 taskId、sequence、type、stage、completedSteps、totalSteps、previewVersion。sequence 单任务严格递增，写后 flush；previewVersion 从 0 开始，本轮最多为 1。preview_saved 额外包含路径、pointCount、decodeSeconds、elapsedSeconds。decodeSeconds 包括 GPU 同步，elapsedSeconds 还包括校验、采点与文件保存。

summary.json 记录模型配置、seed、预览 actualStep/pointCount/耗时/randomStateUnchanged、最终状态和总耗时。普通模式 totalSeconds 含预处理与模型加载，generationSeconds 仅含推理/导出；对照模式每个子任务 totalSeconds 不含共享模型加载，根 summary.totalSeconds 为整个对照耗时。

对照模式的 baseline/ 和 with_preview/ 各自有 events.jsonl、summary.json、final.glb；预览仅存在 with_preview/。根目录还有 comparison.json 与汇总 summary.json。对比解码后的顶点/面是否完全相同；若不同，报告双向最近顶点距离除以 baseline 包围盒对角线的 mean/p99/max。p99≤0.005 仅标为按此度量接近，不宣称网格拓扑或完整表面完全一致。随机状态检查需结合 with_preview.summary.preview.randomStateUnchanged 查看。

## 离线测试

```sh
.venv/bin/python -m unittest discover -s tests -v
```

测试不加载模型、不下载权重，检查真实 `_export` 的 scale_factor 调用、回调索引、同线程、独立 RNG、原坐标系、按面积采点、事件版本、普通异常继续、设备异常停止及无效网格拒绝。真实模型实测结果在完成后记录于下节，离线测试不能替代它。

## 当前 Mac 真实实测（2026-11-17）

已实际运行上述 `--compare` 流程，使用本机缓存 mini 非 turbo 权重、MPS、`assets/example_images/004.png`、seed=1234，所有分辨率/步数/分块参数均为默认值。没有通过 Gradio 发起另一条推理，没有新下载权重或升级依赖。

| 检查 | 结果 |
| --- | --- |
| 模型加载 | 只加载一次，28.31 秒 |
| baseline 生成/导出 | 成功，61.55 秒 |
| with_preview 生成/导出 | 成功，71.77 秒 |
| 实际预览步骤 | 24/30；随后实际记录 25～30 步 |
| 预览点云 | 5000 个有限 XYZ，PLY 和 JSON 均成功保存 |
| 预览解码 | 6.19 秒 |
| 预览完整同步暂停 | 6.36 秒，含采点/校验/保存 |
| 最终网格 | 两次均 198,828 顶点、397,656 三角面 |
| 最终对照 | 顶点、三角面数组逐项完全一致，双向最近顶点距离为 0 |
| 随机状态 | 预览前后 Python / NumPy / Torch CPU / MPS 状态指纹一致 |
| 整项对照 | 165.15 秒，含图片预处理、共享模型加载与两次生成 |

已再次读取磁盘上的两份 final.glb，验证顶点和面相等；验证 PLY 往返读取得到的坐标与 JSON 相符；核对两份事件序列连续递增、sampling_step 均为 1～30，且 previewVersion 仅在成功保存预览后从 0 增至 1。离线 unittest 共 8 项通过（含已有 Gradio 并发检查），无权重加载。

这些是单图、单 seed 的功能和干扰性验证；两次总耗时差异还包含运行波动，不能将其全部解释为预览开销，也不构成质量/性能基准。中间点云来自第 24 步带噪 latent 解码，不能保证与最终表面完全一致。

本次产物位于 `/Users/xiye/.formaspace/preview-probe-20261117/`：`with_preview/preview_step_024.ply`、`with_preview/preview_step_024.json`、`with_preview/final.glb`、两份 events.jsonl/summary.json，以及根目录 comparison.json、summary.json、probe.log。输出保存在仓库外，没有提交生成模型或点云。

本轮新增 `scripts/preview_probe.py`、`tests/test_preview_probe.py` 和本文。工作区原有的 Gradio 串行队列改动及其测试保持原样；本轮没有修改 Gradio、pipeline、scheduler、Java、前端或依赖文件。
