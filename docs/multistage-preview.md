# 多阶段真实点云预览

Java 新任务默认开启多阶段预览。30 步计划是：

| 实际完成步数 | VAE 解码分辨率 | 表面采样点数 |
| --- | --- | --- |
| 12 | 64 | 1000 |
| 18 | 96 | 2500 |
| 24 | 128 | 5000 |

其他采样步数使用 `floor(steps × 0.4/0.6/0.8)`，限制到 1～steps-1，重复步数合并。尊重 Java 原有 previewResolution、pointCount 上限。只有新任务生成计划；已保存任务继续使用自己的参数快照。

## 协议和恢复

Python `parameters` 新增可选 `previewStages`：

```json
{"seed":1234,"steps":30,"previewEnabled":true,"previewStages":[{"step":12,"resolution":64,"pointCount":1000},{"step":18,"resolution":96,"pointCount":2500},{"step":24,"resolution":128,"pointCount":5000}]}
```

最多三个阶段，step 必须严格递增且小于 steps，resolution 仅 64/96/128，pointCount 为 1～5000。明确传入计划时不使用 previewStep/previewResolution/pointCount 单阶段参数；未传计划保持原单阶段协议，不自动改变旧客户端。`previewEnabled=false` 不生成预览。

事件协议仍为 v1。每次成功保存递增 previewVersion；失败不占版本。因此 version=1 不一定来自计划第一阶段。每次 preview_ready.sourceStep 和 pointCount 必须与创建任务时持久化的计划匹配，不按固定第 24 步或固定 5000 点校验。

每个成功阶段使用独立 preview_step_NNN.json/PLY 保存。`GET /api/v1/tasks/{id}/previews/{version}` 从持久事件日志查对应版本，完成后仍可读取早期版本。首快照只包含最新成功版本，不做事件历史重播。Java 恢复时按 1～latestVersion 补齐尚未入库的版本，每版独立持久存储；Java 浏览器 HTTP 接口保持原路径和鉴权。

## 真实性和内存边界

每个阶段在采样回调线程里克隆该步真实 `outputs.prev_sample`，经完整 VAE 导出与零等值面提取，再使用局部 RNG 从真实三角面按面积采样。坐标不变，不用最终模型降采样代替早期结果，不把 latent 当 XYZ。预览同步暂停后续采样，并增加总解码开销，不声称减少总推理时间。

预览最多 128 分辨率、5000 点、2000 numChunks，最终解码仍受原 128/256 和任务 numChunks 限制。MPS 临时 decode 引用释放后清 allocator cache，不禁用内存水位保护，不并行解码或同时加载第二模型。普通无表面/解码失败发非致命 preview_failed，保留上一版并继续；设备 OOM、设备异常或全局 RNG 变化按原逻辑终止，不假装预览成功。

## 真实验收（2026-11-18）

使用已有单个 mini/MPS 服务、004.png、seed=1234、30 步、最终 256、numChunks=4000，通过 Java 随机端口原生浏览器 WS 协议与独立 formaspace_multistage_live_test 数据库完成：

- 初始候选 6/15/24 中，第 6 步 64 分辨率解码场约 -0.9976～-0.9932，无真实零等值面，发送 preview_failed 后继续；没有制造早期点云。因此默认调整为 12/18/24。
- 12/18/24 均成功，真实点数 1000/2500/5000；解码耗时约 0.82/2.58/6.06 秒，三个阶段随机状态指纹均未变化。
- 完整点云与每包 50 点的浏览器通道一致；首版后断线领取新票据恢复；三版 Java HTTP positions 逐点一致。
- result_saving 后作品提交，再发送 task_completed；最终 GLB 7,158,596 字节，198,828 顶点、397,656 三角面；任务总耗时约 70.73 秒，没有发生 OOM。
- 实际点云的固定投影视图如下，未做形状插值；早期稀疏和变形是采样中间状态，不保证任意输入都在第一阶段形成可识别表面。

![第 12、18、24 步的真实点云](assets/multistage-real-point-clouds.png)

Python 34 项受控测试通过，覆盖三阶段原始 latent/预算、连续成功版本、早期普通失败后继续及全部版本 HTTP 恢复。Java 187 项默认回归：163 通过、24 按环境跳过；包括持久计划、旧协议、步数去重、非法点数/版本拒绝与断线补全三版。真实模型测试日志 `/tmp/formaspace-multistage-live.log`，最终复验 `/tmp/formaspace-multistage-live-final.log`。

React 已实现新版逐步显示时保留旧版并渐隐，最多两层 GPU 资源；首次坐标归一化保持不变，OrbitControls 不因版本切换重建，最终 GLB 加载后替代点云。前端 build/lint 通过，但未执行真实浏览器 UI 自动化或远程 TLS/代理验收。这里的真实 WS 验收使用带 Origin 的 JDK 原生客户端。
