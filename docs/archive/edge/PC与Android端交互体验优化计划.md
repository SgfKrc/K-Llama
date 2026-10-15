# PC 与 Android 端交互体验优化计划

> 状态：历史参考 / 待拆分
>
> 更新日期：2026-07-14
> 适用范围：PC 与 Android 交互体验优化的历史记录与待拆分事项，不作为当前能力说明
>
> 本文混合记录了已完成、部分完成和未实施事项，不能据标题或验收清单推断当前能力。新的分布式方向以[混合分布式推理体系规划](../distributed/混合分布式推理体系规划.md)和[Android 版本远期计划](Android版本远期计划.md)为准；剩余交互事项后续应迁移为可独立关闭的 issue 后归档本文。

## 背景

当前 PC 端、Android 端和 Android 本地推理链路已经基本打通，但实际试用中暴露出多类体验与产品形态问题：Android 端交互仍不够成熟，控制面板相对 PC 版缺失较多功能；PC/Android 双端缺少复制回答、应用内日志查看等常用能力；同时后续还需要区分普通 Android 版和手机极简版，并为独显 PC 版增加实验用多模型能力。

本文记录已发现的待优化点、影响范围、推荐方案和验收标准，供后续迭代实现。

## 总体目标

1. 改善 Android 聊天输入体验，避免软键盘遮挡输入框或对话内容。
2. 为 PC 和 Android 双端增加一键复制回答功能。
3. 补齐 Android 版缺失的 PC 版用户设置能力，避免控制面板过度阉割。
4. 在普通 Android 版之外，后续额外推出手机用“极简版”，保留当前简化功能，并尽量压缩 APK、运行时和模型存储占用。
5. 独显 PC 版增加多模型实验支持，允许加载更重模型做效果对比。
6. PC 和 Android 均改为静默日志模式，提供应用内查看/清理日志入口。
7. 明确 Android 节点能力边界：不主动参与 llama.cpp 层间拆分；如果接收任务，也只能作为完整推理 Worker，且仅普通版考虑，极简版不参与。
8. 远期研究复杂问题的任务级推理链拆分，将复杂问题拆成多个完整子任务并分发给 Full Inference Worker。
9. 总 README 去除涉及成员真名的信息，满足导师合规要求。
10. PC 独显版引入 Gerrit 风格的主节点转让/备用主节点审查机制，仅 PC 独显版节点可投票，合计 >= +2 通过。
11. Android 普通版与极简版使用独立签名密钥和不同包名，避免相互干扰。

---

## 优化项 1：Android 软键盘遮挡聊天界面

### 现象

Android 端进入聊天页后，用户点击底部输入框唤起软键盘时，当前界面不会自动随键盘抬升，导致：

- 输入框可能被键盘遮挡；
- 最新对话内容不能自动滚动到可见区域；
- 用户需要手动收起键盘或拖动列表，体验不友好。

### 影响范围

- Android 客户端聊天页；
- 主要涉及 `ChatScreen` 输入栏、消息列表和 Scaffold / Column 布局；
- 与全有模式 / 全无模式无关，属于通用 UI 问题。

### 推荐方案

在 Android Compose 聊天页中适配 IME Insets：

1. 根布局或聊天内容容器使用 `imePadding()`，让底部输入栏随软键盘上移。
2. 消息列表使用 `LazyColumn` 并配合 `BringIntoViewRequester` 或滚动控制，在新消息/键盘弹出时保持最后消息可见。
3. 输入区域使用 `navigationBarsPadding()` + `imePadding()`，避免同时被系统导航栏和软键盘遮挡。
4. 检查 Activity 是否使用合适的 `windowSoftInputMode`。如果当前 edge-to-edge 设置导致默认 resize 失效，需要在 Compose 侧显式处理 WindowInsets。

### 可能涉及文件

- `android/app/src/main/java/com/qlh/inference/ui/ChatScreen.kt`
- `android/app/src/main/java/com/qlh/inference/MainActivity.kt`
- `android/app/src/main/AndroidManifest.xml`

### 验收标准

- 点击输入框后，输入框始终位于软键盘上方；
- 键盘弹出时，最后一条消息或正在输入区域不被遮挡；
- 发送消息后，消息列表自动滚动到底部；
- 横屏/三键导航/手势导航下均不出现明显遮挡。

---

## 优化项 2：PC 和 Android 缺少一键复制回答功能

### 现象

当前用户如果想复制 AI 的某一条回答，需要手动选中文本。PC 端和 Android 端都没有明确的“复制回答”按钮，对长回答尤其不方便。

### 影响范围

- PC Web 前端聊天消息气泡；
- Android Compose 聊天消息气泡；
- 不影响后端推理逻辑，只涉及前端/客户端交互。

### 推荐方案

为 assistant 消息增加一键复制入口。

#### PC 端

在 AI 回答气泡中增加复制按钮：

- 鼠标悬停或消息右上角显示“复制”按钮；
- 点击后调用 `navigator.clipboard.writeText(answerText)`；
- 复制成功后短暂显示“已复制”；
- 复制失败时给出提示或回退到旧式复制方案。

#### Android 端

在 AI 回答气泡中增加复制入口：

- 气泡右上角显示复制图标，或长按消息弹出“复制回答”；
- 使用 Android `ClipboardManager` 写入剪贴板；
- 复制成功后通过 Snackbar / Toast 显示“已复制”；
- 仅复制回答正文，不复制 metrics、思考开关状态或内部字段。

### 可能涉及文件

PC 端：

- `frontend/src/components/ChatPanel.jsx`
- 可能涉及 `frontend/src/App.jsx` 中的全局提示 / 状态管理

Android 端：

- `android/app/src/main/java/com/qlh/inference/ui/ChatScreen.kt`

### 验收标准

- PC 端每条 AI 回答旁都有可发现的复制入口；
- Android 端每条 AI 回答可一键复制或长按复制；
- 复制内容与回答文本完全一致；
- 复制成功有明确反馈；
- 用户消息可选是否也支持复制，但最低要求是 assistant 回答支持复制。

---

## 优化项 3：Android 版缺失较多 PC 版用户设置功能

### 现象

Android 版本目前控制面板简化过多，缺少很多 PC 版已有的用户设置能力。对于普通 Android 版而言，过度阉割会导致用户无法调整关键运行参数，也不利于把 Android 端作为完整客户端继续演示。

### 需要补齐的方向

普通 Android 版应尽量对齐 PC 端用户设置，但不强行加入 Android 不具备的分层拆分能力。

建议补齐：

1. **连接设置**：主节点地址、端口、连接测试、连接状态展示。
2. **推理参数**：最大生成 token、temperature、top_p、上下文长度等。
3. **模式切换**：全有模式 / 全无模式状态更清晰，展示当前运行引擎。
4. **模型管理**：SAF 模型目录、已选模型、模型大小、刷新、卸载/释放模型。
5. **日志入口**：查看本地日志、复制日志、清理日志。
6. **基础状态**：设备信息、可用内存、native runtime 是否可用、当前 ABI。

不建议补齐：

- Android 主动参与 PC 版 Transformer 层间拆分分配；
- Android 端复杂节点管理和层覆盖逻辑；
- 重模型实验相关设置。

### 可能涉及文件

- `android/app/src/main/java/com/qlh/inference/ui/SettingsScreen.kt`
- `android/app/src/main/java/com/qlh/inference/MainViewModel.kt`
- `android/app/src/main/java/com/qlh/inference/data/SettingsDataStore.kt`
- `android/app/src/main/java/com/qlh/inference/service/ModelManager.kt`

### 验收标准

- Android 普通版至少覆盖 PC 端常用用户设置；
- 用户不需要回到 PC 才能修改基本推理参数；
- Android 控制面板不再给人“只剩聊天壳”的感觉；
- 不引入 Android 当前无法可靠支持的层间拆分功能。

---

## 优化项 4：Android 普通版与手机极简版分离

### 背景

完成基础交互优化后，Android 应额外推出“手机极简版”。极简版保留当前简化功能，面向普通手机用户，避免复杂控制面板和完整模型目录管理带来的认知负担。同时，极简版应把安装包体积、运行时占用、缓存占用和模型存储占用都作为核心约束，尽量压缩空间占用。

### 产品定位

| 版本 | 定位 | 功能范围 |
|------|------|----------|
| Android 普通版 | 完整客户端 | 全有模式、本地 GGUF、SAF 模型目录、较完整设置、日志查看 |
| Android 极简版 | 手机轻量入口 | 简化 UI、固定/内置模型策略、少量参数、尽量无需管理完整 models 目录、尽量压缩空间占用 |

### 极简版模型策略

极简版将来考虑使用更小/更简化的大模型，只使用 INT4 或等价轻量量化形态。因此：

- 极简版不需要完整 `models/` 目录概念；
- 不要求用户理解 Safetensors / GGUF / 多模型目录；
- 可以只暴露“下载/选择一个推荐模型”或“使用内置推荐模型”的入口；
- 不提供复杂日志查看和清理入口；
- 不提供节点管理、层分配、重模型实验等功能。

### 极简版空间压缩策略

极简版需要把空间占用作为产品约束，而不是只做 UI 简化：

1. **APK 体积压缩**：开启 R8、resource shrink、native strip，只保留必要 ABI，优先 `arm64-v8a`。
2. **依赖裁剪**：不打包普通版才需要的日志查看、完整控制面板、多模型管理、Worker 接收任务等模块。
3. **模型存储压缩**：优先使用单一推荐 INT4 小模型，不要求完整 `models/` 目录结构，不保留多份缓存副本。
4. **运行时缓存控制**：限制 SAF cache / 下载缓存大小，提供自动清理策略，避免长期占用手机存储。
5. **功能入口压缩**：只保留聊天、连接/模式基础设置、必要模型状态提示，隐藏高级参数。
6. **分发物拆分**：极简 APK 与普通 APK 分开构建和分发，不把普通版 native/高级资源带进极简包。

### 技术建议

后续可用 Gradle product flavor 区分：

- `fullRelease`：Android 普通版；
- `liteRelease`：Android 极简版。

不同 flavor 可通过 `BuildConfig` 控制：

- 是否显示高级设置；
- 是否显示日志入口；
- 是否显示完整模型管理；
- 是否允许多模型选择；
- 是否包含 Worker 接收任务能力；
- 是否启用普通版 native/高级资源。

### 验收标准

- 普通版和极简版功能定位清晰；
- 极简版启动路径更短，不暴露复杂控制面板；
- 极简版 APK、缓存、模型存储占用明显低于普通版；
- 极简版不依赖完整 models 目录；
- 极简版不包含 Worker 接收任务能力；
- 普通版仍保留高级设置、日志能力和可选 Worker 能力。

---

## 优化项 5：独显版增加多模型实验支持

### 背景

PC 独显版作为主节点和实验平台，需要支持更重模型，用于测试模型规模、推理效果和性能差异。重模型仅用于实验，不进入集显版、Android 普通版、Android 极简版的功能范围。

### 范围界定

| 版本 | 是否支持重模型实验 | 说明 |
|------|------------------|------|
| PC 独显版 | 是 | 主节点实验平台，可支持多模型/重模型 |
| PC 集显版 | 否 | 保持 GGUF/轻量模型，控制安装包和资源占用 |
| Android 普通版 | 否 | 仅本地轻量 GGUF / 转发 PC |
| Android 极简版 | 否 | 仅推荐小模型/INT4 简化模型 |

### 推荐方案

1. 独显版模型管理支持多个模型配置：
   - 模型名称；
   - 模型路径；
   - 模型类型；
   - 推荐显存；
   - 默认最大上下文；
   - 是否实验模型。
2. 前端设置中增加“实验模型”区域，仅独显版显示。
3. 后端根据配置选择模型加载路径，避免影响集显版默认逻辑。
4. 分发服务器和 README 标注：重模型不随安装包分发，需要用户自行下载。

### 可能涉及文件

- `src/config.py`
- `src/model_module.py`
- `src/model_downloader.py`
- `frontend/src/components/SettingsModal.jsx`
- `frontend/src/components/AdminPanel.jsx`
- `packaging/qlh-cuda.spec`
- `packaging/setup-cuda.iss`

### 验收标准

- 独显版可以选择多个模型配置；
- 集显版/Android 不显示重模型实验入口；
- 未下载重模型时有明确提示，不影响默认模型启动；
- 实验模型的加载失败不会破坏基础推理功能。

---

## 优化项 6：PC / Android 静默日志与应用内日志管理

### 现象与要求

PC 版不应继续显示后台控制台窗口，Android 也不应把日志暴露为用户需要手动查看系统 logcat 的形式。两端都应静默保存日志到本地，并提供应用内一键查看和清理日志功能。

极简版不提供日志查看/清理功能，避免界面复杂化。

### PC 端推荐方案

1. 打包版改为无控制台窗口启动：
   - PyInstaller `console=False` 或使用 windowed bootloader；
   - 后端 stdout/stderr 重定向到本地日志文件。
2. 日志保存到用户可写目录：
   - 安装目录 `logs/`，或 `%LOCALAPPDATA%/QLH-Edge-Inference/logs/`；
   - 按日期滚动，例如 `qlh-YYYY-MM-DD.log`。
3. 前端提供日志入口：
   - 查看最近日志；
   - 复制日志；
   - 清理日志；
   - 打开日志目录。

### Android 端推荐方案

1. 应用内写本地日志文件：
   - `context.filesDir/logs/` 或 `context.cacheDir/logs/`；
   - 按日期或大小滚动。
2. 设置页增加日志管理入口：
   - 查看日志；
   - 复制日志；
   - 分享日志；
   - 清理日志。
3. 保留 `Logcat` 输出用于开发，但用户态问题排查优先依赖应用内日志。

### 极简版限制

- 不显示日志管理入口；
- 仅内部静默记录必要崩溃/错误；
- 如需排查，用户安装普通版或导出系统日志。

### 可能涉及文件

PC 端：

- `packaging/qlh-cpu.spec`
- `packaging/qlh-cuda.spec`
- `packaging/launcher.py`
- `src/api_server.py`
- `frontend/src/components/SettingsModal.jsx`

Android 端：

- `android/app/src/main/java/com/qlh/inference/service/InferenceService.kt`
- `android/app/src/main/java/com/qlh/inference/ui/SettingsScreen.kt`
- 可新增 `android/app/src/main/java/com/qlh/inference/logging/` 相关工具类

### 验收标准

- PC 打包版启动后不再显示后台控制台窗口；
- PC 后端异常能写入本地日志；
- Android 运行时错误能写入应用内日志；
- 普通版提供查看/复制/清理日志；
- 极简版不显示日志入口。

---

## 优化项 7：Android 节点能力边界与任务接收能力研究

### 当前原则

Android 由于 llama.cpp 难以进行 Transformer 层间拆分，当前不参与 PC 集群的主动层分配。也就是说，Android 不作为“某几层 Transformer 的执行节点”加入链式分层流水线。

### 能力边界

Android 如果要接收任务，原则上只能接收**完整推理任务**，不能接收层间拆分任务：

- 可以接收完整 prompt；
- 使用 Android 本地 llama.cpp 完整推理；
- 返回完整回答或流式 token；
- 不接收 hidden states；
- 不执行部分 Transformer 层；
- 不参与 PC 主节点的 layer assignment。

因此，“不参与主动分层分配”不等于“完全不能贡献算力”。Android 普通版后续可以研究作为完整推理 Worker；Android 极简版不考虑接收任务。

### 推荐定位

Android 节点能力按版本区分：

| Android 版本 | 是否接收任务 | 任务类型 | 说明 |
|--------------|--------------|----------|------|
| 普通版 | 可研究 | 仅完整推理任务 | 接收完整 prompt，本机完整 GGUF 推理后返回结果 |
| 极简版 | 不考虑 | 无 | 只作为手机轻量聊天入口，避免耗电、发热和后台占用 |

普通版即使支持 Worker，也必须是用户显式开启，默认不接收后台任务。

### 可能模式

| 模式 | Android 是否适合 | 说明 |
|------|----------------|------|
| 层间拆分节点 | 暂不适合 | llama.cpp 层间拆分困难，需要反向工程/改 native |
| 完整推理 Worker | 仅普通版可研究 | 接收完整请求，本机完整跑 GGUF；极简版不参与 |
| 远程薄客户端 | 已支持 | Android 只负责 UI，PC 集群推理 |
| 本地离线终端 | 已支持/继续完善 | Android 全有模式 |

### P4 细化方案：Android 普通版完整推理 Worker

P4 的目标不是让 Android 加入 PC 端 Transformer 层拆分，而是新增一种独立节点类型：`FullInferenceWorker`。它接收完整文本任务，在 Android 本机用 llama.cpp 跑完整 GGUF 模型，然后把文本结果返回给 PC 主节点。

#### P4 目标

- PC 主节点能发现、登记、展示 Android 普通版 Worker。
- Android 普通版可在用户显式开启后接收完整推理任务。
- Android Worker 只接收 prompt/messages/generation_config，不接收 hidden states、KV cache、logits 或层间中间张量。
- PC 调度器能区分 `layer_worker` 与 `full_inference_worker`，避免 Android 被误纳入 layer assignment。
- Worker 模式默认关闭，且可被用户随时停止。

#### P4 非目标

- 不实现 Android 参与 Transformer 层间拆分。
- 不改 llama.cpp 源码来暴露层级 hidden state。
- 不让 Android 极简版接收任何 PC 主节点任务。
- 不把 Android 作为重模型实验节点。
- 不默认后台常驻运行，避免用户无感耗电、发热和网络暴露。

#### 节点类型划分

| 节点类型 | 典型设备 | 可接收任务 | 不可接收任务 |
|----------|----------|------------|--------------|
| `pc_master` | PC 独显版/集显版 | 用户请求、调度、汇总、本地推理 | 无 |
| `pc_layer_worker` | PC 独显从节点 | PyTorch 层间拆分任务 | Android 完整任务协议 |
| `pc_full_worker` | PC 本地完整模型实例 | 完整推理子任务 | 层间拆分以外的中间张量任务 |
| `android_full_worker` | Android 普通版 | 完整文本推理任务 | layer assignment、hidden states、KV cache |
| `android_lite_client` | Android 极简版 | 无，仅聊天 UI / 远程客户端 | 所有 Worker 任务 |

#### Worker 生命周期

| 状态 | 含义 | 进入条件 | 退出条件 |
|------|------|----------|----------|
| `disabled` | 默认关闭 | 首次安装、用户关闭 Worker | 用户手动开启 |
| `enabled_idle` | 可接收任务但当前空闲 | 模型已加载、网络已配对、温控正常 | 收到任务 / 用户关闭 / 健康检查失败 |
| `running` | 正在执行完整推理 | 接收并确认任务 | 推理完成 / 取消 / 超时 / 异常 |
| `cooling_down` | 暂停接单降温 | 温度过高、连续任务过多 | 冷却时间结束且健康检查通过 |
| `paused` | 用户临时暂停 | 用户点击暂停、低电量、非 Wi-Fi 策略触发 | 用户恢复 / 充电与网络条件恢复 |
| `error` | Worker 不可用 | 模型加载失败、端口失败、协议异常 | 用户修复后重新初始化 |

端侧必须提供明显的前台状态提示：当前是否接单、正在执行哪个任务、预计耗时、停止按钮、最近一次错误。普通版可以用 Android foreground service 承载任务，极简版不包含该 service。

#### 协议草案

注册与心跳必须与现有 PC 分层节点协议区分，建议新增 `worker_kind` 字段。

```json
{
  "node_id": "android-full-9f2c",
  "node_type": "android",
  "worker_kind": "full_inference",
  "app_variant": "full",
  "worker_enabled": true,
  "engine": "llama_cpp",
  "model": {
    "model_id": "qwen-1_8b-gguf",
    "format": "gguf",
    "quant": "Q4_K_M",
    "max_context": 4096
  },
  "capabilities": {
    "stream": true,
    "cancel": true,
    "max_concurrent_tasks": 1
  },
  "health": {
    "battery_percent": 82,
    "charging": true,
    "network": "wifi",
    "thermal_state": "normal"
  }
}
```

完整推理任务只传递文本和生成参数：

```json
{
  "task_id": "full-20260710-0001",
  "task_type": "full_inference",
  "messages": [
    {"role": "user", "content": "请总结这段文本..."}
  ],
  "generation_config": {
    "max_tokens": 256,
    "temperature": 0.7,
    "top_p": 0.9
  },
  "deadline_ms": 60000,
  "stream": false
}
```

返回结果保持文本级别：

```json
{
  "task_id": "full-20260710-0001",
  "status": "completed",
  "text": "总结结果...",
  "metrics": {
    "prompt_tokens": 128,
    "completion_tokens": 180,
    "tokens_per_second": 7.4,
    "elapsed_ms": 24300
  }
}
```

禁止字段：
- `hidden_states`
- `past_key_values`
- `layer_start` / `layer_end`
- `logits`
- 任意 torch tensor 二进制载荷

#### 调度规则

1. `task_type == "layer_forward"` 时，只能选择 PC PyTorch layer worker。
2. `task_type == "full_inference"` 时，才允许选择 `android_full_worker`。
3. Android Worker 必须同时满足以下条件才可接单：
   - `worker_enabled == true`
   - `app_variant == "full"`
   - `engine == "llama_cpp"`
   - 已加载模型且上下文长度满足任务要求
   - 电量、温控、网络策略通过
   - 当前并发数低于 `max_concurrent_tasks`
4. Android Worker 不参与重模型实验，不接收超过本地模型上下文的任务。
5. 任务超时、断连或用户停止时，PC 主节点必须回退到本地完整推理或标记任务失败，不影响现有层拆分流水线。

#### 安全与用户控制

- Worker 模式默认关闭，首次开启需弹出说明：耗电、发热、局域网暴露、模型输出可能较慢。
- 推荐仅允许局域网配对，使用一次性配对码或主节点 token。
- 支持“仅充电时接单”“仅 Wi-Fi 接单”“电量低于阈值暂停”“温度过高暂停”。
- Android 端任务运行期间显示前台通知，并提供停止按钮。
- PC 端 UI 必须标出该节点是“完整任务 Worker”，不能显示为“分层节点”。
- 日志只记录任务 ID、耗时、错误码和模型信息；默认不落完整 prompt，避免隐私问题。

#### P4 实施阶段

| 阶段 | 目标 | 主要改动 | 验收 |
|------|------|----------|------|
| P4.0 协议定稿 | 定义完整任务 Worker 协议 | 新增协议文档 / 更新调度数据结构 | PC 能区分 `worker_kind` |
| P4.1 注册与展示 | Android 普通版可注册为完整 Worker | PC 节点列表、Android 设置开关 | UI 展示为“完整任务节点” |
| P4.2 本机任务接口 | Android 暴露完整推理 HTTP/WebSocket 接口 | `InferenceService` / llama.cpp 调用桥接 | 单任务 prompt -> 文本返回 |
| P4.3 主节点手动分发 | PC 可手动选择 Android Worker 跑完整任务 | `scheduler.py` / `api_server.py` | 手动分发成功，失败可回退 |
| P4.4 自动调度实验 | 满足条件时自动选择空闲 Worker | 调度策略、超时、取消、重试 | 不影响 layer pipeline |
| P4.5 压测与保护 | 温控、电量、断连、并发限制 | Android 前台通知、日志、健康检查 | 长时间运行可控，不误接层任务 |

#### P4 可能涉及文件

- `src/scheduler.py`
- `src/api_server.py`
- `src/tcp_comm.py` 或新增 `src/full_worker_protocol.py`
- `frontend/src/components/AdminPanel.jsx`
- `frontend/src/components/NodesPanel.jsx`
- `android/app/src/main/java/com/qlh/inference/service/InferenceService.kt`
- `android/app/src/main/java/com/qlh/inference/ui/SettingsScreen.kt`
- `android/app/src/main/java/com/qlh/inference/worker/`

### P4 验收标准

- Android 不被错误加入 layer assignment；
- 仅 Android 普通版可选作为完整推理 Worker；
- Android 极简版不显示、不启用 Worker 接收任务能力；
- 主节点 UI 清楚区分“分层节点”和“完整任务节点”；
- Android Worker 默认关闭，开启后只接收 `full_inference` 任务；
- Android Worker 运行期间有前台状态、停止按钮和错误提示；
- 断连、超时、取消、低电量、温控异常都能进入可解释状态；
- 功能关闭时不影响 PC 本地推理和 PC 层拆分推理。

---

## 优化项 8：远期计划 - 复杂问题任务级推理链拆分

### 定位

这是远期计划，不作为当前 Android 普通版/极简版的近期目标，也不替代 PC 端已有的 Transformer 层间拆分流水线。

任务级推理链拆分指的是：把一个复杂问题拆成多个可以独立完整推理的子任务，分发给多个 Full Inference Worker 并行处理，最后由主节点汇总结果。它解决的是复杂任务的并行编排问题，而不是单次 token 生成或单次 model forward 的加速问题。

### 与层拆分的区别

| 维度 | Transformer 层拆分 | 任务级推理链拆分 |
|------|--------------------|------------------|
| 拆分对象 | 模型内部层 | 用户问题 / 推理任务 |
| 节点输入 | hidden states / KV 相关中间状态 | 完整文本 prompt / 子任务描述 |
| 节点输出 | hidden states 或 logits | 文本答案 / 结构化结果 |
| 是否需要完整模型 | 不一定 | 需要完整推理能力 |
| Android 普通版是否适合 | 不适合 | 可作为远期研究 |
| Android 极简版是否适合 | 不适合 | 不考虑 |

### P5 细化方案：复杂问题任务级推理链拆分

P5 建议建立在 P4 的 `FullInferenceWorker` 协议已经稳定之后。它不是“把一次模型推理拆得更快”，而是把一个复杂用户问题编排成多个完整子任务，让多个 Worker 并行完成，再由主节点进行归纳、核对和最终回答。

#### P5 目标

- 主节点能判断一个请求是否适合任务级拆分。
- 主节点能生成结构化任务图，而不是只生成一组散乱 prompt。
- PC / Android 普通版 Full Inference Worker 能并行执行完整子任务。
- 主节点能汇总多个子任务输出，形成单一最终答案。
- 任务链功能可以显式开启/关闭，关闭后不影响普通聊天和 PC 层拆分推理。

#### P5 非目标

- 不替代 Transformer 层拆分流水线。
- 不用于单 token 解码加速。
- 不要求所有 Worker 使用同一模型，但必须在结果里记录模型来源。
- 不让 Android 极简版参与。
- 不默认处理隐私敏感或长上下文原文，除非用户明确允许分发。

### 典型流程

```text
复杂问题
  -> 主节点判断是否适合任务级拆分
  -> 生成 TaskGraph
  -> 按 Worker 能力、上下文长度、负载和用户策略分配子任务
  -> PC / Android 普通版 Full Inference Worker 并行完整推理
  -> 主节点收集子结果
  -> 汇总、去重、冲突检测、必要时追加验证任务
  -> 形成最终回答
```

### 任务图模型

P5 不建议直接把问题简单切成 N 份，而是使用轻量任务图 `TaskGraph` 表示推理链。每个节点都是文本级完整任务，不涉及模型内部张量。

| 节点类型 | 作用 | 是否可并行 | 典型执行位置 |
|----------|------|------------|--------------|
| `decompose` | 将复杂问题拆成子任务 | 否 | PC 主节点 |
| `retrieve_context` | 准备局部上下文、文件片段、用户指定资料 | 可并行 | PC 主节点 |
| `worker_inference` | 子任务完整推理 | 可并行 | PC Full Worker / Android 普通版 Worker |
| `cross_check` | 对多个子结果做互相核对 | 可并行 | PC 主节点或高能力 Worker |
| `aggregate` | 汇总、去重、排序、合成最终答案 | 否 | PC 主节点 |
| `final_review` | 检查最终答案是否遗漏、矛盾或越权 | 否 | PC 主节点 |

任务图草案：

```json
{
  "chain_id": "chain-20260710-0001",
  "mode": "task_chain",
  "user_goal": "比较三种部署方案并给出推荐",
  "limits": {
    "max_subtasks": 5,
    "max_parallel": 3,
    "deadline_ms": 180000
  },
  "nodes": [
    {
      "node_id": "task-a",
      "type": "worker_inference",
      "title": "分析方案 A",
      "input_policy": "summary_only",
      "requires_full_model": true
    },
    {
      "node_id": "task-b",
      "type": "worker_inference",
      "title": "分析方案 B",
      "input_policy": "summary_only",
      "requires_full_model": true
    },
    {
      "node_id": "merge",
      "type": "aggregate",
      "depends_on": ["task-a", "task-b"]
    }
  ]
}
```

### 拆分策略

第一阶段不建议做复杂 Agent 系统，先采用“规则 + 模板 + 人工可解释”的拆分策略：

| 场景 | 拆分方式 | 示例 |
|------|----------|------|
| 多方案比较 | 每个方案一个子任务，再汇总对比 | A/B/C 方案优缺点 |
| 多文档总结 | 每份文档或章节一个子任务，再做总览 | 多份日志、报告、会议记录 |
| 代码审查 | 按文件/模块拆分，再合成风险列表 | 前端、后端、Android 分开审查 |
| 多角度分析 | 按视角拆分 | 技术、成本、风险、用户体验 |
| 争议性结论 | 多 Worker 独立判断，再交叉核对 | 架构路线选择 |

拆分限制：
- 默认最多 3 个子任务，研究模式可扩到 5-8 个；
- 子任务必须能独立完成，不能依赖 Worker 之间直接通信；
- 每个子任务输入必须控制在 Worker 本地模型上下文以内；
- Android Worker 只分配短上下文、低敏感、低优先级子任务；
- 涉及隐私、密钥、个人数据、未保存文件全文时，默认不分发到 Android Worker。

### Worker 选择策略

| Worker 类型 | 适合任务 | 默认优先级 | 限制 |
|-------------|----------|------------|------|
| PC 主节点本地模型 | 汇总、最终审查、高敏感任务 | 最高 | 占用主节点资源 |
| PC Full Worker | 长上下文、复杂子任务、批量并行 | 高 | 需要完整模型加载 |
| Android 普通版 Full Worker | 短上下文、轻量总结、低敏感子任务 | 低 | 默认关闭、温控/电量限制 |
| PC layer worker | 不参与 P5 完整子任务 | 无 | 只用于 Transformer 层拆分 |
| Android 极简版 | 不参与 | 无 | 只做聊天 UI / 远程客户端 |

调度器需要先判断任务种类：
- `layer_forward` -> 只走现有 PC PyTorch 层拆分；
- `full_inference` -> 可走 P4 完整 Worker；
- `task_chain` -> 由 P5 编排器拆成多个 `full_inference` 子任务。

### 汇总与质量控制

P5 的核心风险不是“任务发不出去”，而是“多个子结果合成后变得不可靠”。因此主节点汇总必须显式处理来源、冲突和置信度。

| 汇总步骤 | 目标 | 输出 |
|----------|------|------|
| 归一化 | 把不同 Worker 输出整理成统一结构 | `summary` / `claims` / `risks` / `open_questions` |
| 去重 | 合并重复观点 | 去重后的要点列表 |
| 冲突检测 | 找出互相矛盾的结论 | `conflicts[]` |
| 证据标记 | 标记每条结论来自哪个子任务/Worker | `source_task_ids[]` |
| 追加验证 | 对冲突或低置信点追加 `cross_check` 子任务 | 验证结果 |
| 最终生成 | 输出给用户的一段完整回答 | final answer |

推荐让 Worker 返回结构化结果，减少主节点汇总负担：

```json
{
  "task_id": "task-a",
  "status": "completed",
  "result": {
    "summary": "方案 A 部署简单，但扩展性弱。",
    "key_points": ["部署成本低", "横向扩容困难"],
    "risks": ["高并发下瓶颈明显"],
    "confidence": 0.72,
    "open_questions": ["缺少真实压测数据"]
  },
  "worker": {
    "node_id": "android-full-9f2c",
    "model_id": "qwen-1_8b-gguf"
  }
}
```

### 失败与回退策略

| 故障 | 处理方式 |
|------|----------|
| Worker 超时 | 子任务重试一次，仍失败则回退 PC 主节点或标记缺失 |
| Android 断连 | 取消该子任务，不影响其他子任务 |
| 子任务结果质量差 | 主节点追加验证任务或忽略该结果 |
| 子任务之间冲突 | 进入 `cross_check`，最终回答中说明不确定性 |
| 可用 Worker 不足 | 降级为串行执行或普通单模型回答 |
| 任务链整体超时 | 返回已完成部分 + 未完成说明，或回退单模型 |

### 用户体验设计

- P5 默认隐藏在“研究模式 / 复杂任务模式”中，不作为普通聊天默认路径。
- PC 前端显示任务链进度：拆分中、并行执行中、汇总中、完成/部分失败。
- 用户可展开查看子任务标题、执行节点、耗时、状态，但默认不展示冗长中间输出。
- 用户可取消整个任务链，取消后主节点向所有 Worker 下发 cancel。
- 最终回答需要提示“已使用任务级拆分”，并在调试模式展示 Worker 和模型来源。

### 与 Android Worker 的关系

任务级推理链拆分只考虑 Android 普通版作为 Full Inference Worker，且必须满足：

- Android 普通版只接收完整子任务；
- 不接收 hidden states；
- 不执行部分 Transformer 层；
- 不参与 layer assignment；
- Worker 模式必须用户显式开启，默认关闭；
- Android 极简版不参与任务接收和任务链拆分；
- Android Worker 只执行主节点分配的单个子任务，不参与任务拆分和最终汇总。

### 适用场景

适合：

- 多角度分析；
- 多文档总结；
- 多方案比较；
- 代码审查多个文件；
- 搜索、阅读、汇总类任务；
- 需要多个独立观点再综合的复杂问题。

不适合：

- 普通单轮聊天；
- 单个回答的 token 级加速；
- 单次模型 forward 加速；
- PC 层间拆分流水线的直接替代。

### P5 实施阶段

| 阶段 | 目标 | 主要改动 | 验收 |
|------|------|----------|------|
| P5.0 概念验证 | 手动定义 2-3 个子任务并行执行 | 复用 P4 `full_inference` 接口 | 主节点能收集并汇总文本 |
| P5.1 TaskGraph 数据结构 | 引入 `chain_id`、节点、依赖、状态 | `graph_orchestrator.py` / 新增 `task_chain.py` | 可保存和恢复任务链状态 |
| P5.2 规则拆分器 | 根据场景生成子任务 | prompt 模板 / 简单规则 | 多方案、多文档、代码审查可拆分 |
| P5.3 Worker 调度 | 按能力和策略分配子任务 | `scheduler.py` | Android 只接收合规短任务 |
| P5.4 汇总器 | 汇总、去重、冲突检测 | `api_server.py` / `task_chain.py` | 输出单一最终答案 |
| P5.5 UI 与取消 | 展示任务链进度和取消能力 | PC 前端 | 用户能观察和中止任务链 |
| P5.6 质量评估 | 对比单模型回答与任务链回答 | 测试集 / 人工评审 | 确认收益大于调度成本 |

### 可能涉及文件

- `src/graph_orchestrator.py`
- `src/scheduler.py`
- `src/api_server.py`
- `src/full_worker_protocol.py`
- 可新增 `src/task_chain.py`
- `frontend/src/components/AdminPanel.jsx`
- 可新增 `frontend/src/components/TaskChainPanel.jsx`
- `android/app/src/main/java/com/qlh/inference/service/InferenceService.kt`
- 可新增 PC/Android Worker 协议文档

### 验收标准（远期）

- 主节点能区分“层拆分任务”和“任务级推理链任务”；
- 复杂问题能被拆成多个完整子任务；
- Full Inference Worker 能接收完整子任务并返回文本结果；
- 主节点能汇总多个 Worker 的结果；
- Android 普通版可选参与，极简版不参与；
- Android Worker 不参与拆分和汇总，只执行完整子任务；
- 任务链支持取消、超时、部分失败和降级；
- 最终回答能标记主要来源、冲突和不确定性；
- 该功能关闭时不影响现有 PC 层拆分推理。

---

## 优化项 9：总 README 去除成员真名

### 要求

导师要求总 README 去除涉及真名的信息。需要避免公开出现成员真实姓名。

### 推荐方案

修改根目录 `README.md`：

- 删除或匿名化团队分工表中的成员真名；
- 删除页脚中的成员真名；
- 可保留项目团队、学校、指导教师等非敏感信息；
- 如需保留分工，用“项目负责人 / 成员 A / 成员 B”或“模型优化组 / 分布式组 / 前端文档组”等角色描述代替。

### 验收标准

- 根 README 不再出现成员真实姓名；
- 项目职责说明仍清晰；
- 不影响项目介绍、使用说明和技术文档索引。

---

## 优化项 10：PC独显版主节点转让与备用主节点审查机制

### 背景

PC 独显版已经支持主节点转让和指定备用主节点，但目前缺少变更审查流程。在生产环境中，主节点角色的转让或备用主节点的指定/变更属于高风险操作——错误的主节点配置可能导致推理请求路由失败、丢失会话信息或破坏分布式流水线。

需要一个类似 Gerrit 的审查机制来保护这类操作。

### Gerrit 风格审查机制设计

#### 审查对象

- 主节点角色转让（当前主节点 → 另一节点）；
- 备用主节点的指定、变更或移除；
- 集群拓扑中的关键配置变更（如主节点/备用节点角色切换）。

#### 投票规则

| 投票角色 | 可投 | 说明 |
|----------|------|------|
| 管理员 | -1 / 0 / +1 | 项目管理方，通过邮箱通知审查请求 |
| 独显版软件投票 | -1 / 0 / +1 | 任何运行 PC 独显版的节点均可投票 |
| 软件版本条件 | 仅独显版节点可以投票 | PC 集显版、Android 各版本不参与此审查 |

#### 通过条件

最终得票 >= **+2** 才可通过并执行变更。

典型通过场景：

| 场景 | 管理员 | 独显版节点 A | 独显版节点 B | 合计 | 结果 |
|------|--------|-------------|-------------|------|------|
| 单管理员赞同 | +1 | 0 | 0 | +1 | 不通过 |
| 管理员 + 一个独显节点赞同 | +1 | +1 | 0 | +2 | 通过 |
| 两个独显节点赞同（无管理员） | 0 | +1 | +1 | +2 | 通过 |
| 管理员 + 独显节点各 -1 | -1 | -1 | 0 | -2 | 阻止 |

#### 审查流程

```text
1. 任一独显版节点发起转让/备用主节点变更请求
2. 系统生成审查票，序列化为 JSON 并持久化
3. 通过邮件通知管理员审查链接
4. 所有 PC 独显版节点可在管理面板看到待审票
5. 管理员和独显节点可以投 +1 / 0 / -1
6. 当合计 >= +2 时，变更通过并自动执行
7. 当合计 <= -2 时，变更被阻止并需要重新发起
8. 审查票有超时时间（如 48 小时），超时自动关闭
9. 审查票关闭后，变更不可执行，需要重新发起
```

#### 邮件通知

- 管理员邮箱在配置文件中设定，可以为空；
- 审查请求通过后端 SMTP 或系统通知发送；
- 如果浏览器前端不可访问，管理员可通过 CLI / API 投票。

### 软件版本限制

| 软件版本 | 能否发起审查 | 能否投票 |
|----------|-------------|---------|
| PC 独显版 | 可以 | 可以 |
| PC 集显版 | 不可以 | 不可以 |
| Android 普通版 | 不可以 | 不可以 |
| Android 极简版 | 不可以 | 不可以 |

### 可能涉及文件

- `src/config.py` — 管理员邮箱、审查票超时参数
- `src/scheduler.py` — 主节点转让、备用节点变更任务
- `src/api_server.py` — 审查 REST API（CRUD 审查票、投票）
- `frontend/src/components/AdminPanel.jsx` — 管理面板审查 UI
- `frontend/src/components/SettingsModal.jsx` — 集群配置变更入口
- 可新增 `src/review.py` — 审查票状态机 + 持久化
- 可新增 `src/mailer.py` — 邮件通知封装

### 验收标准

- 主节点转让需经过审查投票才能执行；
- 审查票创建后持久化，不因节点重启丢失；
- 独显版节点可在管理面板看到待审票并投票；
- 合计 >= +2 时变更自动执行，<= -2 时自动阻止；
- 超时审查票自动关闭，不阻塞后续变更；
- PC 集显版、Android 各版本不显示审查 UI 且无法投票。

---

## 优化项 11：Android 普通版与极简版签名分离

### 背景

当前 Android release APK 使用同一个 keystore（`qlh-release.jks`）签名。后续需要区分普通版和极简版两个 APK，如果它们共用同一个签名密钥：

- 两个 APK 的包名相同则无法共存安装；
- 如果包名不同但签名相同，更新时可能出现签名冲突；
- 从安全和分发角度，不同产品线应该使用独立签名。

### 推荐方案

| APK | 包名 | 签名密钥 | 说明 |
|-----|------|---------|------|
| Android 普通版 | `com.qlh.inference` | `qlh-release.jks` | 当前主线签名 |
| Android 极简版 | `com.qlh.inference.lite` | `qlh-lite-release.jks` | 独立签名，包名加 `.lite` 后缀 |

### 为什么签名应该不同

1. **包名不同 + 签名不同 = 完全独立的两个 App**：用户可以同时安装普通版和极简版，互不干扰。
2. **安全隔离**：如果极简版的签名意外泄露，不会影响普通版的 APK 签名安全。
3. **分发通道独立**：Google Play、F-Droid、直接分发的 APK 各自可以用不同签名策略。
4. **未来可能拆分权限/能力**：极简版本后续可能裁剪更多能力（如网络权限、后台服务），不同签名有助于平台区分。

### 实现步骤

1. **生成极简版 keystore**：
   ```bash
   keytool -genkey -v -keystore qlh-lite-release.jks \
     -keyalg RSA -keysize 2048 -validity 10000 \
     -alias qlh-lite-release
   ```
2. **新增 `keystore-lite.properties`**：存放极简版签名信息（同样不提交 git）。
3. **修改 Gradle flavor 配置**：为 `lite` flavor 指定独立的 `signingConfig`。
4. **Android 源码 `build.gradle.kts`** 中使用两个 `signingConfigs` 分别绑定 `full` / `lite` 构建类型。
5. **AndroidManifest** 中 `lite` flavor 的包名改为 `com.qlh.inference.lite`。

### 可能涉及文件

- `android/keystore-lite.properties` — 新增，不提交 git
- `android/qlh-lite-release.jks` — 新增，不提交 git
- `android/app/build.gradle.kts` — product flavor + signingConfig
- `android/app/src/lite/AndroidManifest.xml` — 极简版独立包名
- `android/.gitignore` — 补充 `*lite*.jks` + `keystore-lite.properties`

### 验收标准

- 普通版 APK：签名使用 `qlh-release.jks`，包名 `com.qlh.inference`
- 极简版 APK：签名使用 `qlh-lite-release.jks`，包名 `com.qlh.inference.lite`
- 同一设备上可同时安装普通版和极简版
- keystore 和 `.properties` 均被 `.gitignore` 排除

---

## 难度分级与分期路线

### 难度分级定义

| 难度 | 含义 | 特征 |
|------|------|------|
| 低 | UI/文档/单端小改 | 不改核心推理链路，风险低，验证路径短 |
| 中 | 双端体验或设置补齐 | 涉及前端/Android 状态管理，但不改调度核心 |
| 高 | 打包、日志、版本分层、多模型 | 涉及构建产物、运行模式、版本差异和回归测试 |
| 研究级 | 新调度范式或跨端 Worker | 需要协议、调度器、端侧能力和长期实验验证 |

### 分期路线

| 阶段 | 时间定位 | 优化项 | 难度 | 说明 |
|------|----------|--------|------|------|
| P0 | 立即/短期 | README 去真名、Android 键盘遮挡、PC/Android 一键复制回答 | 低 | 合规和基础交互优先，改动范围小 |
| P1 | 短期 | Android 普通版补齐常用设置、Android/PC 应用内日志查看与清理 | 中 | 提升可用性和问题排查能力 |
| P2 | 中期 | PC 打包版静默后台窗口、日志静默落盘、Android 普通版/极简版 flavor 分离 | 高 | 涉及打包、运行时、版本差异和分发说明 |
| P2.5 | 中期 | Android 普通版/极简版签名分离（独立 keystore + 不同包名） | 中 | 涉及 Gradle flavor、签名配置、.gitignore 更新 |
| P3 | 中期/长期 | PC 独显版多模型/重模型实验支持、主节点转让/备用主节点审查机制 | 高 | 只进入独显版，涉及调度器 + 审查票状态机 + 邮件通知 |
| P4 | 长期 | Android 普通版完整推理 Worker | 研究级 | Android 只接收完整任务，不接收层间拆分任务，默认关闭 |
| P5 | 远期 | 复杂问题任务级推理链拆分 | 研究级 | 多 Worker 子任务并行 + 主节点汇总，不替代 Transformer 层拆分 |

### 四种软件版本的优化边界

| 软件版本 | 近期优化重点 | 不做/暂不做 |
|----------|--------------|-------------|
| PC 集显版 | 静默日志、复制回答、基础设置稳定、轻量 GGUF/CPU 路线 | 重模型实验、Android Worker、任务链主调度 |
| PC 独显版 | 多模型实验、重模型效果对比、主节点转让审查、备用主节点管理、主节点能力增强、远期任务链主调度 | Android 极简化策略、Android 签名管理 |
| Android 普通版 | 键盘适配、复制回答、设置补齐、日志管理、本地 GGUF、独立签名、远期完整推理 Worker | 层间拆分节点、重模型实验、投票审查、任务链拆分/汇总 |
| Android 极简版 | 极限压缩体积和存储占用、保留轻量聊天入口 | 日志管理、Worker 接收任务、完整 models 目录、高级设置 |

---

## 建议实现顺序

1. **合规优先**：总 README 去除成员真名。
2. **输入可用性**：修 Android 键盘遮挡问题。
3. **基础交互**：增加 Android / PC 一键复制回答。
4. **Android 普通版补齐**：补 PC 端常用用户设置、日志入口。
5. **日志体验**：PC 改静默窗口 + 双端应用内日志查看/清理。
6. **产品分层**：规划 Android 普通版 / 极简版 flavor，独立签名与包名。
7. **审查机制**：PC 独显版主节点转让/备用主节点审查票 + 投票（P3 级）。
8. **实验能力**：独显 PC 版多模型/重模型实验支持。
9. **完整任务 Worker 研究**：Android 普通版完整推理 Worker 能力，极简版不参与。
10. **远期任务链编排**：复杂问题任务级推理链拆分，多 Worker 子任务并行 + 主节点汇总。

## 回归测试清单

### Android 普通版

- 打开聊天页，点击输入框，确认输入框不会被键盘遮挡；
- 连续发送多条消息，确认列表自动滚动到底部；
- 长回答生成后，点击复制按钮，粘贴到其他 App 验证内容完整；
- 全有模式和全无模式均测试一次；
- 设置页能修改常用推理参数；
- 应用内能查看、复制、清理日志；
- 远期 Worker 模式默认关闭，开启后只接收 `full_inference` 完整子任务；
- Worker 运行期间有前台通知、停止按钮和错误提示；
- 低电量、非 Wi-Fi、温控异常时能暂停接单；
- 不接收 `layer_forward`、hidden states、KV cache、logits 或 tensor 二进制载荷。

### Android 极简版

- 启动路径短，不显示复杂控制面板；
- APK、运行时缓存、模型存储占用尽量压缩；
- 不要求用户管理完整 models 目录；
- 不显示日志管理入口；
- 不显示 Worker 接收任务入口，也不会接收 PC 主节点任务；
- 保留聊天核心功能。
- 使用独立签名密钥和包名，与普通版无冲突可共存安装。

### PC 集显版

- 打包版启动不显示后台控制台窗口；
- 日志静默写入本地；
- 前端能查看/复制/清理日志；
- 不显示重模型实验入口；
- 不显示审查票投票入口。

### PC 独显版

- 支持默认模型正常推理；
- 可配置实验重模型；
- 未下载实验模型时提示明确；
- 重模型加载失败不影响默认模型；
- 日志静默保存并可在应用内查看；
- 主节点转让需审查投票通过才执行；
- 只有 PC 独显版节点可参与投票；
- 合计 >= +2 通过，<= -2 阻止；
- 超时审查票自动关闭；
- P4 关闭时不展示 Android 完整 Worker 自动调度入口；
- P4 开启时能区分 `layer_worker` 与 `full_inference_worker`；
- P5 关闭时所有聊天请求仍走普通聊天或现有层拆分；
- P5 开启时能创建、取消、超时回退一个 2-3 子任务的任务链。

### P4/P5 研究功能专项

- Android 普通版注册为 `android_full_worker` 后，不出现在 layer assignment 列表；
- PC 主节点向 Android Worker 下发 `full_inference` 任务，能收到文本结果和耗时指标；
- Android Worker 断连、取消、超时后，PC 主节点能回退本地或标记子任务失败；
- 任务链模式下，主节点能生成 `TaskGraph`，并将子任务拆成多个 `full_inference`；
- 子任务结果能被汇总为一个最终回答，并标记冲突、不确定性和来源；
- Android Worker 只执行子任务，不执行任务拆分、交叉审查和最终汇总；
- P4/P5 全部关闭时，现有 PC 本地推理、PC 层拆分推理、Android 全有/全无模式均不受影响。

### README 合规

- 根 README 不再出现成员真实姓名；
- 团队职责以匿名角色或小组方式展示。
