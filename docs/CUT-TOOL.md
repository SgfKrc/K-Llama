# 裁切工具：已退场

> 状态：**已退场**
> 更新日期：2026-11-03
> 适用范围：裁切工件生成器（`cut.py`）退场的原因、仍在生效的 sidecar 契约格式、以及重建时的陷阱；读者为部署与工具维护者。替代物是整模 + `LLAMA_LAYER_RANGE`（`PATCHES.md` 第二代）。

生成器已退场，它写出的 provenance sidecar 仍是活契约（§2）。

---

## 1. 为什么退场

裁切工件是早期唯一的加载方式：每段一个 `.gguf`，只含自己那几层。
现行形态是**整模 + `LLAMA_LAYER_RANGE`**，每台机器持有同一份完整 checkpoint，
加载时只创建、只读取自己那几层。

三个原因：

1. **MTP 头丢了**。裁切件没有 `blk.40`，而投机解码要它——`topology-local-32-8.toml`
   的注释写明这是直接动因。
2. **多一份 1.04 GB 重复张量**（每段都要带 token_embd / output_norm / output）。
3. **工件与整模会不同步**。换了量化版本要重切全部段，且"工件过期"是一类静默故障。

`cut.py`（335 行）已移到 `_archive/tools/`（本地保留，不入库）。它零 import，
不留任何运行时依赖。

---

## 2. 仍在生效的契约：sidecar

`topology.provenance()` 曾读工件旁边的 `<artifact>.qlh.json`，`verify_ranges()` 在一个段
**没有声明 `layer_range`** 时回退到它。那份拓扑与这两个脚本都已随 v2 退场（归档在
`slimarc/prototype`），本分支没有任何代码读 sidecar；本节只剩**格式约定**（下面那张表）在生效。
A/B/C 三个用例读 `archive-q35-2b` 的工件（range 与字节数都从 sidecar 取），D 不读磁盘：
工件归档后 D 照跑，A/B/C 报 SKIP。

格式（`test_topology.ProvenanceTest` 钉住）：

| 字段 | 含义 |
| --- | --- |
| `generator` / `generator_version` | 固定 `"cut.py"` / `1` |
| `mode` | `head` / `middle` / `tail` |
| `source` | 源整模路径 |
| `artifact` | 工件路径 |
| `hidden_size` / `architecture` | 从源模型读 |
| `block_count` / `n_layer` / `nextn_predict_layers` | **源模型**的这三个值 |
| `kept_block_count` | **本段**保留的层数 |
| `source_layer_range` | **本段**在源模型里的 `[lo, hi)` |
| `artifact_size` | 工件字节数，用于发现"文件被换掉但 sidecar 留下了" |
| `first_local_layer_maps_to` | 本段第 0 层在源模型里的层号 |
| `tensors_kept` / `tensors_dropped` | 张量计数 |
| `trim_layers` | head 模式为 0，其余为 `lo` |
| `artifact_contains` | `token_embd` / `output_norm` / `output_weight` / `can_serve_tail` 四个布尔 |

**顶层字段描述源模型，`kept_block_count` / `source_layer_range` / `artifact_contains`
描述本段。** 这个双口径划分必须守住：混写之后下游再也分不清工件出自哪个模型、
覆盖哪一段（存量 head4 工件顶层写 `block_count=33` 而 GGUF 内实际是 4）。

同样的信息也写进 GGUF 的 `qlh.*` 键作为兜底，但验证只读 sidecar——读回 GGUF 键要
约 7 秒/个，因为 `tokenizer.ggml.tokens`（248k 字符串）放在头部，任何读者都要先扫过它。
sidecar 是毫秒级。

---

## 3. 工具退场前的验证结论

**三种模式与旧工具逐项一致**。用 `qwen3-5-9b-local-Q4_K_M.gguf`（5.38 GB，`block_count=33`、
`n_layer=32`、`interval=4`）：

| 模式 | 参数 | 保留/丢弃张量 | 结果 |
| --- | --- | --- | --- |
| head | `--keep-head 4` | 55/387 | 与旧工具一致 |
| middle | `--k 8 --end 24` | 214/228 | 一致 |
| tail | `--k 8` | 336/106 | 一致 |

**fail-closed 生效**：三种非法配置都以退出码 2 拒绝，错误指向具体参数——
`--keep-head 6` 不是 `full_attention_interval(4)` 的整数倍；`--end=22` 同理；
段内层数不是 4 的倍数。

**`output.weight` 的取舍规则**（这条规则对现在的整模形态同样成立）：一个段要能承担
末段职责，需要 `output_norm`，且要么有独立 `output.weight`，要么 tie embeddings
（`token_embd` 兼作 `lm_head`）。判断结果记在 `artifact_contains.can_serve_tail`。

**两个已知陷阱**：字典字面量会立即求值所有 value（head 模式下 `args.end is None` 直接
`TypeError`）；manifest 计数在调用方和函数内各 `len()` 一次导致重复。两者都不影响现行
形态，但重建工具时会再遇到。

---

## 4. 如果要重建

参考实现：`_archive/tools/cut.py`（本机，不入库）。

但在动手之前先问一句：**现在需要切段吗**。现行方案是整模 + `layer_range`：同一个 gguf 能被
多个进程（或多个 context）各按范围加载，**同机多段（v2.1 的 `--split-tail-range`）是一等形态**，`verify/verify.sh`
组 1/3 跑的就是它，收益是省一次跨机往返——所以"同一台机器拆两段"不再是设计错误，那条限制
只属于 v2 的 Python 驱动链（每个节点一段）。切段只在"某台机器必须只持有一段、又不能读整模"
时才需要。真要裁，先看 `PATCHES.md` 第一代与第二代的差别，别把已经解决的 MTP 头问题重新引回来。
