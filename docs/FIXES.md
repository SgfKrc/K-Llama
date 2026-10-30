# 已修复的缺陷

> 状态：**现行**
> 更新日期：2026-10-29

---

## 1. shim 无法加载 ggml 后端（`count=0`）

**症状**：`llama_model_load_from_file failed`，stderr 显示 `backends loaded: count=0`。
模型文件存在、DLL 齐全，但加载必然失败。

**根因**：`qlh_keep_head_gpu.c` 里用函数指针当模块句柄：

```c
// 错误
DWORD n = GetModuleFileNameA((HMODULE) &qlh_ensure_backends_loaded, self, sizeof(self));
```

`GetModuleFileNameA` 的第二个参数必须是真正的 `HMODULE`。传函数地址进去时它返回 0，
`self` 保持未初始化，于是 `ggml_backend_load_all_from_path()` 拿到垃圾路径、
什么也没加载，而错误被静默吞掉（函数返回值没检查）。

**修复**：

```c
HMODULE self_module = NULL;
if (GetModuleHandleExA(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS |
                       GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                       (LPCSTR) &qlh_ensure_backends_loaded, &self_module)
    && self_module != NULL) {
    DWORD n = GetModuleFileNameA(self_module, self, (DWORD) sizeof(self));
    ...
}
```

`GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS` 是"从模块内某个地址反查模块句柄"的官方做法。

**验证**：

```
[qlh_keep_head] loading backends from self dir=H:\slimarc-proto\shim\bin
load_backend: loaded CUDA backend from ...\ggml-cuda.dll
load_backend: loaded CPU backend from ...\ggml-cpu-haswell.dll
[qlh_keep_head] backends loaded: count=2
```

**教训**：这个 bug 之所以难查，是因为它**不报错**。`count=0` 是唯一线索，而
`ggml_backend_load_all_from_path` 的失败被忽略。加一行 `fprintf` 打出实际路径
立刻就能定位——诊断输出比猜测便宜。

---

## 2. `_putenv_s` 设置的变量，shim 读不到

**症状**：`SA_BACKEND_DIR=(unset)`，尽管 Python 侧已经调用 `ucrtbase._putenv_s`。

**根因**：shim 是独立 DLL，有自己的 CRT 实例。`ucrtbase._putenv_s` 只更新该 CRT 的
私有副本，不写进程环境块，因此跨 DLL 不可见。

**修复**（两处都设，因为不同链接方式的行为不同）：

```python
os.environ[key] = value          # 写进程环境块，所有 DLL 的 getenv() 都能读
_UCRT._putenv_s(key, value)      # 兼容按 CRT 实例取值的构建
```

**注意**：第一版注释写的是"`os.environ` 无效、`_putenv_s` 才有效"，与实测相反。
原实现（旧仓库 `llama_keep_head.py`）用子进程隔离，所以从未遇到这个问题——
子进程的环境由 `subprocess` 的 `env=` 参数完整传递。

---

## 3. 内联 keep-head 后丢失后端目录传递

**症状**：从旧仓库 `sys.path` 注入改为内联 `keep_head.py` 后，`smoke.py` 立刻失败。

**原因**：`engine.py` 构造的 `environment` 里带了 `SA_BACKEND_DIR`，但内联版的
`_apply_native_env` 只有 `_putenv_s` 路径（见缺陷 2），值传不到 shim。

**修复**：见缺陷 2。

---

## 4. 两进程首轮吞吐不可信

**症状**：`chain_client.py --repeat 3` 的第一轮总是 5–6 t/s，后续 15–18 t/s。

**根因**：冷启动。模型页未进 page cache，`pp32` 从 18.3 跳到 38.5 t/s 是直接证据。

**处置**：稳态统计从第 2 轮起。已在 `MEASUREMENTS.md` §1.1 记录。

---

## 5. 后台负载污染测量

**症状**：同一配置在 15–19 t/s 之间波动，单进程基线也从 19.6 掉到 16.2。

**根因**：本机同时运行 DeepSeek Harness（约 2.8 核）与浏览器，而测试负载只有 3 线程，
量级相同。

**处置**：worker 支持 `--cpus` 绑核（`stage_worker.py`），
GPU worker 固定 `0,1`、CPU worker 固定 `2-7`。绑定后波动收窄。
