"""读 GGUF 的 tensor 类型，按层统计字节数，确认它是不是混合量化。

用途：切分前算「我这几层到底占多少」。整模大小除以层数只在单一量化下成立；
一旦某些层是 Q8_0、embedding 是 F16，那个平均值会骗人。

    python scripts/inspect_gguf_quant.py <model.gguf>
"""

import sys
from collections import Counter, defaultdict

import gguf


def bytes_of(tensor) -> int:
    """tensor 的字节数。用 gguf 的量化尺寸表，不读数据本体。"""
    t = tensor.tensor_type
    block, size = gguf.quants.GGML_QUANT_SIZES[t]
    return tensor.n_elements // block * size


def main() -> int:
    if len(sys.argv) != 2:
        print(__doc__)
        return 2
    reader = gguf.GGUFReader(sys.argv[1])

    layer_bytes: dict[int, int] = defaultdict(int)
    layer_types: dict[int, Counter] = defaultdict(Counter)
    outside: dict[str, int] = defaultdict(int)
    all_types: Counter = Counter()

    for tensor in reader.tensors:
        n = bytes_of(tensor)
        all_types[tensor.tensor_type.name] += n
        parts = tensor.name.split(".")
        if len(parts) >= 3 and parts[0] == "blk" and parts[1].isdigit():
            idx = int(parts[1])
            layer_bytes[idx] += n
            layer_types[idx][tensor.tensor_type.name] += n
        else:
            outside[tensor.name] += n

    total = sum(layer_bytes.values()) + sum(outside.values())
    print(f"文件：{sys.argv[1]}")
    print(f"总字节：{total:,} = {total / 2**30:.2f} GiB")
    print(f"\n量化类型分布（按字节）：")
    for name, n in all_types.most_common():
        print(f"  {name:<10} {n / 2**30:>8.3f} GiB  {n / total * 100:>5.1f}%")

    if not layer_bytes:
        print("\n没有 blk.* 张量，看不出分层。")
        return 0

    layers = sorted(layer_bytes)
    sizes = [layer_bytes[i] for i in layers]
    avg = sum(sizes) / len(sizes)
    spread = (max(sizes) - min(sizes)) / avg * 100 if avg else 0
    print(f"\n分层字节（{len(layers)} 层，块 {layers[0]}..{layers[-1]}）：")
    print(f"  平均 {avg / 2**30:.4f} GiB/层，最大 {max(sizes) / 2**30:.4f}，"
          f"最小 {min(sizes) / 2**30:.4f}")
    print(f"  离散度 {spread:.2f}% —— "
          f"{'基本均匀' if spread < 1 else '不均匀，平均值不能用'}")

    # 只列出与平均不同的层，均匀时不刷屏
    outliers = [(i, layer_bytes[i]) for i in layers
                if abs(layer_bytes[i] - avg) > avg * 0.01]
    if outliers:
        print(f"\n  偏离平均 >1% 的层：")
        for i, n in outliers:
            kinds = ", ".join(f"{k}×{v / 2**30:.3f}G" for k, v in
                              layer_types[i].most_common(3))
            print(f"    blk.{i:<3} {n / 2**30:.4f} GiB  ({kinds})")

    if outside:
        print(f"\n层外张量（embedding / output / norms）：{sum(outside.values()) / 2**30:.3f} GiB")
        for name, n in sorted(outside.items(), key=lambda kv: -kv[1])[:5]:
            print(f"  {name:<32} {n / 2**30:.4f} GiB")

    # 切分参考：前 k 层的累计字节
    print(f"\n前 k 层累计（切分时看这个，不要看 k/40×总量）：")
    acc = 0
    for k, i in enumerate(layers, start=1):
        acc += layer_bytes[i]
        if k % 8 == 0 or k == len(layers):
            print(f"  0:{k:<3} {acc / 2**30:>7.2f} GiB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
