import struct, sys, json

path = r"H:\qlh_models\dflash\dflash-merged-Q4_K_M.gguf"
f = open(path, "rb")
assert f.read(4) == b"GGUF"
ver = struct.unpack("<I", f.read(4))[0]
n_tensors = struct.unpack("<Q", f.read(8))[0]
n_kv      = struct.unpack("<Q", f.read(8))[0]
print("gguf version", ver, "tensors", n_tensors, "kv", n_kv)

def rd_str():
    n = struct.unpack("<Q", f.read(8))[0]
    return f.read(n).decode("utf-8", "replace")

def rd_val(t):
    if t == 0: return struct.unpack("<B", f.read(1))[0]
    if t == 1: return struct.unpack("<b", f.read(1))[0]
    if t == 2: return struct.unpack("<H", f.read(2))[0]
    if t == 3: return struct.unpack("<h", f.read(2))[0]
    if t == 4: return struct.unpack("<I", f.read(4))[0]
    if t == 5: return struct.unpack("<i", f.read(4))[0]
    if t == 6: return struct.unpack("<f", f.read(4))[0]
    if t == 7: return bool(struct.unpack("<B", f.read(1))[0])
    if t == 8: return rd_str()
    if t == 9:
        et = struct.unpack("<I", f.read(4))[0]; n = struct.unpack("<Q", f.read(8))[0]
        return [rd_val(et) for _ in range(n)]
    if t == 10: return struct.unpack("<Q", f.read(8))[0]
    if t == 11: return struct.unpack("<q", f.read(8))[0]
    if t == 12: return struct.unpack("<d", f.read(8))[0]
    raise ValueError("type %d" % t)

kv = {}
for _ in range(n_kv):
    k = rd_str(); t = struct.unpack("<I", f.read(4))[0]
    kv[k] = rd_val(t)

tensors = []
for _ in range(n_tensors):
    name = rd_str()
    n_dims = struct.unpack("<I", f.read(4))[0]
    dims = [struct.unpack("<Q", f.read(8))[0] for _ in range(n_dims)]
    ttype = struct.unpack("<I", f.read(4))[0]
    off = struct.unpack("<Q", f.read(8))[0]
    tensors.append((name, dims, ttype))

GGML = {0:"f32",1:"f16",2:"q4_0",3:"q4_1",6:"q5_0",7:"q5_1",8:"q8_0",9:"q8_1",10:"q2_K",11:"q3_K",12:"q4_K",13:"q5_K",14:"q6_K",15:"q8_K",39:"q4_K_M?"}

print("\n-- target/dflash metadata --")
for k in sorted(kv):
    if any(s in k for s in ("dflash", "target", "lilicorr", "embedding", "general.arch", "block")):
        v = kv[k]
        if isinstance(v, list) and len(v) > 16:
            v = "<list[%d]>" % len(v)
        print(" ", k, "=", v)

print("\n-- tensors (%d) --" % len(tensors))
for name, dims, t in tensors:
    print("  %-42s %-6s %s" % (name, GGML.get(t, t), dims))

print("\nlilicorr tensors:", [n for n, _, _ in tensors if "lilicorr" in n.lower()])
print("lilicorr metadata:", [k for k in kv if "lilicorr" in k.lower()])
