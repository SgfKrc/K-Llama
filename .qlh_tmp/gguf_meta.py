import struct, sys

def rd(f, fmt):
    sz = struct.calcsize(fmt)
    b = f.read(sz)
    if len(b) != sz:
        raise EOFError
    return struct.unpack(fmt, b)[0]

def rd_str(f):
    n = rd(f, "<Q")
    return f.read(n).decode("utf-8", "replace")

def rd_val(f, t):
    if t == 0: return rd(f, "<B")
    if t == 1: return rd(f, "<b")
    if t == 2: return rd(f, "<H")
    if t == 3: return rd(f, "<h")
    if t == 4: return rd(f, "<I")
    if t == 5: return rd(f, "<i")
    if t == 6: return rd(f, "<f")
    if t == 7: return bool(rd(f, "<B"))
    if t == 8: return rd_str(f)
    if t == 9:
        et = rd(f, "<I"); n = rd(f, "<Q")
        return [rd_val(f, et) for _ in range(n)]
    if t == 10: return rd(f, "<Q")
    if t == 11: return rd(f, "<q")
    if t == 12: return rd(f, "<d")
    raise ValueError("type %d" % t)

path = sys.argv[1]
keys = sys.argv[2:] if len(sys.argv) > 2 else None

with open(path, "rb") as f:
    magic = f.read(4)
    ver = rd(f, "<I")
    n_tensors = rd(f, "<Q")
    n_kv = rd(f, "<Q")
    print("magic=%r version=%d n_tensors=%d n_kv=%d" % (magic, ver, n_tensors, n_kv))
    for _ in range(n_kv):
        k = rd_str(f)
        t = rd(f, "<I")
        v = rd_val(f, t)
        if keys is None or any(s in k for s in keys):
            s = repr(v)
            if len(s) > 300:
                s = s[:300] + "..."
            print("%-60s = %s" % (k, s))
