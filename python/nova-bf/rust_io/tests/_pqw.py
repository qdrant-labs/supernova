"""Minimal hand-rolled parquet writer (thrift compact) for adversarial files."""
import struct
# ---- thrift compact ----
def uvar(v):
    out = bytearray()
    while True:
        b = v & 0x7f; v >>= 7
        if v: out.append(b | 0x80)
        else: out.append(b); return bytes(out)
def zz(v): return uvar((v << 1) ^ (v >> 63))
I32, I64, BIN, LIST, STRUCT, BTRUE, BFALSE = 5, 6, 8, 9, 12, 1, 2
class S:
    """struct: list of (id, type, value). value: int / bytes / S / ('list', etype, [items]) / bool"""
    def __init__(self, *fields, long_ids=False): self.fields = [f for f in fields if f is not None]; self.long_ids = long_ids
    def enc(self):
        out = bytearray(); last = 0
        for fid, ty, val in self.fields:
            if ty in (BTRUE, BFALSE, 'bool'):
                ty = BTRUE if val else BFALSE
            d = fid - last
            if 0 < d <= 15 and not self.long_ids: out.append((d << 4) | ty)
            else: out.append(ty); out += zz(fid)
            last = fid
            if ty in (I32, I64): out += zz(val)
            elif ty == BIN:
                b = val.encode() if isinstance(val, str) else val
                out += uvar(len(b)) + b
            elif ty == STRUCT: out += val.enc()
            elif ty == LIST:
                et, items = val
                n = len(items)
                out.append((n << 4) | et) if n < 15 else (out.append(0xF0 | et), out.__iadd__(uvar(n)))
                for it in items:
                    if et in (I32, I64): out += zz(it)
                    elif et == BIN:
                        b = it.encode() if isinstance(it, str) else it; out += uvar(len(b)) + b
                    elif et == STRUCT: out += it.enc()
            elif ty in (BTRUE, BFALSE): pass
            else: raise ValueError(ty)
        out.append(0)
        return bytes(out)
# ---- parquet enums ----
FLOAT, FLBA, INT32 = 4, 7, 1
REQ, OPT, REP = 0, 1, 2
def selem(name, type=None, rep=None, nchild=None, conv=None, tlen=None, logical=None):
    f = []
    if type is not None: f.append((1, I32, type))
    if tlen is not None: f.append((2, I32, tlen))
    if rep is not None: f.append((3, I32, rep))
    f.append((4, BIN, name))
    if nchild is not None: f.append((5, I32, nchild))
    if conv is not None: f.append((6, I32, conv))
    if logical is not None: f.append((10, STRUCT, S((logical, STRUCT, S()))))  # 3=LIST,15=FLOAT16
    return S(*f)
# ---- levels ----
def rle_run(value, count, bw):
    return uvar(count << 1) + int(value).to_bytes((bw + 7) // 8, 'little') if bw else uvar(count << 1)
def packed_run(vals, bw, truncate_to=None):
    assert len(vals) % 8 == 0
    acc = 0
    for i, v in enumerate(vals): acc |= v << (i * bw)
    nbytes = len(vals) * bw // 8
    body = acc.to_bytes(nbytes, 'little')
    if truncate_to is not None: body = body[:truncate_to]
    return uvar((len(vals) // 8) << 1 | 1) + body
def bw_of(m): return m.bit_length()
def rep_levels_rle(n_values, dim):
    out = b''
    for r in range(n_values // dim):
        out += rle_run(0, 1, 1)
        if dim > 1: out += rle_run(1, dim - 1, 1)
    return out
def def_levels_rle(n, maxdef): return rle_run(maxdef, n, bw_of(maxdef))
# ---- pages ----
def page_header(ptype, usize, csize, sub=None, sub_id=None, extra=(), long_ids=False, crc=None):
    f = [(1, I32, ptype), (2, I32, usize), (3, I32, csize)]
    if crc is not None: f.append((4, I32, crc))
    if sub is not None: f.append((sub_id, STRUCT, sub))
    f += list(extra)
    return S(*f, long_ids=long_ids).enc()
def v1_page(values_bytes, n, rep, deflv, encoding=0, maxdef=1, hdr_kw=None, sub_extra=()):
    body = struct.pack('<I', len(rep)) + rep
    if maxdef > 0: body += struct.pack('<I', len(deflv)) + deflv
    body += values_bytes
    sub = S((1, I32, n), (2, I32, encoding), (3, I32, 3), (4, I32, 3), *sub_extra)
    return page_header(0, len(body), len(body), sub, 5, **(hdr_kw or {})) + body
def v2_page(values_bytes, n, rows, rep, deflv, encoding=0, nulls=0, is_compressed=None, hdr_kw=None):
    body = rep + deflv + values_bytes
    f = [(1, I32, n), (2, I32, nulls), (3, I32, rows), (4, I32, encoding), (5, I32, len(deflv)), (6, I32, len(rep))]
    if is_compressed is not None: f.append((7, 'bool', is_compressed))
    return page_header(3, len(body), len(body), S(*f), 8, **(hdr_kw or {})) + body
def dict_page(dict_bytes, n, encoding=0, hdr_kw=None):
    return page_header(2, len(dict_bytes), len(dict_bytes), S((1, I32, n), (2, I32, encoding)), 7, **(hdr_kw or {})) + dict_bytes
# ---- file ----
def write_file(path, schema, chunks, leaf_type=FLOAT, path_in_schema=("e", "list", "element"),
               codec=0, file_num_rows=None, dict_off=None, extra_cols=None, kv=None, cc_extra=()):
    """chunks: list of dict(pages=bytes, n_rows=int, num_values=int, has_dict=bool, dict_len=int,
       override_len/override_dpo/override_dict_off)"""
    out = bytearray(b'PAR1')
    rgs = []
    total_rows = 0
    for c in chunks:
        start = len(out)
        out += c['pages']
        dpo = start + c.get('dict_len', 0) if c.get('has_dict') else start
        md = [(1, I32, leaf_type), (2, LIST, (I32, [0, 3])), (3, LIST, (BIN, list(path_in_schema))),
              (4, I32, codec), (5, I64, c['num_values']), (6, I64, len(c['pages'])),
              (7, I64, c.get('override_len', len(c['pages']))), (9, I64, c.get('override_dpo', dpo))]
        if c.get('has_dict') or 'override_dict_off' in c:
            md.append((11, I64, c.get('override_dict_off', start)))
        cc = S(*[x for x in cc_extra if x[0] < 2], (2, I64, start), (3, STRUCT, S(*md)), *[x for x in cc_extra if x[0] > 3])
        rgs.append(S((1, LIST, (STRUCT, [cc])), (2, I64, len(c['pages'])), (3, I64, c['n_rows'])))
        total_rows += c['n_rows']
    fmd = S((1, I32, 1), (2, LIST, (STRUCT, schema)), (3, I64, file_num_rows if file_num_rows is not None else total_rows),
            (4, LIST, (STRUCT, rgs)), (5, LIST, (STRUCT, [S((1, BIN, k), (2, BIN, v)) for k, v in (kv or {}).items()])) if kv else None, (6, BIN, "pqw-adversarial"))
    f = fmd.enc()
    out += f + struct.pack('<I', len(f)) + b'PAR1'
    open(path, 'wb').write(out)
STD_SCHEMA = [selem("schema", nchild=1), selem("e", rep=OPT, nchild=1, conv=3, logical=3),
              selem("list", rep=REP, nchild=1), selem("element", type=FLOAT, rep=OPT)]
