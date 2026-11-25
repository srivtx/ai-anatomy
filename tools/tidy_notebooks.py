"""Post-pass on black output (run last): bin-pack exploded print() calls; blank lines between top-level steps."""
import io, re, sys, tokenize, pathlib
W = 100
BLOCK = re.compile(r"^(for|while|if|with|try|async)\b")
CONT = re.compile(r"^(else|elif|except|finally)\b|^[\)\]\}]")

def protected(src):
    keep = set()
    for t in tokenize.generate_tokens(io.StringIO(src).readline):
        if t.end[0] > t.start[0] and t.type != tokenize.NL:
            keep.update(range(t.start[0] + 1, t.end[0] + 1))   # 1-based lines inside a multi-line token
    return keep

def pack(lines):
    out, i = [], 0
    while i < len(lines):
        m = re.match(r"^(\s*)print\($", lines[i])
        if m:
            ind = m.group(1); j = i + 1; args = []
            while j < len(lines) and re.match(re.escape(ind) + r"    \S", lines[j]) and lines[j].endswith(","):
                args.append(lines[j].strip()); j += 1
            if len(args) > 2 and j < len(lines) and lines[j] == ind + ")" and not any(a.startswith("#") for a in args):
                rows, cur = [], ""
                for a in args:
                    cand = f"{cur} {a}" if cur else a
                    if cur and len(ind) + 4 + len(cand) > W: rows.append(cur); cur = a
                    else: cur = cand
                rows.append(cur.rstrip(","))
                out += [ind + "print("] + [ind + "    " + r for r in rows] + [ind + ")"]
                i = j + 1; continue
        out.append(lines[i]); i += 1
    return out

def space(lines, keep):
    out, cell, prev = [], None, "blank"
    for n, ln in enumerate(lines, 1):
        if n in keep or ln.startswith("# %%") or not ln.strip() or cell == "md" or ln[0].isspace():
            if ln.startswith("# %%"): cell = "md" if "[markdown]" in ln else "code"; prev = "blank"
            elif not ln.strip() and n not in keep: prev = "blank"
            elif ln[:1].isspace() and n not in keep and prev == "block-head": prev = "inblock"
            out.append(ln); continue
        if CONT.match(ln): out.append(ln); continue
        kind = "comment" if ln.startswith("#") else "block-head" if BLOCK.match(ln) else "code"
        need = (prev in ("code", "inblock") and kind in ("comment", "block-head")) or (prev == "inblock" and kind == "code")
        if need and out and out[-1].strip(): out.append("")
        prev = "code" if kind == "comment" else kind
        if kind == "comment": prev = "comment"
        out.append(ln)
    return out

for p in map(pathlib.Path, sys.argv[1:]):
    src = p.read_text()
    packed = "\n".join(pack(src.split("\n")))
    p.write_text("\n".join(space(packed.split("\n"), protected(packed))))
