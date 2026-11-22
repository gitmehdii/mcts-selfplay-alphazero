"""Validation stricte des slides (règles du format Slides) : balises, limites, tailles de texte.

    python slides_src/tools/validate.py slides_src/project/slides
"""
import glob
import os
import re
import sys
from html.parser import HTMLParser

VOID = {"br", "img", "hr"}


class V(HTMLParser):
    def __init__(s):
        super().__init__(convert_charrefs=True)
        s.stack, s.errs, s.n, s.svg, s.absn, s.divd, s.maxd = [], [], 0, 0, [], 0, 0

    def handle_starttag(s, tag, attrs):
        a = dict(attrs)
        st = (a.get("style") or "").replace(" ", "")
        if tag == "svg":
            s.svg += 1
        if s.svg and tag != "svg":
            s.stack.append((tag, s.getpos()))
            return
        s.n += 1
        cls = (a.get("class") or "").split()
        pinned = "position:absolute" in st or any(c in cls for c in ("ax", "lab", "ranks", "foot", "crumb"))
        if pinned and s.absn:
            s.absn[-1] += 1
        if tag in VOID:
            return
        if tag == "div":
            s.divd += 1
            s.maxd = max(s.maxd, s.divd)
            s.absn.append(0)
        s.stack.append((tag, s.getpos()))

    def handle_startendtag(s, tag, attrs):
        s.handle_starttag(tag, attrs)
        if (s.svg and tag != "svg") or tag not in VOID:
            s.handle_endtag(tag)

    def handle_endtag(s, tag):
        if tag in VOID:
            return
        if not s.stack:
            s.errs.append(f"</{tag}> orphelin ligne {s.getpos()[0]} col {s.getpos()[1]}")
            return
        t, pos = s.stack[-1]
        if t != tag:
            s.errs.append(f"</{tag}> ligne {s.getpos()[0]} col {s.getpos()[1]} alors que <{t}> {pos} est ouvert")
            return
        s.stack.pop()
        if tag == "svg":
            s.svg -= 1
        if tag == "div" and not s.svg:
            k = s.absn.pop()
            s.divd -= 1
            if k > 24:
                s.errs.append(f"div avec {k} enfants positionnés (>24)")


total = 0
for f in sorted(glob.glob(os.path.join(sys.argv[1], "*.html"))):
    src = open(f).read()
    sid = os.path.basename(f)[:-5]
    v = V()
    v.feed(src)
    v.close()
    e = list(v.errs)
    if v.stack:
        e.append("non fermés : " + ", ".join(f"<{t}>{p}" for t, p in v.stack))
    if not re.match(r'\s*<section id="%s"' % sid, src):
        e.append("id de section ≠ nom de fichier")
    if not re.search(r"</aside>\s*</section>\s*$", src):
        e.append("<aside> pas en dernier")
    if v.n > 200:
        e.append(f"{v.n} éléments (>200)")
    if v.maxd > 15:
        e.append(f"div imbriqués sur {v.maxd} niveaux (>15)")
    for m in re.finditer(r"font-size:\s*(\d+)px", src):
        if int(m.group(1)) < 24:
            e.append(f"font-size {m.group(1)}px < 24")
    for bad in ("!important", "z-index", "position:fixed", "white-space:pre", "data:"):
        if bad in src:
            e.append("interdit : " + bad)
    total += len(e)
    print(f"{sid:12s} {v.n:3d} él.  div≤{v.maxd:2d}  " + ("OK" if not e else "\n    - " + "\n    - ".join(e)))
print("TOTAL PROBLÈMES :", total)
