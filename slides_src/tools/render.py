"""Aperçu hors ligne strict des slides : JS coupé, réseau coupé, vraies polices ; signale débordements et chevauchements.

    python slides_src/tools/fetch_fonts.py            # une fois : polices dans slides_src/fonts/
    python slides_src/tools/render.py slides_src slides_src/fonts/local.css OUT_DIR [ids,séparés,par,virgules]
"""
import asyncio
import json
import os
import sys

from playwright.async_api import async_playwright

DECK, FONTCSS, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
ONLY = sys.argv[4].split(",") if len(sys.argv) > 4 and sys.argv[4] else None
BASE = """* { margin: 0 } body { font: 32px/1.2 sans-serif }
body > section { width: 1920px !important; height: 1080px !important; overflow: hidden !important; box-sizing: border-box !important; position: relative; isolation: isolate }
section[hidden], aside, noscript { display: none !important }
x-shape, x-icon, x-embed { display: block }"""
CHECK = r"""() => { const out = [];
 for (const s of document.querySelectorAll('body > section')) { const sr = s.getBoundingClientRect(); const T = [];
  for (const el of s.querySelectorAll('p,h1,h2,h3,div')) { const r = el.getBoundingClientRect(); if (!r.width || !r.height) continue;
   if (r.right > sr.right + 1 || r.bottom > sr.bottom + 1 || r.left < sr.left - 1 || r.top < sr.top - 1) out.push(`${s.id}: hors cadre <${el.tagName}> "${(el.textContent||'').trim().slice(0,40)}"`);
   if (['P','H1','H2','H3'].includes(el.tagName)) { if (el.scrollWidth > el.clientWidth + 2) out.push(`${s.id}: texte qui déborde "${el.textContent.trim().slice(0,40)}"`); T.push([el, r]); } }
  for (let i = 0; i < T.length; i++) for (let j = i + 1; j < T.length; j++) { const [a, ra] = T[i], [b, rb] = T[j]; if (a.contains(b) || b.contains(a)) continue;
   const ox = Math.min(ra.right, rb.right) - Math.max(ra.left, rb.left), oy = Math.min(ra.bottom, rb.bottom) - Math.max(ra.top, rb.top);
   if (ox > 3 && oy > 3) out.push(`${s.id}: chevauchement "${a.textContent.trim().slice(0,30)}" / "${b.textContent.trim().slice(0,30)}"`); } }
 return out; }"""


async def main():
    idx = json.load(open(os.path.join(DECK, "project/deck.json")))
    order = [s for s in idx["order"] if not ONLY or s in ONLY]
    secs = "".join(open(os.path.join(DECK, f"project/slides/{s}.html")).read() for s in order)
    page = (f"<!doctype html><html><head><meta charset='utf-8'><style>{BASE}</style>"
            f"<link rel='stylesheet' href='/fonts/local.css'><link rel='stylesheet' href='/project/deck.css'></head>"
            f"<body>{secs}</body></html>")
    fonts_root = os.path.dirname(os.path.dirname(os.path.abspath(FONTCSS)))
    async with async_playwright() as p:
        b = await p.chromium.launch(args=["--host-resolver-rules=MAP * ^NOTFOUND"])
        ctx = await b.new_context(java_script_enabled=False, viewport={"width": 1920, "height": 1080})
        nav = {"n": 0}

        async def route(r):
            path = r.request.url.split("://", 1)[1].partition("/")[2]
            if r.request.is_navigation_request():
                nav["n"] += 1
                return await (r.fulfill(body=page, content_type="text/html") if nav["n"] == 1 else r.abort("aborted"))
            for prefix, base in (("project/", DECK), ("fonts/", fonts_root)):
                if path.startswith(prefix):
                    fp = os.path.realpath(os.path.join(base, path))
                    root = os.path.realpath(os.path.join(base, prefix))
                    if fp.startswith(root) and os.path.isfile(fp):
                        ct = "text/css" if fp.endswith(".css") else "font/woff2"
                        return await r.fulfill(path=fp, content_type=ct)
            await r.abort("aborted")

        await ctx.route("**/*", route)
        pg = await ctx.new_page()
        await pg.goto("http://deck.local/")
        await pg.evaluate("() => document.fonts.ready")
        await pg.wait_for_timeout(500)
        await pg.evaluate("""() => document.querySelectorAll('body > section').forEach(s => { let flow = false;
            for (const c of s.children) { if (getComputedStyle(c).position !== 'absolute') flow = true; else if (!flow) c.style.zIndex = -1; } })""")
        msgs = await pg.evaluate(CHECK)
        print("\n".join("!! " + m for m in msgs) or "aucun débordement ni chevauchement")
        os.makedirs(OUT, exist_ok=True)
        for i, s in enumerate(await pg.query_selector_all("body > section")):
            await s.scroll_into_view_if_needed()
            await pg.wait_for_timeout(150)
            await s.screenshot(path=os.path.join(OUT, f"{idx['order'].index(order[i]) + 1:02d}_{order[i]}.png"),
                               animations="disabled")
        await b.close()


asyncio.run(main())
