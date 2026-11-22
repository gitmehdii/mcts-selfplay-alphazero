"""Télécharge Space Grotesk, Inter et JetBrains Mono (Google Fonts) dans slides_src/fonts/ pour l’aperçu."""
import os
import re
import urllib.request

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(HERE, "fonts")
URL = ("https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400..700&family=Inter:wght@400;600;700"
       "&family=JetBrains+Mono:wght@500;700&display=swap")
UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120 Safari/537.36"

os.makedirs(OUT, exist_ok=True)
css = urllib.request.urlopen(urllib.request.Request(URL, headers={"User-Agent": UA})).read().decode()
faces = []
for subset, body in re.findall(r"/\* (.*?) \*/\s*@font-face \{(.*?)\}", css, re.S):
    if subset not in ("latin", "latin-ext"):
        continue
    fam = re.search(r"font-family: '([^']+)'", body).group(1)
    w = re.search(r"font-weight: ([\d ]+);", body).group(1)
    url = re.search(r"url\((https://[^)]+)\)", body).group(1)
    fn = f"{fam.replace(' ', '')}-{w.replace(' ', '_')}-{subset}.woff2"
    urllib.request.urlretrieve(url, os.path.join(OUT, fn))
    ur = re.search(r"unicode-range: ([^;]+);", body)
    faces.append(f"@font-face {{ font-family: '{fam}'; font-weight: {w}; src: url('/fonts/{fn}') format('woff2'); "
                 f"unicode-range: {ur.group(1) if ur else 'U+0-10FFFF'}; }}")
open(os.path.join(OUT, "local.css"), "w").write("\n".join(faces))
print(len(faces), "polices dans", OUT)
