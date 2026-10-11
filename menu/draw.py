"""Draw flat SVG illustrations for menu items that have no photo.

Run: python3 menu/draw.py  -> writes menu/img/*.svg (800x600)
"""
import math, os, random

W, H = 800, 600
OUT = os.path.join(os.path.dirname(__file__), "img")

DEFS = """<defs>
<radialGradient id="plate" cx="50%" cy="45%" r="60%"><stop offset="0" stop-color="#ffffff"/><stop offset=".8" stop-color="#f1efeb"/><stop offset="1" stop-color="#d9d4cc"/></radialGradient>
<linearGradient id="wood" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="#b98458"/><stop offset="1" stop-color="#8e5f3b"/></linearGradient>
<linearGradient id="bread" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="#e9a95a"/><stop offset="1" stop-color="#b8702f"/></linearGradient>
<linearGradient id="glass" x1="0" y1="0" x2="1" y2="0"><stop offset="0" stop-color="#ffffff" stop-opacity=".55"/><stop offset=".25" stop-color="#ffffff" stop-opacity=".12"/><stop offset=".8" stop-color="#ffffff" stop-opacity=".05"/><stop offset="1" stop-color="#ffffff" stop-opacity=".4"/></linearGradient>
<linearGradient id="can" x1="0" y1="0" x2="1" y2="0"><stop offset="0" stop-color="#000" stop-opacity=".25"/><stop offset=".3" stop-color="#fff" stop-opacity=".35"/><stop offset=".45" stop-color="#fff" stop-opacity="0"/><stop offset="1" stop-color="#000" stop-opacity=".3"/></linearGradient>
<filter id="sh" x="-20%" y="-20%" width="140%" height="140%"><feGaussianBlur stdDeviation="10"/></filter>
</defs>"""


def svg(body, bg):
    return f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}">{DEFS}{bg}{body}</svg>'


def table(c1="#cfa57c", c2="#b88a5f"):
    # wooden table with planks
    s = f'<rect width="{W}" height="{H}" fill="{c1}"/>'
    for i, y in enumerate(range(0, H, 120)):
        s += f'<rect y="{y}" width="{W}" height="118" fill="{c2 if i % 2 else c1}"/>'
        s += f'<path d="M0 {y+40} C200 {y+30} 400 {y+55} {W} {y+42}" stroke="#000" stroke-opacity=".06" stroke-width="3" fill="none"/>'
        s += f'<path d="M0 {y+85} C250 {y+95} 500 {y+75} {W} {y+88}" stroke="#000" stroke-opacity=".05" stroke-width="2" fill="none"/>'
        s += f'<rect y="{y+118}" width="{W}" height="2" fill="#000" fill-opacity=".15"/>'
    return s


def cafe(wall="#e8ddd0", top="#a77b55"):
    return (f'<rect width="{W}" height="{H}" fill="{wall}"/>'
            f'<circle cx="650" cy="120" r="160" fill="#fff" fill-opacity=".35"/>'
            f'<rect y="420" width="{W}" height="180" fill="{top}"/>'
            f'<rect y="420" width="{W}" height="8" fill="#000" fill-opacity=".12"/>')


def shadow(cx, cy, rx, ry, o=.25):
    return f'<ellipse cx="{cx}" cy="{cy}" rx="{rx}" ry="{ry}" fill="#000" opacity="{o}" filter="url(#sh)"/>'


def plate(cx=400, cy=310, rx=300, ry=230):
    return (shadow(cx, cy + 18, rx, ry) +
            f'<ellipse cx="{cx}" cy="{cy}" rx="{rx}" ry="{ry}" fill="url(#plate)"/>'
            f'<ellipse cx="{cx}" cy="{cy}" rx="{rx*.72}" ry="{ry*.72}" fill="none" stroke="#000" stroke-opacity=".06" stroke-width="4"/>')


def scatter(n, cx, cy, rx, ry, seed):
    r = random.Random(seed)
    pts = []
    while len(pts) < n:
        x, y = r.uniform(-1, 1), r.uniform(-1, 1)
        if x * x + y * y < 1:
            pts.append((cx + x * rx, cy + y * ry, r.uniform(0, 360), r.random()))
    return pts


def leaf(x, y, a, s, col="#5aa83a", vein="#3f7f25"):
    return (f'<g transform="translate({x:.0f} {y:.0f}) rotate({a:.0f}) scale({s:.2f})">'
            f'<path d="M-40 0 C-25 -28 25 -28 40 0 C25 26 -25 26 -40 0Z" fill="{col}"/>'
            f'<path d="M-36 0 L36 0" stroke="{vein}" stroke-width="3"/></g>')


def arugula(x, y, a, s):
    return (f'<g transform="translate({x:.0f} {y:.0f}) rotate({a:.0f}) scale({s:.2f})">'
            '<path d="M-45 0 L-30 -8 -26 -20 -14 -10 -6 -24 4 -10 14 -22 20 -8 34 -14 45 0 32 10 20 6 12 20 2 8 -8 20 -16 8 -28 14Z" fill="#4c8f2e"/>'
            '<path d="M-42 0 L42 0" stroke="#2f6619" stroke-width="2.5"/></g>')


def tomato_half(x, y, s=1):
    return (f'<g transform="translate({x:.0f} {y:.0f}) scale({s})"><circle r="24" fill="#d8322a"/><circle r="17" fill="#ef5a43"/>'
            '<circle cx="-7" cy="-4" r="4" fill="#f6d36b"/><circle cx="7" cy="-4" r="4" fill="#f6d36b"/><circle cx="0" cy="8" r="4" fill="#f6d36b"/></g>')


def cube(x, y, s, top="#fbf6e9", side="#e9e0c8"):
    return (f'<g transform="translate({x:.0f} {y:.0f}) scale({s:.2f})"><path d="M-16 -6 L0 -14 L16 -6 L0 2Z" fill="{top}"/>'
            f'<path d="M-16 -6 L0 2 L0 18 L-16 10Z" fill="{side}"/><path d="M16 -6 L0 2 L0 18 L16 10Z" fill="{side}" opacity=".8"/></g>')


def walnut(x, y, a):
    return (f'<g transform="translate({x:.0f} {y:.0f}) rotate({a:.0f})"><path d="M-16 0 C-16 -14 16 -14 16 0 C16 12 -16 12 -16 0Z" fill="#8a5a2b"/>'
            '<path d="M-12 -2 C-6 4 -2 -6 0 2 C2 -6 6 4 12 -2" stroke="#5e3b18" stroke-width="2.5" fill="none"/></g>')


def bowl(cx, cy, rx, ry, col="#f4f1ec", rim="#3b6e8f"):
    return (shadow(cx, cy + ry * .9, rx, ry * .5) +
            f'<path d="M{cx-rx} {cy} A{rx} {ry*1.6} 0 0 0 {cx+rx} {cy}Z" fill="{rim}"/>'
            f'<ellipse cx="{cx}" cy="{cy}" rx="{rx}" ry="{ry}" fill="{col}"/>'
            f'<ellipse cx="{cx}" cy="{cy+4}" rx="{rx*.9}" ry="{ry*.82}" fill="#e2ddd4"/>')


# ---------- items ----------

def kabak_cips():
    s = plate()
    for x, y, a, t in scatter(26, 400, 300, 220, 150, 1):
        if t < .55:  # zucchini chip
            s += (f'<g transform="translate({x:.0f} {y:.0f}) rotate({a:.0f})"><ellipse rx="34" ry="30" fill="#c79a3c"/>'
                  '<ellipse rx="29" ry="25" fill="#e8c46a"/><circle r="9" fill="#d9b154" opacity=".7"/>'
                  '<ellipse rx="34" ry="30" fill="none" stroke="#5b7a2a" stroke-width="4"/></g>')
        else:  # jalapeno ring
            s += (f'<g transform="translate({x:.0f} {y:.0f}) rotate({a:.0f})"><circle r="22" fill="#3d8a2c"/><circle r="15" fill="#c9de8b"/>'
                  '<circle cx="-4" cy="2" r="3" fill="#f2f0d0"/><circle cx="5" cy="-3" r="3" fill="#f2f0d0"/><circle r="22" fill="none" stroke="#c88a2d" stroke-width="3" opacity=".6"/></g>')
    s += ('<g transform="translate(600 430)"><ellipse rx="52" ry="24" fill="#ddd" /><ellipse rx="46" ry="19" fill="#f6f1e6"/>'
          '<circle cx="-12" cy="-3" r="3" fill="#7aa64a"/><circle cx="10" cy="2" r="3" fill="#7aa64a"/></g>')
    return svg(s, table())


def peynir_tabagi():
    s = shadow(400, 330, 330, 170)
    s += '<rect x="80" y="150" width="640" height="320" rx="34" fill="url(#wood)"/>'
    s += '<rect x="80" y="150" width="640" height="320" rx="34" fill="none" stroke="#6e4527" stroke-width="6"/>'
    s += '<circle cx="740" cy="310" r="18" fill="#6e4527"/><circle cx="740" cy="310" r="8" fill="url(#wood)"/>'
    for i in range(3):  # cheese wedges
        x = 160 + i * 70
        s += (f'<g transform="translate({x} {250+i*25}) rotate({-15+i*12})"><path d="M0 0 L120 -30 L120 40Z" fill="#f2c94c"/>'
              f'<path d="M120 -30 L132 -24 L132 46 L120 40Z" fill="#e2b53a"/><circle cx="70" cy="2" r="7" fill="#d9a92d"/><circle cx="95" cy="18" r="5" fill="#d9a92d"/></g>')
    for x, y, a, t in scatter(7, 470, 250, 70, 50, 3):  # salami
        s += (f'<g transform="translate({x:.0f} {y:.0f}) rotate({a:.0f})"><circle r="38" fill="#9e2b2b"/><circle r="33" fill="#c0413c"/>'
              '<circle cx="-12" cy="-8" r="5" fill="#f3d3c3"/><circle cx="10" cy="6" r="4" fill="#f3d3c3"/><circle cx="4" cy="-16" r="3" fill="#f3d3c3"/><circle cx="-6" cy="14" r="4" fill="#f3d3c3"/></g>')
    for i in range(3):  # smoked folded slices
        s += (f'<g transform="translate({560+i*40} {370-i*12}) rotate({-20+i*8})"><path d="M-60 0 C-40 -40 -10 30 10 -10 C30 -45 55 20 70 -5 L70 20 C55 45 30 -15 10 20 C-10 55 -40 -15 -60 25Z" fill="#c45a5a"/>'
              '<path d="M-60 25 C-40 -15 -10 55 10 20" stroke="#f0c2b5" stroke-width="5" fill="none"/></g>')
    for x, y, a, t in scatter(9, 330, 410, 90, 30, 4):  # olives
        s += f'<ellipse cx="{x:.0f}" cy="{y:.0f}" rx="14" ry="11" fill="{"#2c2a26" if t < .5 else "#6b7d2a"}"/><ellipse cx="{x-4:.0f}" cy="{y-4:.0f}" rx="4" ry="3" fill="#fff" opacity=".35"/>'
    for x, y, a, t in scatter(8, 650, 220, 40, 30, 5):  # grapes
        s += f'<circle cx="{x:.0f}" cy="{y:.0f}" r="15" fill="#6d3b73"/><circle cx="{x-5:.0f}" cy="{y-5:.0f}" r="4" fill="#fff" opacity=".35"/>'
    return svg(s, table("#3c4a43", "#34413a"))


def karman_corman():
    s = shadow(400, 470, 300, 60, .35)
    s += '<path d="M120 260 L680 260 L620 500 L180 500Z" fill="#a8743f"/>'
    for i in range(6):
        y = 280 + i * 38
        s += f'<path d="M{130+i*10} {y} L{670-i*10} {y}" stroke="#7d5229" stroke-width="5"/>'
    for i in range(14):
        x = 150 + i * 38
        s += f'<path d="M{x} 262 L{x+ (400-x)*.11:.0f} 498" stroke="#7d5229" stroke-width="3" opacity=".6"/>'
    s += '<path d="M150 262 L650 262 L640 250 L160 250Z" fill="#fff" opacity=".9"/><path d="M160 250 L650 250 L600 236 L210 236Z" fill="#e33" opacity=".85"/>'
    r = random.Random(7)
    for i in range(22):  # fries
        x = r.uniform(180, 620); h = r.uniform(110, 190); a = r.uniform(-18, 18)
        s += f'<rect x="{x:.0f}" y="{262-h:.0f}" width="20" height="{h:.0f}" rx="4" fill="#f4c542" stroke="#d99e22" stroke-width="2" transform="rotate({a:.0f} {x:.0f} 262)"/>'
    for x, y in [(250, 210), (330, 190), (520, 200), (600, 225)]:  # onion rings
        s += f'<ellipse cx="{x}" cy="{y}" rx="44" ry="36" fill="none" stroke="#c7832e" stroke-width="18"/><ellipse cx="{x}" cy="{y}" rx="44" ry="36" fill="none" stroke="#e7a64b" stroke-width="8"/>'
    for x, y in [(420, 215), (370, 240), (460, 245), (300, 250)]:  # nuggets
        s += f'<path d="M{x-36} {y} C{x-40} {y-30} {x+30} {y-40} {x+38} {y-5} C{x+42} {y+22} {x-30} {y+30} {x-36} {y}Z" fill="#d48a35"/><circle cx="{x-8}" cy="{y-8}" r="3" fill="#a5601f"/><circle cx="{x+12}" cy="{y+4}" r="3" fill="#a5601f"/>'
    return svg(s, table())


def salad(kind):
    s = bowl(400, 300, 280, 190, rim="#2f5d7c" if kind != "roka" else "#8c4a2f")
    pts = scatter(30, 400, 295, 220, 135, {"tulum": 11, "sezar": 12, "roka": 13}[kind])
    for x, y, a, t in pts:
        if kind == "roka":
            s += arugula(x, y, a, .9 + t * .4)
        else:
            s += leaf(x, y, a, 1 + t * .5, "#6dbb45" if t < .5 else "#8fd15c", "#4b8f2c")
    if kind == "tulum":
        for x, y, a, t in scatter(10, 400, 290, 170, 100, 21):
            s += cube(x, y, 1.3)
        for x, y, a, t in scatter(8, 400, 300, 180, 110, 22):
            s += walnut(x, y, a)
        for x, y, a, t in scatter(5, 400, 300, 160, 90, 23):
            s += tomato_half(x, y, .8)
    if kind == "sezar":
        for x, y, a, t in scatter(7, 400, 290, 150, 90, 31):  # grilled chicken strips
            s += (f'<g transform="translate({x:.0f} {y:.0f}) rotate({a:.0f})"><rect x="-45" y="-14" width="90" height="28" rx="12" fill="#e9c08a"/>'
                  '<path d="M-25 -14 L-35 14 M0 -14 L-10 14 M25 -14 L15 14" stroke="#9b6233" stroke-width="5"/></g>')
        for x, y, a, t in scatter(10, 400, 300, 190, 110, 32):
            s += cube(x, y, 1, "#e6b866", "#c48e3f")
        for x, y, a, t in scatter(10, 400, 300, 190, 110, 33):
            s += f'<path d="M{x:.0f} {y:.0f} l22 -6 l-4 10 z" fill="#fbf2d5" transform="rotate({a:.0f} {x:.0f} {y:.0f})"/>'
        s += '<path d="M260 260 C320 230 360 320 420 280 C480 240 520 330 560 290" stroke="#f6efdc" stroke-width="9" fill="none" stroke-linecap="round" opacity=".9"/>'
    if kind == "roka":
        for x, y, a, t in scatter(7, 400, 300, 180, 100, 41):
            s += f'<circle cx="{x:.0f}" cy="{y:.0f}" r="20" fill="#d8322a"/><circle cx="{x-6:.0f}" cy="{y-6:.0f}" r="5" fill="#fff" opacity=".4"/>'
        for x, y, a, t in scatter(12, 400, 300, 190, 110, 42):
            s += f'<path d="M{x:.0f} {y:.0f} l28 -8 l-6 12 z" fill="#f7ecc8" transform="rotate({a:.0f} {x:.0f} {y:.0f})"/>'
        s += '<path d="M280 240 C340 280 380 220 440 260 C500 300 520 240 560 270" stroke="#5a2a14" stroke-width="7" fill="none" stroke-linecap="round" opacity=".8"/>'
    return svg(s, table("#e7e2d8", "#ddd6ca"))


def sandwich(filling):
    s = shadow(400, 420, 330, 60, .35)
    s += '<path d="M60 360 C60 300 120 280 400 280 C680 280 740 300 740 360 C740 400 690 420 400 420 C110 420 60 400 60 360Z" fill="url(#bread)"/>'
    s += '<path d="M80 340 L720 340" stroke="#fff" stroke-opacity=".25" stroke-width="6"/>'
    if filling == "tavuk":
        for i in range(8):
            s += f'<rect x="{90+i*78}" y="245" width="90" height="38" rx="14" fill="#e3b77d" transform="rotate({(-1)**i*6} {130+i*78} 265)"/><path d="M{110+i*78} 250 l-8 28 M{135+i*78} 250 l-8 28" stroke="#9b6233" stroke-width="5"/>'
    elif filling == "kofte":
        for i in range(6):
            s += f'<ellipse cx="{125+i*110}" cy="262" rx="62" ry="28" fill="#6b3a1f"/><ellipse cx="{115+i*110}" cy="254" rx="40" ry="12" fill="#8a4f2a"/><path d="M{80+i*110} 268 l90 0" stroke="#3d1f0e" stroke-width="5" opacity=".5"/>'
    else:  # kokorec
        r = random.Random(9)
        for i in range(70):
            x = r.uniform(80, 720); y = r.uniform(225, 285)
            s += f'<rect x="{x:.0f}" y="{y:.0f}" width="{r.uniform(14,30):.0f}" height="{r.uniform(10,18):.0f}" rx="5" fill="{r.choice(["#7d4a26","#9a5d2f","#5f3519","#b07040"])}" transform="rotate({r.uniform(-40,40):.0f} {x:.0f} {y:.0f})"/>'
        for i in range(60):
            x = r.uniform(80, 720); y = r.uniform(220, 285)
            s += f'<circle cx="{x:.0f}" cy="{y:.0f}" r="2.5" fill="{r.choice(["#c0392b","#5a7d2a"])}"/>'
    for i in range(7):  # lettuce + tomato
        s += f'<path d="M{70+i*95} 290 C{90+i*95} 260 {140+i*95} 260 {165+i*95} 292Z" fill="#7cc04b"/>'
    if filling != "kokorec":
        for i in range(5):
            s += f'<rect x="{110+i*125}" y="282" width="80" height="16" rx="8" fill="#e04a3a"/>'
    s += '<g transform="translate(0 -62)">'
    s += '<path d="M60 285 C60 200 130 175 400 175 C670 175 740 200 740 285 C740 300 700 300 400 300 C100 300 60 300 60 285Z" fill="url(#bread)"/>'
    for i in range(5):
        s += f'<path d="M{150+i*125} 200 C{180+i*125} 185 {210+i*125} 185 {240+i*125} 205" stroke="#8a5221" stroke-width="7" fill="none" stroke-linecap="round"/>'
    s += '<ellipse cx="330" cy="200" rx="160" ry="14" fill="#fff" opacity=".18"/></g>'
    return svg(s, table())


def durum(filling):
    s = shadow(400, 440, 300, 50, .35)
    s += '<g transform="rotate(-18 400 320)">'
    s += '<rect x="130" y="230" width="520" height="170" rx="85" fill="#efd9a8"/>'
    for i in range(6):
        s += f'<ellipse cx="{220+i*80}" cy="{270+(i%2)*60}" rx="18" ry="9" fill="#b88a4a" opacity=".55"/>'
    s += '<path d="M300 230 L650 230 Q735 315 650 400 L300 400Z" fill="#f7f3ea"/><path d="M300 230 L300 400" stroke="#d8d0bf" stroke-width="4"/>'
    s += '<path d="M330 260 L620 260 M330 300 L630 300 M330 340 L630 340 M330 375 L610 375" stroke="#c0392b" stroke-width="3" opacity=".45"/>'
    s += '<ellipse cx="140" cy="315" rx="48" ry="85" fill="#e3c58c"/><ellipse cx="140" cy="315" rx="38" ry="72" fill="#7cc04b"/>'
    if filling == "tavuk":
        s += '<rect x="112" y="268" width="45" height="22" rx="8" fill="#e3b77d"/><rect x="118" y="300" width="40" height="22" rx="8" fill="#d9a865"/><rect x="110" y="332" width="46" height="22" rx="8" fill="#e3b77d"/>'
    else:
        s += '<ellipse cx="135" cy="285" rx="22" ry="16" fill="#6b3a1f"/><ellipse cx="145" cy="318" rx="22" ry="16" fill="#7d4524"/><ellipse cx="132" cy="350" rx="22" ry="16" fill="#6b3a1f"/>'
    s += '<circle cx="150" cy="300" r="7" fill="#e04a3a"/><circle cx="125" cy="335" r="7" fill="#e04a3a"/></g>'
    s += plate_side()
    return svg(s, table())


def plate_side():
    return ('<g transform="translate(650 480)"><ellipse rx="70" ry="30" fill="#2f5d7c"/><ellipse rx="62" ry="24" fill="#f4f1ec"/>'
            '<path d="M-40 0 l20 -8 l20 8 l20 -6 l20 6" stroke="#7a3" stroke-width="6" fill="none"/>'
            '<circle cx="-30" cy="6" r="6" fill="#e04a3a"/><circle cx="25" cy="8" r="6" fill="#e04a3a"/></g>')


def taco():
    s = plate(400, 320, 320, 220)
    for dx, rot in [(-110, -8), (120, 10)]:
        s += f'<g transform="translate({400+dx} 320) rotate({rot})">'
        s += '<rect x="-120" y="-90" width="240" height="180" rx="22" fill="#e8b36a"/>'
        for k in range(-3, 4):
            s += f'<path d="M{k*34-20} -88 L{k*34+20} 88" stroke="#a8692e" stroke-width="9" opacity=".7"/>'
        s += '</g>'
    # cut face showing filling
    s += '<g transform="translate(400 320)"><rect x="-26" y="-92" width="52" height="184" rx="10" fill="#f0c784"/>'
    s += '<rect x="-20" y="-80" width="40" height="40" rx="6" fill="#f4c542"/><rect x="-20" y="-35" width="40" height="34" rx="8" fill="#d48a35"/><rect x="-20" y="4" width="40" height="30" rx="6" fill="#ffd34d"/><rect x="-20" y="38" width="40" height="40" rx="6" fill="#f4c542"/>'
    s += '<path d="M-20 0 C-10 10 10 -6 20 4" stroke="#fff3b0" stroke-width="6" fill="none"/></g>'
    s += '<g transform="translate(640 470)"><ellipse rx="45" ry="20" fill="#ddd"/><ellipse rx="38" ry="15" fill="#c0392b"/></g>'
    return svg(s, table())


# ---------- drinks ----------

def can(color, band, label, accent="#fff"):
    s = shadow(400, 520, 140, 22, .35)
    s += f'<rect x="300" y="120" width="200" height="400" rx="26" fill="{color}"/>'
    s += f'<rect x="300" y="250" width="200" height="120" fill="{band}"/>'
    s += f'<text x="400" y="330" text-anchor="middle" font-family="Arial Black, Arial, sans-serif" font-size="46" font-weight="900" fill="{accent}">{label}</text>'
    s += '<rect x="300" y="120" width="200" height="400" rx="26" fill="url(#can)"/>'
    s += '<path d="M312 120 Q400 92 488 120 L488 130 L312 130Z" fill="#cfd3d6"/><ellipse cx="400" cy="112" rx="88" ry="16" fill="#e4e7ea"/><ellipse cx="420" cy="110" rx="26" ry="7" fill="#b8bec3"/>'
    s += '<rect x="312" y="500" width="176" height="18" rx="9" fill="#b8bec3"/>'
    for x, y in [(240, 470), (565, 455), (590, 500)]:
        s += f'<circle cx="{x}" cy="{y}" r="5" fill="#fff" opacity=".6"/>'
    return s


def glass(liquid, fill=.78, extras="", x=400, w=200, top=110, bot=520, foam=None, ice=0, bubbles=0, seed=1):
    s = shadow(x, bot + 4, w * .7, 20, .3)
    t = (bot - top)
    ly = bot - t * fill
    s += f'<path d="M{x-w/2} {top} L{x+w/2} {top} L{x+w/2-18} {bot} L{x-w/2+18} {bot}Z" fill="#fff" fill-opacity=".25" stroke="#fff" stroke-opacity=".7" stroke-width="4"/>'
    s += f'<path d="M{x-w/2+4+ (ly-top)/t*14:.0f} {ly:.0f} L{x+w/2-4-(ly-top)/t*14:.0f} {ly:.0f} L{x+w/2-21} {bot-4} L{x-w/2+21} {bot-4}Z" fill="{liquid}"/>'
    if foam:
        s += f'<rect x="{x-w/2+8}" y="{ly-22:.0f}" width="{w-16}" height="26" rx="10" fill="{foam}"/>'
    r = random.Random(seed)
    for i in range(ice):
        ix = x - 60 + i * 45 + r.uniform(-8, 8); iy = ly + 10 + r.uniform(0, 40)
        s += f'<rect x="{ix:.0f}" y="{iy:.0f}" width="44" height="40" rx="7" fill="#fff" opacity=".45" transform="rotate({r.uniform(-20,20):.0f} {ix+22:.0f} {iy+20:.0f})"/>'
    for i in range(bubbles):
        s += f'<circle cx="{x + r.uniform(-w/2+30, w/2-30):.0f}" cy="{r.uniform(ly+10, bot-15):.0f}" r="{r.uniform(2,6):.1f}" fill="#fff" opacity=".7"/>'
    s += extras
    s += f'<path d="M{x-w/2} {top} L{x+w/2} {top} L{x+w/2-18} {bot} L{x-w/2+18} {bot}Z" fill="url(#glass)"/>'
    s += f'<ellipse cx="{x}" cy="{top}" rx="{w/2}" ry="10" fill="none" stroke="#fff" stroke-opacity=".8" stroke-width="3"/>'
    return s


def lemon_slice(x, y, r=46):
    s = f'<circle cx="{x}" cy="{y}" r="{r}" fill="#f2d33a"/><circle cx="{x}" cy="{y}" r="{r-7}" fill="#fbeb8a"/>'
    for k in range(8):
        a = k * math.pi / 4
        s += f'<path d="M{x} {y} L{x+math.cos(a)*(r-10):.0f} {y+math.sin(a)*(r-10):.0f}" stroke="#f2d33a" stroke-width="3"/>'
    return s


def mint(x, y):
    return leaf(x, y, -30, .7, "#3c9a4a", "#2a7036") + leaf(x + 25, y - 10, 20, .6, "#46ad55", "#2a7036")


def bottle(body, cap, label, label_col, liquid_op=.35, h=420):
    s = shadow(400, 528, 120, 20, .3)
    s += f'<path d="M340 525 L340 260 C340 220 370 200 380 170 L380 120 L420 120 L420 170 C430 200 460 220 460 260 L460 525Z" fill="{body}" fill-opacity="{liquid_op}" stroke="{body}" stroke-width="4"/>'
    s += f'<rect x="374" y="96" width="52" height="30" rx="5" fill="{cap}"/>'
    s += f'<rect x="340" y="330" width="120" height="110" fill="{label_col}"/>'
    s += f'<text x="400" y="395" text-anchor="middle" font-family="Arial Black, Arial, sans-serif" font-size="26" font-weight="900" fill="#fff">{label}</text>'
    s += '<path d="M352 260 L352 510" stroke="#fff" stroke-opacity=".6" stroke-width="8" stroke-linecap="round"/>'
    return s


def drinks():
    D = {}
    D["su"] = bottle("#7fc4e8", "#2b7bb9", "SU", "#2b7bb9", .3)
    D["soda"] = (bottle("#9fd3a8", "#2e7d4f", "SODA", "#2e7d4f", .35))
    D["soda-limon"] = glass("#e9f6d0", .8, lemon_slice(500, 140, 50), bubbles=40, ice=3, seed=3)
    D["limonata"] = glass("#f7e46b", .85, lemon_slice(300, 130, 52) + mint(420, 130), ice=3, seed=4)
    D["7up"] = can("#1f9a4b", "#ffffff", "LİMON", "#1f9a4b")
    D["gazoz"] = bottle("#cfe8d5", "#c0392b", "GAZOZ", "#c0392b", .45)
    D["kola"] = can("#1d4fa3", "#d62f2f", "KOLA")
    D["yedigun"] = can("#f08a1c", "#ffffff", "PORTAKAL", "#f08a1c").replace('font-size="46"', 'font-size="34"')
    D["ice-tea"] = glass("#b4602a", .8, lemon_slice(470, 135, 42) + mint(330, 120), ice=4, seed=6)
    D["meyve-suyu"] = glass("#f59a23", .85, '<g transform="translate(560 470)"><circle r="60" fill="#f08a1c"/><circle cx="-18" cy="-20" r="10" fill="#fff" opacity=".3"/><path d="M0 -60 l12 -18" stroke="#4b7a2a" stroke-width="6"/></g>', seed=7)
    D["ayran"] = glass("#fbfbf7", .82, foam="#ffffff", seed=8)
    D["ayran-kutu"] = can("#f4f6f8", "#2a6fb5", "AYRAN").replace('rx="26" fill="#f4f6f8"', 'rx="26" fill="#f4f6f8"')
    return D


def hot():
    D = {}
    def mug(liquid, crema):
        s = shadow(400, 500, 210, 30, .3)
        s += '<ellipse cx="400" cy="495" rx="210" ry="40" fill="#f4f1ec"/><ellipse cx="400" cy="490" rx="170" ry="28" fill="#e2ddd4"/>'
        s += '<path d="M270 230 L530 230 L515 470 C510 490 290 490 285 470Z" fill="#fbfaf8"/>'
        s += '<path d="M525 280 C600 280 600 400 515 400" stroke="#fbfaf8" stroke-width="26" fill="none"/>'
        s += f'<ellipse cx="400" cy="232" rx="130" ry="26" fill="{liquid}"/><ellipse cx="400" cy="234" rx="110" ry="18" fill="{crema}"/>'
        s += '<path d="M285 260 L300 460" stroke="#000" stroke-opacity=".05" stroke-width="20"/>'
        s += steam(400, 180)
        return s
    D["americano"] = mug("#3b2214", "#5c3720")
    s = shadow(400, 470, 160, 24, .3)
    s += '<ellipse cx="400" cy="465" rx="150" ry="30" fill="#f4f1ec"/><ellipse cx="400" cy="460" rx="115" ry="20" fill="#e2ddd4"/>'
    s += '<path d="M320 330 L480 330 L470 440 C465 458 335 458 330 440Z" fill="#fbfaf8"/><path d="M478 350 C520 350 520 410 472 410" stroke="#fbfaf8" stroke-width="16" fill="none"/>'
    s += '<ellipse cx="400" cy="332" rx="80" ry="16" fill="#3b2214"/><ellipse cx="400" cy="333" rx="66" ry="11" fill="#b5773f"/>' + steam(400, 290)
    D["espresso"] = s
    s = shadow(380, 470, 180, 24, .3)
    s += '<ellipse cx="380" cy="465" rx="175" ry="34" fill="#c7a24a"/><ellipse cx="380" cy="460" rx="160" ry="27" fill="#f6f1e6"/>'
    s += '<path d="M310 320 L450 320 L440 440 C436 458 324 458 320 440Z" fill="#fbfaf8"/>'
    s += '<path d="M310 320 L450 320 L448 345 L312 345Z" fill="#c7a24a"/><path d="M320 380 l20 -14 l20 14 l20 -14 l20 14 l20 -14 l20 14" stroke="#2f5d7c" stroke-width="5" fill="none"/>'
    s += '<path d="M448 340 C485 340 485 395 442 395" stroke="#fbfaf8" stroke-width="14" fill="none"/>'
    s += '<ellipse cx="380" cy="322" rx="70" ry="14" fill="#8a5a33"/><ellipse cx="380" cy="323" rx="60" ry="10" fill="#b98a5c"/>'
    s += '<g transform="translate(560 440)"><path d="M-30 -10 l30 -15 l30 15 l-30 15z" fill="#f6c4d4"/><path d="M-30 -10 l30 15 l0 26 l-30 -15z" fill="#e89ab3"/><path d="M30 -10 l-30 15 l0 26 l30 -15z" fill="#d97f9c"/></g>'
    s += '<g transform="translate(215 455)"><rect x="-26" y="-32" width="52" height="56" rx="6" fill="#fff" fill-opacity=".35" stroke="#fff" stroke-opacity=".8" stroke-width="3"/><rect x="-22" y="-14" width="44" height="34" rx="4" fill="#cfe9f7" opacity=".7"/></g>'
    D["turk-kahvesi"] = s
    s = shadow(400, 480, 150, 24, .3)
    s += '<ellipse cx="400" cy="478" rx="140" ry="30" fill="#fbfaf8"/><ellipse cx="400" cy="474" rx="120" ry="22" fill="#e2ddd4"/><path d="M280 474 a120 22 0 0 0 240 0" stroke="#c0392b" stroke-width="4" fill="none"/>'
    # tulip tea glass
    s += '<path d="M335 170 C330 240 370 280 370 320 C370 360 330 400 340 470 L460 470 C470 400 430 360 430 320 C430 280 470 240 465 170Z" fill="#fff" fill-opacity=".25" stroke="#fff" stroke-opacity=".8" stroke-width="4"/>'
    s += '<path d="M338 205 C338 250 372 285 372 320 C372 360 334 400 344 466 L456 466 C466 400 428 360 428 320 C428 285 462 250 462 205Z" fill="#b8361f"/>'
    s += '<path d="M338 205 C338 250 372 285 372 320 L428 320 C428 285 462 250 462 205Z" fill="#d1502a" opacity=".6"/>'
    s += '<path d="M350 180 C352 240 380 280 380 320" stroke="#fff" stroke-opacity=".55" stroke-width="7" fill="none"/>'
    s += '<path d="M470 160 L520 470" stroke="#c9ccd0" stroke-width="7" stroke-linecap="round"/><ellipse cx="522" cy="470" rx="10" ry="5" fill="#c9ccd0"/>'
    s += '<g transform="translate(600 450)"><path d="M-20 -8 l20 -10 l20 10 l-20 10z" fill="#fff"/><path d="M-20 -8 l20 10 l0 18 l-20 -10z" fill="#eee"/><path d="M20 -8 l-20 10 l0 18 l20 -10z" fill="#ddd"/></g>' + steam(400, 140)
    D["cay"] = s
    return D


def steam(x, y):
    return (f'<path d="M{x-30} {y} C{x-50} {y-30} {x-10} {y-50} {x-30} {y-90}" stroke="#fff" stroke-opacity=".7" stroke-width="7" fill="none" stroke-linecap="round"/>'
            f'<path d="M{x+20} {y-5} C{x} {y-40} {x+40} {y-60} {x+20} {y-100}" stroke="#fff" stroke-opacity=".6" stroke-width="7" fill="none" stroke-linecap="round"/>')


def main():
    os.makedirs(OUT, exist_ok=True)
    files = {
        "kabak-cips": kabak_cips(), "peynir-tabagi": peynir_tabagi(), "karman-corman": karman_corman(),
        "tulum-salata": salad("tulum"), "sezar-salata": salad("sezar"), "roka-salata": salad("roka"),
        "ea-tavuk": sandwich("tavuk"), "ea-kofte": sandwich("kofte"), "ea-kokorec": sandwich("kokorec"),
        "tavuk-durum": durum("tavuk"), "kofte-durum": durum("kofte"), "french-taco": taco(),
    }
    for k, body in drinks().items():
        files[k] = svg(body, cafe("#dfe9ee", "#7f5a3e"))
    for k, body in hot().items():
        files[k] = svg(body, cafe("#e8ddd0", "#8a6142"))
    for k, v in files.items():
        with open(os.path.join(OUT, k + ".svg"), "w") as f:
            f.write(v)
    print(len(files), "drawings")


if __name__ == "__main__":
    main()
