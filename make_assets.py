"""Generate the project's artwork (hero banner, pipeline illustration, logo).  Usage: python make_assets.py"""
import math
import random

from PIL import Image, ImageDraw, ImageFilter, ImageFont

from src.config import ASSETS

S = 2  # supersampling factor for smooth edges
NAVY, DEEP, TEAL, AQUA, SKY, AMBER, CORAL, MINT = (11, 34, 51), (14, 59, 74), (15, 181, 166), (111, 227, 208), (91, 168, 255), (244, 183, 63), (255, 92, 108), (45, 212, 160)
ORANGE, YELLOW = (255, 122, 61), (245, 200, 76)


def font(size, bold=False):
    for p in ("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
              "arial.ttf"):
        try:
            return ImageFont.truetype(p, size * S)
        except Exception:
            continue
    return ImageFont.load_default()


def gradient(w, h, c1, c2, diagonal=True):
    img = Image.new("RGB", (w, h))
    px = img.load()
    for y in range(h):
        for x in range(w):
            t = (x / w * 0.7 + y / h * 0.3) if diagonal else y / h
            px[x, y] = tuple(int(c1[i] + (c2[i] - c1[i]) * t) for i in range(3))
    return img


def glow(base, xy, r, color, alpha=120):
    layer = Image.new("RGBA", base.size, (0, 0, 0, 0))
    ImageDraw.Draw(layer).ellipse([xy[0] - r, xy[1] - r, xy[0] + r, xy[1] + r], fill=color + (alpha,))
    base.alpha_composite(layer.filter(ImageFilter.GaussianBlur(r / 2)))


def hero(w=1800, h=420):
    W, H = w * S, h * S
    img = gradient(W // 4, H // 4, NAVY, (13, 110, 120)).resize((W, H), Image.BICUBIC).convert("RGBA")
    random.seed(7)
    # network mesh (top-left): nodes joined to near neighbours
    pts = [(random.randint(0, int(W * 0.55)), random.randint(0, H)) for _ in range(46)]
    d = ImageDraw.Draw(img, "RGBA")
    for i, a in enumerate(pts):
        for b in pts[i + 1:]:
            if math.dist(a, b) < 330 * S / 2:
                d.line([a, b], fill=AQUA + (38,), width=S)
    for x, y in pts:
        d.ellipse([x - 4 * S, y - 4 * S, x + 4 * S, y + 4 * S], fill=AQUA + (150,))
    # 6G tower + concentric signal arcs
    tx, ty = int(W * 0.70), int(H * 0.30)
    for k in range(1, 7):
        r = k * 62 * S
        d.arc([tx - r, ty - r, tx + r, ty + r], 200, 340, fill=AQUA + (max(30, 200 - k * 28),), width=3 * S)
    glow(img, (tx, ty), 60 * S, TEAL, 150)
    d = ImageDraw.Draw(img, "RGBA")
    d.polygon([(tx, ty), (tx - 22 * S, H), (tx + 22 * S, H)], fill=(8, 24, 36, 255))
    d.ellipse([tx - 9 * S, ty - 9 * S, tx + 9 * S, ty + 9 * S], fill=AQUA + (255,))
    # factory skyline
    base = H
    x = int(W * 0.50)
    for bw, bh, saw in [(190, 150, 4), (150, 210, 3), (230, 130, 5), (170, 180, 3), (120, 110, 2)]:
        bw, bh = bw * S, bh * S
        d.rectangle([x, base - bh, x + bw, base], fill=(8, 24, 36, 255))
        for s in range(saw):                                          # saw-tooth roof
            sx = x + s * bw // saw
            d.polygon([(sx, base - bh), (sx + bw // saw, base - bh), (sx, base - bh - 38 * S)], fill=(8, 24, 36, 255))
        for wx in range(x + 14 * S, x + bw - 18 * S, 30 * S):         # lit windows
            for wy in range(base - bh + 24 * S, base - 20 * S, 34 * S):
                if random.random() < 0.55:
                    d.rectangle([wx, wy, wx + 14 * S, wy + 14 * S], fill=random.choice([AMBER, AMBER, AQUA, MINT]) + (210,))
        x += bw + 12 * S
    for cx, chh in [(int(W * 0.545), 330), (int(W * 0.575), 280)]:     # chimneys
        d.rectangle([cx, base - chh * S, cx + 26 * S, base - 150 * S], fill=(8, 24, 36, 255))
        glow(img, (cx + 13 * S, base - chh * S), 22 * S, AMBER, 90)
    # rising efficiency bars (right edge)
    d = ImageDraw.Draw(img, "RGBA")
    for i, (hh, col) in enumerate([(70, CORAL), (110, AMBER), (160, MINT), (210, MINT)]):
        bx = int(W * 0.90) + i * 38 * S
        d.rounded_rectangle([bx, base - hh * S - 30 * S, bx + 24 * S, base - 30 * S], radius=6 * S, fill=col + (230,))
    img = img.convert("RGB").resize((w, h), Image.LANCZOS)
    img.save(ASSETS / "hero.jpg", quality=88)


def icon(d, kind, cx, cy, col):
    r = 46 * S
    if kind == "machine":
        d.rounded_rectangle([cx - r, cy - r * 0.55, cx + r, cy + r * 0.7], radius=10 * S, fill=col)
        d.rectangle([cx - r * 0.6, cy - r * 0.95, cx - r * 0.2, cy - r * 0.55], fill=col)
        for i in range(3):
            d.ellipse([cx - r * 0.7 + i * r * 0.55, cy - r * 0.2, cx - r * 0.35 + i * r * 0.55, cy + r * 0.15], fill=(20, 22, 29))
    elif kind == "sensor":
        d.ellipse([cx - 14 * S, cy - 14 * S, cx + 14 * S, cy + 14 * S], fill=col)
        for k in (28, 46, 64):
            d.arc([cx - k * S, cy - k * S, cx + k * S, cy + k * S], 215, 325, fill=col, width=5 * S)
    elif kind == "tower":
        d.polygon([(cx, cy - r), (cx - r * 0.45, cy + r), (cx + r * 0.45, cy + r)], fill=col)
        d.line([(cx - r * 0.3, cy + r * 0.3), (cx + r * 0.3, cy - r * 0.1)], fill=(255, 255, 255), width=4 * S)
        d.line([(cx + r * 0.3, cy + r * 0.3), (cx - r * 0.3, cy - r * 0.1)], fill=(255, 255, 255), width=4 * S)
        for k in (20, 34):
            d.arc([cx - k * S, cy - r * 0.9 - k * S, cx + k * S, cy - r * 0.9 + k * S], 205, 335, fill=col, width=4 * S)
    elif kind == "brain":
        for dx, dy in [(-22, -10), (22, -10), (0, -26), (-14, 14), (14, 14)]:
            d.ellipse([cx + (dx - 22) * S, cy + (dy - 22) * S, cx + (dx + 22) * S, cy + (dy + 22) * S], fill=col)
        for dx, dy in [(-22, -10), (22, -10), (0, 4)]:
            d.ellipse([cx + (dx - 5) * S, cy + (dy - 5) * S, cx + (dx + 5) * S, cy + (dy + 5) * S], fill=(255, 255, 255))
    elif kind == "dash":
        d.rounded_rectangle([cx - r, cy - r * 0.7, cx + r, cy + r * 0.7], radius=10 * S, fill=col)
        for i, (hh, c) in enumerate([(0.35, CORAL), (0.6, AMBER), (0.95, SKY)]):
            bx = cx - r * 0.6 + i * r * 0.45
            d.rectangle([bx, cy + r * 0.5 - r * hh, bx + r * 0.3, cy + r * 0.5], fill=c)


def pipeline(w=1500, h=300):
    W, H = w * S, h * S
    img = Image.new("RGB", (W, H), (20, 22, 29)).convert("RGBA")
    d = ImageDraw.Draw(img, "RGBA")
    steps = [("machine", "Machines", "50 industrial units", ORANGE), ("sensor", "Sensors", "heat · vibration · power", YELLOW),
             ("tower", "6G network", "latency · packet loss", SKY), ("brain", "AI classifier", "High · Medium · Low", (165, 139, 255)),
             ("dash", "Live dashboard", "act before losses grow", MINT)]
    xs = [int(W * (0.1 + 0.2 * i)) for i in range(5)]
    cy = int(H * 0.40)
    for i in range(4):
        d.line([(xs[i] + 80 * S, cy), (xs[i + 1] - 80 * S, cy)], fill=(*TEAL, 120), width=3 * S)
        ax = xs[i + 1] - 80 * S
        d.polygon([(ax, cy), (ax - 14 * S, cy - 8 * S), (ax - 14 * S, cy + 8 * S)], fill=(*TEAL, 200))
    for (kind, title, sub, col), x in zip(steps, xs):
        pale = tuple(int(bg + (c - bg) * 0.22) for c, bg in zip(col, (20, 22, 29)))
        d.ellipse([x - 70 * S, cy - 70 * S, x + 70 * S, cy + 70 * S], fill=pale)
        icon(d, kind, x, cy, col)
        for txt, f, y, c in [(title, font(24, True), cy + 92 * S, (236, 238, 243)), (sub, font(17), cy + 130 * S, (139, 145, 161))]:
            tw = d.textlength(txt, font=f)
            d.text((x - tw / 2, y), txt, font=f, fill=c)
    img.convert("RGB").resize((w, h), Image.LANCZOS).save(ASSETS / "pipeline.png")


def logo(size=160):
    W = size * S
    img = Image.new("RGBA", (W, W), (0, 0, 0, 0))
    g = gradient(W // 4, W // 4, (255, 138, 76), (232, 83, 31)).resize((W, W), Image.BICUBIC).convert("RGBA")
    mask = Image.new("L", (W, W), 0)
    ImageDraw.Draw(mask).rounded_rectangle([0, 0, W, W], radius=W // 4, fill=255)
    img.paste(g, (0, 0), mask)
    d = ImageDraw.Draw(img)
    pts = [(0.18, 0.62), (0.34, 0.62), (0.44, 0.28), (0.58, 0.78), (0.68, 0.45), (0.82, 0.45)]
    d.line([(x * W, y * W) for x, y in pts], fill=(255, 255, 255), width=int(0.07 * W), joint="curve")
    for x, y in (pts[0], pts[-1]):
        d.ellipse([x * W - 0.05 * W, y * W - 0.05 * W, x * W + 0.05 * W, y * W + 0.05 * W], fill=(255, 255, 255))
    img.resize((size, size), Image.LANCZOS).save(ASSETS / "logo.png")


if __name__ == "__main__":
    ASSETS.mkdir(exist_ok=True)
    pipeline(); logo()
    print("assets written to", ASSETS)
