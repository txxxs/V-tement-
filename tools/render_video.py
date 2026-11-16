#!/usr/bin/env python3
"""LYXOR – film de lancement 27 s, 9:16, 2160x3840, 30 fps.
Uniquement les photos et le logo fournis : cadrages, zooms et panoramiques sur les vraies images."""
import sys, subprocess, math
import numpy as np
from PIL import Image, ImageFilter, ImageDraw, ImageFont
from multiprocessing import Pool

W, H = 2160, 3840
FPS = 30
DUR = 27.0
NF = int(DUR * FPS)
BEAT = 0.5
GOLD = np.array([255, 226, 170], np.float32)   # doré chaud, doux
FONT = "/usr/share/fonts/opentype/inter/Inter-ExtraLight.otf"

# ------------------------------------------------------------------ sources
def load_stage(path):
    """Place la photo dans un cadre 9:16, fond = prolongement sombre flouté de la photo elle-même."""
    im = Image.open(path).convert("RGB")
    w, h = im.size
    sh = int(round(w * 16 / 9))
    bg = im.resize((w, sh), Image.BILINEAR).filter(ImageFilter.GaussianBlur(60))
    bg = Image.eval(bg, lambda v: int(v * 0.55))
    off = (sh - h) // 2
    # masque dégradé pour fondre le haut et le bas de la photo dans le fond
    m = np.ones((h, w), np.float32)
    f = 70
    ramp = np.linspace(0, 1, f) ** 1.5
    if off > 0:
        m[:f] *= ramp[:, None]
        m[-f:] *= ramp[::-1][:, None]
    mask = Image.fromarray((m * 255).astype(np.uint8))
    bg.paste(im, (0, off), mask)
    # suréchantillonnage x2 + netteté légère (aucune invention de détail)
    st = bg.resize((w * 2, sh * 2), Image.LANCZOS).filter(ImageFilter.UnsharpMask(2, 70, 2))
    return st, w * 2, h * 2, off * 2, sh * 2


class Src:
    def __init__(self, path):
        self.img, self.w, self.h, self.off, self.sh = load_stage(path)


# ------------------------------------------------------------------ prises de vue (fx, fy, largeur de fenêtre / largeur image, rotation°)
# s = source : B (dos), F (face)
# Chaque plan : (src, (fx,fy,ww,rot) début, (fx,fy,ww,rot) fin, durée en temps, balayage lumineux)
SHOTS = [
    ("F", (0.50, 0.085, 0.42, 0), (0.50, 0.095, 0.32, 0), 2, True),     # étiquette LYXOR du col
    ("F", (0.50, 0.40, 0.78, 0), (0.50, 0.50, 1.00, 0), 2, False),      # face entière, recul
    ("B", (0.36, 0.19, 0.46, 0), (0.64, 0.19, 0.46, 0), 2, True),       # BEYOND LIMITS, panoramique
    ("F", (0.67, 0.285, 0.52, 0), (0.67, 0.285, 0.34, 0), 2, True),     # écusson brodé, zoom progressif
    ("B", (0.50, 0.50, 1.00, -2), (0.50, 0.42, 0.74, 0), 2, False),     # dos entier, plongée douce
    ("B", (0.50, 0.42, 0.62, 0), (0.50, 0.42, 0.42, 0), 2, True),       # LYXOR brodé
    ("F", (0.12, 0.52, 0.42, 0), (0.12, 0.42, 0.42, 0), 2, False),      # manche gauche, couture
    ("B", (0.50, 0.30, 0.52, 0), (0.50, 0.30, 0.37, 0), 2, True),       # ornement supérieur
    ("F", (0.46, 0.13, 0.55, 0), (0.62, 0.13, 0.46, 0), 1, False),      # col côtelé
    ("B", (0.27, 0.46, 0.42, 0), (0.32, 0.52, 0.36, 0), 1, True),       # volute gauche
    ("B", (0.72, 0.64, 0.40, 0), (0.66, 0.58, 0.35, 0), 1, True),       # volute basse droite
    ("B", (0.22, 0.665, 0.42, 0), (0.78, 0.665, 0.42, 0), 1, False),    # LUXURY STREETWEAR
    ("F", (0.67, 0.30, 0.84, 3), (0.67, 0.30, 0.60, 0), 1, True),       # écusson, plan moyen
    ("B", (0.50, 0.57, 0.52, 0), (0.50, 0.52, 0.40, 0), 1, True),       # ornement inférieur
    ("F", (0.50, 0.50, 1.00, -3), (0.50, 0.52, 0.90, 0), 1, False),     # silhouette, contre-plongée
    ("B", (0.75, 0.40, 0.42, 0), (0.70, 0.35, 0.37, 0), 1, False),      # volute droite
    ("B", (0.50, 0.08, 0.58, 0), (0.50, 0.11, 0.46, 0), 1, False),      # encolure arrière
    ("F", (0.90, 0.50, 0.42, 0), (0.88, 0.52, 0.40, 0), 1, False),      # manche droite
    ("B", (0.10, 0.36, 0.42, 0), (0.14, 0.42, 0.42, 0), 1, False),      # épaule / manche dos
    ("B", (0.30, 0.78, 0.40, 0), (0.34, 0.72, 0.36, 0), 1, True),       # volute basse gauche
    ("B", (0.50, 0.20, 0.68, 0), (0.50, 0.20, 0.52, 0), 1, True),       # BEYOND LIMITS serré
    ("B", (0.50, 0.50, 1.00, 0), (0.50, 0.46, 0.88, 0), 1, True),       # dos entier final
]
# durées : 8 plans d'1 s puis 14 plans de 0,5 s = 15 s
SHOT_SECONDS = [1.0 if d == 2 else 0.5 for (_, _, _, d, _) in [(s[0], s[1], s[2], s[3], s[4]) for s in SHOTS]]
assert abs(sum(SHOT_SECONDS) - 15.0) < 1e-6, sum(SHOT_SECONDS)
SEQ_T0 = 7.0
SHOT_STARTS = np.cumsum([0] + SHOT_SECONDS[:-1]) + SEQ_T0
DIP_BEFORE = {4, 12}   # fondus noirs rapides avant ces plans (index)

# ------------------------------------------------------------------ utilitaires
def ease(u):
    u = min(max(u, 0.0), 1.0)
    return u * u * (3 - 2 * u)


def ease_io(u):
    u = min(max(u, 0.0), 1.0)
    return 0.5 - 0.5 * math.cos(math.pi * u)


def crop_view(src, a, b, u, punch=0.0):
    fx = a[0] + (b[0] - a[0]) * u
    fy = a[1] + (b[1] - a[1]) * u
    ww = a[2] + (b[2] - a[2]) * u
    rot = a[3] + (b[3] - a[3]) * u
    ww = ww / (1 + punch)
    fx = min(max(fx, ww / 2 + 0.012), 1 - ww / 2 - 0.012)   # reste dans la photo (pas de bord visible)
    cx = fx * src.w
    cy = src.off + fy * src.h * 1.0 + 0.0
    # src.w/h sont déjà x2 ; fy est relatif à la hauteur de la photo
    win_w = ww * src.w
    s = win_w / W
    th = math.radians(rot)
    c, sn = math.cos(th), math.sin(th)
    a_ = s * c
    b_ = -s * sn
    d_ = s * sn
    e_ = s * c
    c0 = cx - (a_ * W / 2 + b_ * H / 2)
    f0 = cy - (d_ * W / 2 + e_ * H / 2)
    out = src.img.transform((W, H), Image.AFFINE, (a_, b_, c0, d_, e_, f0), resample=Image.BICUBIC)
    return np.asarray(out, dtype=np.float32)


# vignette / grain (pré-calculés dans chaque process)
_VIG = None
_GRAIN = None
_SRCS = {}
_LOGO = None


def init():
    global _VIG, _GRAIN, _SRCS, _LOGO
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    r = np.sqrt(((xx - W / 2) / (W / 2)) ** 2 + ((yy - H / 2) / (H / 2)) ** 2)
    _VIG = (1.0 - 0.38 * np.clip(r - 0.35, 0, 1) ** 1.6)[..., None].astype(np.float32)
    rng = np.random.default_rng(3)
    _GRAIN = [(rng.standard_normal((H // 2, W // 2)).astype(np.float32)) for _ in range(5)]
    _SRCS = {"F": Src("assets/tshirt_face.jpg"), "B": Src("assets/tshirt_dos.jpg")}
    lg = Image.open("assets/logo_lyxor_clean.png").convert("RGB")
    _LOGO = lg
    init_fx()


def logo_layer(width_px, cy, scale=1.0, alpha=1.0, sweep=None):
    """Logo fidèle (mise à l'échelle uniforme), composé en additif sur noir. Retourne float32 HxWx3 (0..255)."""
    lg = _LOGO
    w0 = int(width_px * scale)
    h0 = int(lg.size[1] * w0 / lg.size[0])
    im = lg.resize((w0, h0), Image.LANCZOS)
    arr = np.asarray(im, np.float32)
    lum = arr.max(2, keepdims=True) / 255.0
    if sweep is not None:
        # sweep : position 0..1 d'une bande lumineuse diagonale traversant le logo (masquée par le logo lui-même)
        yy, xx = np.mgrid[0:h0, 0:w0].astype(np.float32)
        pos = -0.3 + 1.6 * sweep
        d = (xx / w0 + 0.25 * yy / h0) - pos
        band = np.exp(-(d / 0.07) ** 2)
        arr = arr * (1 + 0.55 * band[..., None]) + band[..., None] * lum * np.array([60, 52, 38], np.float32)
        arr = np.clip(arr, 0, 255)
    full = np.zeros((H, W, 3), np.float32)
    x0 = (W - w0) // 2
    y0 = int(cy - h0 / 2)
    full[y0:y0 + h0, x0:x0 + w0] = arr * alpha
    return full, (x0, y0, w0, h0)


def glow(layer, rad=40, k=0.45):
    small = Image.fromarray(np.clip(layer, 0, 255).astype(np.uint8)).resize((W // 8, H // 8), Image.BILINEAR)
    small = small.filter(ImageFilter.GaussianBlur(rad / 8))
    g = np.asarray(small.resize((W, H), Image.BILINEAR), np.float32)
    return layer + g * k


def spaced_text(txt, size, tracking_em, alpha=1.0, blur=0.0, y_center=H // 2, color=(255, 255, 255), line_gap=1.35):
    """Texte fin, espacé, parfaitement centré (l'espacement final n'entre pas dans le centrage)."""
    font = ImageFont.truetype(FONT, size)
    lines = txt.split("\n")
    pad = 80
    th = int(size * line_gap * len(lines)) + pad * 2
    canvas = Image.new("L", (W, th), 0)
    dr = ImageDraw.Draw(canvas)
    track = tracking_em * size
    for li, line in enumerate(lines):
        widths = [font.getlength(ch) for ch in line]
        total = sum(widths) + track * (len(line) - 1)
        x = (W - total) / 2
        y = pad + li * size * line_gap
        for ch, wch in zip(line, widths):
            dr.text((x, y), ch, font=font, fill=255)
            x += wch + track
    if blur > 0.1:
        canvas = canvas.filter(ImageFilter.GaussianBlur(blur))
    a = np.asarray(canvas, np.float32) / 255.0 * alpha
    full = np.zeros((H, W, 3), np.float32)
    y0 = int(y_center - th / 2)
    full[y0:y0 + th] = a[..., None] * np.array(color, np.float32)
    return full


def hairline(progress, alpha):
    """Fin trait de lumière doré horizontal qui s'ouvre depuis le centre."""
    full = np.zeros((H, W, 3), np.float32)
    half = int(progress * 560)
    if half < 2 or alpha <= 0:
        return full
    xs = np.arange(W, dtype=np.float32) - W / 2
    prof = np.exp(-(xs / max(half, 1)) ** 4)
    y0 = 1826
    for dy, k in ((-1, .5), (0, 1.0), (1, .5)):
        full[y0 + dy] = (prof[:, None] * GOLD[None, :]) * k * alpha
    return glow(full, 36, 1.6)



# ------------------------------------------------------------------ effets de lumière (inspirés de la référence)
SW, SH = W // 4, H // 4
_GEO = {}
_PART = None


def init_fx():
    global _PART
    yy, xx = np.mgrid[0:SH, 0:SW].astype(np.float32)
    px, py = xx * 4.0, yy * 4.0
    # faisceau : axe partant du haut-droite vers le bas-gauche
    x0, y0 = W * 0.98, -H * 0.04
    dx, dy = -0.55, 1.0
    n = math.hypot(dx, dy); dx, dy = dx / n, dy / n
    along = ((px - x0) * dx + (py - y0) * dy)
    dist = np.abs((px - x0) * dy - (py - y0) * dx)
    _GEO["along"] = along / H
    _GEO["dist"] = dist / W
    # rayons autour du logo final
    cx, cy = W / 2, 1817.0
    ang = np.arctan2(py - cy, px - cx)
    rad = np.hypot(px - cx, (py - cy) * 1.0) / W
    _GEO["ang"], _GEO["rad"] = ang, rad
    rng = np.random.default_rng(11)
    N = 1300
    _PART = dict(x=rng.random(N), y=rng.random(N), vx=rng.normal(0, 0.004, N), vy=rng.uniform(0.004, 0.022, N),
                 ph=rng.uniform(0, 6.28, N), tw=rng.uniform(1.5, 4.5, N), sz=rng.uniform(0.6, 1.4, N))


def shaft_mask(spread, length_fall=1.15):
    al, di = _GEO["along"], _GEO["dist"]
    w = spread * (0.35 + 0.9 * np.clip(al, 0, 1.4))
    m = np.exp(-(di / np.maximum(w, 1e-3)) ** 2) * np.clip(1 - al / length_fall, 0, 1) ** 1.2 * (al > -0.05)
    return m.astype(np.float32)


def up(mask_small):
    im = Image.fromarray(mask_small.astype(np.float32), mode="F").resize((W, H), Image.BILINEAR)
    return np.asarray(im, np.float32)


def dust_layer(t, mask_small, amp):
    """Fines particules dorées qui flottent ; leur éclat suit le masque (faisceau) fourni."""
    P = _PART
    x = (P["x"] + P["vx"] * t + 0.01 * np.sin(t * 0.8 + P["ph"])) % 1.0
    y = (P["y"] + P["vy"] * t) % 1.0
    mx = np.clip((x * SW).astype(int), 0, SW - 1)
    my = np.clip((y * SH).astype(int), 0, SH - 1)
    b = mask_small[my, mx] * (0.45 + 0.55 * np.sin(t * P["tw"] + P["ph"]) ** 2) * P["sz"] * amp
    hw, hh = W // 2, H // 2
    canvas = np.zeros((hh, hw), np.float32)
    ix = np.clip((x * hw).astype(int), 0, hw - 1)
    iy = np.clip((y * hh).astype(int), 0, hh - 1)
    np.add.at(canvas, (iy, ix), b)
    from scipy.ndimage import gaussian_filter
    big = Image.fromarray(gaussian_filter(canvas, 0.9), mode="F").resize((W, H), Image.BILINEAR)
    return np.asarray(big, np.float32) * 5.0


def light_shaft(t):
    a = ease_io((t - 0.5) / 2.0) * (1 - ease_io((t - 3.4) / 0.8))
    if a <= 0.001:
        return np.zeros((H, W, 3), np.float32)
    spread = 0.04 + 0.05 * ease_io((t - 0.5) / 2.5)
    m = shaft_mask(spread)
    beam = up(m * a * 0.60)
    dust = dust_layer(t, m, a * 255.0 * 0.9)
    return (beam[..., None] * GOLD[None, None, :] * 0.78) + dust[..., None] * GOLD[None, None, :] / 255.0 * 1.0


def ray_layer(tt_):
    a = ease_io((tt_ - 0.5) / 1.8)
    if a <= 0.001:
        return np.zeros((H, W, 3), np.float32)
    ang, rad = _GEO["ang"], _GEO["rad"]
    ph = tt_ * 0.35
    r1 = (0.5 + 0.5 * np.sin(ang * 9 + ph)) ** 5
    r2 = (0.5 + 0.5 * np.sin(ang * 17 - ph * 1.3 + 1.7)) ** 6
    r3 = (0.5 + 0.5 * np.sin(ang * 31 + ph * 0.6 + 0.4)) ** 8
    rays = (0.55 * r1 + 0.35 * r2 + 0.30 * r3)
    fall = np.exp(-rad * 3.4) * (1 - np.exp(-rad * 40))
    m = (rays * fall).astype(np.float32) * a * 0.55
    return up(m)[..., None] * GOLD[None, None, :] * 0.8


# ------------------------------------------------------------------ rendu d'une image
def render_frame(fi):
    t = fi / FPS
    frame = np.zeros((H, W, 3), np.float32)
    master = 1.0

    if t < 4.3:
        # ---- Intro : noir, lumière fine, logo
        hl_p = ease_io((t - 0.8) / 1.4)
        hl_a = ease_io((t - 0.8) / 0.9) * (1 - ease_io((t - 2.0) / 0.9))
        frame += light_shaft(t)
        frame += hairline(hl_p, hl_a)
        la = ease_io((t - 1.7) / 1.3)
        lo = 1 - ease_io((t - 3.55) / 0.7)
        a = la * lo
        if a > 0.001:
            sc = 0.94 + 0.08 * ease_io((t - 1.7) / 2.4)
            sw = (t - 2.2) / 1.2 if 2.2 <= t <= 3.4 else None
            lay, _ = logo_layer(1560, 1826, sc, a, sw)
            frame += glow(lay, 44, 0.35)
    elif t < 7.0:
        # ---- Annonce
        ta = (t - 4.35) / 1.1
        a_in = ease_io(ta)
        a_out = 1 - ease_io((t - 6.15) / 0.8)
        a = a_in * a_out
        if a > 0.001:
            track = 0.62 - 0.30 * ease_io(ta) + 0.05 * ease_io((t - 6.0) / 1.0)
            blur = 14 * (1 - ease_in_out_cl(ta)) + 3 * ease_in_out_cl((t - 6.15) / 0.8)
            txt = spaced_text("UNE NOUVELLE\nMARQUE ARRIVE.", 104, track, a, blur, y_center=H // 2 - 20)
            frame += txt
    elif t < 22.0:
        # ---- Séquence produits
        k = int(np.searchsorted(SHOT_STARTS, t, side="right") - 1)
        k = max(0, min(k, len(SHOTS) - 1))
        sname, a0, b0, dd, sweep_on = SHOTS[k]
        dur = SHOT_SECONDS[k]
        tl = t - SHOT_STARTS[k]
        u = ease_io(tl / dur) * 0.55 + (tl / dur) * 0.45   # mouvement continu, départ/arrivée doux
        punch = 0.07 * math.exp(-tl * 22.0)                 # petit « coup » de zoom à chaque coupe
        img = crop_view(_SRCS[sname], a0, b0, u, punch)
        # étalonnage : noirs profonds, contraste doux, léger voile chaud sur les hautes lumières
        img = (img / 255.0)
        img = np.clip((img - 0.015) * 1.06, 0, 1) ** 1.08
        img = img * np.array([1.02, 1.0, 0.96], np.float32)
        if sweep_on:
            # balayage de lumière de studio, en diagonale, sur l'ensemble du plan
            pos = -0.4 + 1.8 * (tl / dur)
            yy = np.linspace(0, 1, H, dtype=np.float32)[:, None]
            xx = np.linspace(0, 1, W, dtype=np.float32)[None, :]
            d = (xx + 0.35 * yy) - pos
            band = np.exp(-(d / 0.16) ** 2)[..., None] * 0.20
            lum = img.max(2, keepdims=True)
            img = img + band * (0.25 + 1.4 * lum) * (GOLD / 255.0)[None, None, :] * 0.8
        frame = np.clip(img, 0, 1) * 255.0 * _VIG
        dl = dust_layer(t, np.ones((SH, SW), np.float32), 120.0)
        frame = frame + dl[..., None] * (GOLD / 255.0)[None, None, :] * 0.9
        # flash lumineux doré sur la coupe
        if tl < 0.25:
            fl = 0.24 * math.exp(-tl * 24.0)
            if k == 0:
                fl *= 0.5
            frame = frame + fl * (frame.max(2, keepdims=True) * 0.6 + 70.0) * (GOLD / 255.0)[None, None, :]
        # fondus noirs
        if k in DIP_BEFORE:
            # entrée en fondu depuis le noir (4 premières images)
            master *= min(1.0, 0.15 + tl / 0.16)
        if (k + 1) in DIP_BEFORE:
            rem = dur - tl
            master *= min(1.0, 0.15 + rem / 0.16)
        if k == len(SHOTS) - 1:
            # montée en intensité sur le dernier plan
            frame = frame * (1 + 0.15 * (tl / dur))
    else:
        # ---- Final : coupe sèche sur noir
        tt_ = t - 22.0
        group_cy = H // 2
        logo_w = 1500
        lh = int(_LOGO.size[1] * logo_w / _LOGO.size[0])
        gap = 150
        txt_h = 56
        total = lh + gap + txt_h
        top = group_cy - total / 2
        logo_cy = top + lh / 2
        txt_cy = top + lh + gap + txt_h / 2
        frame += ray_layer(tt_)
        la = ease_io((tt_ - 0.4) / 1.2)
        if la > 0.001:
            sw = (tt_ - 2.9) / 1.4 if 2.9 <= tt_ <= 4.3 else None
            glow_pulse = 1.0 + 0.35 * math.sin(max(0, (tt_ - 2.9) / 1.4) * math.pi) if 2.9 <= tt_ <= 4.3 else 1.0
            sc = 0.96 + 0.05 * ease_io((tt_ - 0.4) / 4.6)
            lay, _ = logo_layer(logo_w, logo_cy, sc, la, sw)
            frame += glow(lay, 46, 0.30 * glow_pulse)
        ta = (tt_ - 1.6) / 1.1
        if ta > 0:
            a = ease_io(ta)
            track = 0.62 - 0.17 * ease_io(ta)
            blur = 10 * (1 - ease_io(ta))
            frame += spaced_text("OUVERTURE PROCHAINEMENT", 56, track, a, blur, y_center=int(txt_cy), color=(255, 255, 255))
        master *= 1 - ease_io((tt_ - 3.7) / 1.3)

    # grain fin de pellicule (très léger) + masterisation
    g = _GRAIN[fi % 5]
    ox, oy = (fi * 37) % (W // 2), (fi * 91) % (H // 2)
    g = np.roll(np.roll(g, ox, 1), oy, 0)
    g = np.asarray(Image.fromarray(g).resize((W, H), Image.BILINEAR), np.float32)
    lumn = np.clip(frame.max(2, keepdims=True) / 255.0, 0, 1)
    frame = frame + g[..., None] * (1.6 + 2.2 * lumn)
    frame = np.clip(frame * master, 0, 255).astype(np.uint8)
    return frame.tobytes()


def ease_in_out_cl(u):
    return ease_io(u)


def main():
    start = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    end = int(sys.argv[2]) if len(sys.argv) > 2 else NF
    out = sys.argv[3] if len(sys.argv) > 3 else "build/lyxor_video_noaudio.mp4"
    cmd = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
           "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-preset", "medium", "-crf", "19",
           "-pix_fmt", "yuv420p", "-profile:v", "high", "-level", "5.1", "-colorspace", "bt709",
           "-color_primaries", "bt709", "-color_trc", "bt709", "-movflags", "+faststart", out]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    with Pool(4, initializer=init) as pool:
        for i, b in enumerate(pool.imap(render_frame, range(start, end), chunksize=2)):
            p.stdin.write(b)
            if i % 30 == 0:
                print(f"{start + i}/{NF}", flush=True)
    p.stdin.close()
    p.wait()


if __name__ == "__main__":
    main()
