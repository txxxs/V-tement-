#!/usr/bin/env python3
"""Bande-son LYXOR (27 s, 120 BPM) synthétisée : drone sombre, impacts, électro de mode.
Les coupes du montage tombent sur les temps (0,5 s) de la section 7 s -> 22 s."""
import numpy as np
from scipy import signal
from scipy.io import wavfile

SR = 48000
DUR = 27.0
N = int(SR * DUR)
rng = np.random.default_rng(7)
BEAT = 0.5


def tt(d):
    return np.arange(int(SR * d)) / SR


def place(buf, x, t0, gain=1.0, pan=0.0):
    i = int(t0 * SR)
    if i >= buf.shape[1]:
        return
    n = min(len(x), buf.shape[1] - i)
    l = gain * np.cos((pan + 1) * np.pi / 4)
    r = gain * np.sin((pan + 1) * np.pi / 4)
    buf[0, i:i + n] += x[:n] * l
    buf[1, i:i + n] += x[:n] * r


def lp(x, f, order=2):
    return signal.sosfilt(signal.butter(order, f, "low", fs=SR, output="sos"), x)


def hp(x, f, order=2):
    return signal.sosfilt(signal.butter(order, f, "high", fs=SR, output="sos"), x)


def bp(x, f1, f2, order=2):
    return signal.sosfilt(signal.butter(order, [f1, f2], "band", fs=SR, output="sos"), x)


def kick(d=0.45, f0=120, f1=42):
    t = tt(d)
    f = f1 + (f0 - f1) * np.exp(-t * 28)
    ph = 2 * np.pi * np.cumsum(f) / SR
    x = np.sin(ph) * np.exp(-t * 7.5)
    click = hp(rng.standard_normal(len(t)), 2500) * np.exp(-t * 180) * 0.15
    return np.tanh((x + click) * 1.6) * 0.95


def sub_hit(d=3.0, f=38):
    t = tt(d)
    x = np.sin(2 * np.pi * (f + 40 * np.exp(-t * 9)) * t) * np.exp(-t * 1.5)
    n = lp(rng.standard_normal(len(t)), 220) * np.exp(-t * 4) * 0.5
    return np.tanh((x + n) * 1.3)


def hat(d=0.09, open_=False):
    d = 0.35 if open_ else d
    t = tt(d)
    x = hp(rng.standard_normal(len(t)), 7500) * np.exp(-t * (14 if open_ else 55))
    return x * 0.5


def clap(d=0.28):
    t = tt(d)
    n = bp(rng.standard_normal(len(t)), 900, 5500)
    env = np.exp(-t * 22)
    for k in (0.0, 0.011, 0.022):
        env += 0.8 * np.exp(-np.clip(t - k, 0, None) * 120) * (t >= k)
    return n * env * 0.5


def metal(f=1180, d=2.2, ratios=(1, 2.76, 5.4, 8.93, 11.34)):
    """Frappe métallique discrète (partiels inharmoniques)."""
    t = tt(d)
    x = np.zeros_like(t)
    for i, r in enumerate(ratios):
        x += np.sin(2 * np.pi * f * r * t + rng.uniform(0, 6)) * np.exp(-t * (2.2 + i * 1.7)) / (1 + i * 0.7)
    return x * 0.35


def riser(d, f0=300, f1=9000, curve=2.0):
    t = tt(d)
    n = rng.standard_normal(len(t))
    out = np.zeros_like(n)
    blocks = 40
    edges = np.linspace(0, len(t), blocks + 1, dtype=int)
    for b in range(blocks):
        a, c = edges[b], edges[b + 1]
        u = (b + 0.5) / blocks
        fc = f0 * (f1 / f0) ** (u ** (1 / curve))
        seg = n[a:c]
        out[a:c] = bp(seg, max(fc * 0.7, 50), min(fc * 1.3, SR / 2 - 100), 2)
    env = (np.linspace(0, 1, len(t)) ** 2.2)
    return out * env


def reverse_cymbal(d):
    t = tt(d)
    x = hp(rng.standard_normal(len(t)), 3500) * (np.linspace(0, 1, len(t)) ** 3)
    return x * 0.6


def reverb(x2, decay=2.8, wet=1.0, damp=5200):
    n = int(SR * decay)
    t = np.arange(n) / SR
    out = []
    for ch in range(2):
        ir = rng.standard_normal(n) * np.exp(-t * (6.9 / decay))
        ir = lp(ir, damp, 1)
        ir[: int(0.012 * SR)] *= np.linspace(0, 1, int(0.012 * SR))
        ir /= np.sqrt(np.sum(ir ** 2)) + 1e-9
        out.append(signal.fftconvolve(x2[ch], ir)[: x2.shape[1]])
    return np.array(out) * wet


def saw(f, t):
    ph = (f * t) % 1.0
    return 2 * ph - 1


def pad(freqs, d, cutoff_env):
    t = tt(d)
    x = np.zeros_like(t)
    for f in freqs:
        for det in (-0.12, 0.0, 0.12):
            x += saw(f * (1 + det * 0.01), t) * 0.2
    # filtre balayé par blocs
    out = np.zeros_like(x)
    blocks = 60
    edges = np.linspace(0, len(t), blocks + 1, dtype=int)
    zi = None
    for b in range(blocks):
        a, c = edges[b], edges[b + 1]
        fc = cutoff_env((b + 0.5) / blocks)
        sos = signal.butter(2, fc, "low", fs=SR, output="sos")
        if zi is None:
            zi = np.zeros((sos.shape[0], 2))
        out[a:c], zi = signal.sosfilt(sos, x[a:c], zi=zi)
    return out


def main():
    dry = np.zeros((2, N), dtype=np.float64)   # éléments secs
    wetbus = np.zeros((2, N), dtype=np.float64)  # envoi réverbération
    sidechain = np.ones(N)

    A1, A2 = 55.0, 110.0
    # -------- 0 -> 4 s : obscurité, lumière fine, logo
    p = pad([A1 * 2, A1 * 3, A1 * 4.0 * 1.19], 7.0, lambda u: 160 + 500 * u)
    env = np.interp(tt(7.0), [0, 2.0, 3.6, 4.5, 7.0], [0, 0.35, 0.5, 0.45, 0.0])
    place(dry, p * env * 0.55, 0.0, 0.5, -0.1)
    drone = np.sin(2 * np.pi * A1 * tt(7.0)) * np.interp(tt(7.0), [0, 1.8, 5, 7], [0, 0.5, 0.45, 0])
    place(dry, drone, 0.0, 0.45)
    # lumière qui naît : léger souffle aigu qui monte
    r = riser(1.3, 1500, 11000, 1.6)
    place(dry, r, 0.75, 0.16, 0.2)
    place(wetbus, r, 0.75, 0.16)
    # apparition du logo : impact profond + frappe métallique
    place(dry, sub_hit(3.2, 36), 1.65, 0.75)
    m = metal(1320, 2.8)
    place(dry, m, 1.7, 0.32, -0.25)
    place(wetbus, m, 1.7, 0.7)
    place(wetbus, metal(1980, 2.0), 2.55, 0.2)   # reflet du balayage lumineux
    # -------- 4 -> 7 s : « UNE NOUVELLE MARQUE ARRIVE. »
    for k in range(6):
        place(dry, kick(0.5, 90, 36) * 0.45, 4.2 + k * 0.5 * 1.0, 0.55)
        sidechain_idx = int((4.2 + k * 0.5) * SR)
    rr = riser(2.9, 250, 10000, 1.7)
    place(dry, rr, 4.1, 0.38, 0.1)
    place(wetbus, rr, 4.1, 0.2)
    place(wetbus, metal(880, 2.5), 4.3, 0.35)
    # -------- 7 -> 22 s : séquence produits, 120 BPM
    t_start = 7.0
    nb = int(15.0 / BEAT)
    for b in range(nb):
        t0 = t_start + b * BEAT
        prog = b / nb
        g = 0.62 + 0.38 * prog
        place(dry, kick() * 1.0, t0, 0.95 * g)
        i0 = int(t0 * SR)
        dlen = int(0.30 * SR)
        shape = np.minimum(1.0, 0.25 + np.linspace(0, 1, dlen) ** 0.6 * 0.75)
        sidechain[i0:i0 + dlen] = np.minimum(sidechain[i0:i0 + dlen], shape)
        # contretemps : charleston
        place(dry, hat(), t0 + BEAT / 2, 0.28 * g, 0.3)
        if prog > 0.35:  # double croche
            place(dry, hat(), t0 + BEAT / 4, 0.15 * g, -0.3)
            place(dry, hat(), t0 + 3 * BEAT / 4, 0.15 * g, -0.3)
        if b % 2 == 1:
            c = clap()
            place(dry, c, t0, 0.55 * g)
            place(wetbus, c, t0, 0.3)
        # impact de coupe : coup métallique discret toutes les 2 coupes
        if b % 4 == 0 or prog > 0.5:
            mt = metal(2400 + 300 * (b % 3), 0.9)
            place(dry, mt, t0, 0.10 + 0.1 * prog, (-1) ** b * 0.5)
            place(wetbus, mt, t0, 0.3)
        if b % 8 == 0:
            place(dry, sub_hit(1.6, 40) * 0.7, t0, 0.5)
            place(wetbus, reverse_cymbal(0.45) * 0.5, t0 - 0.45, 0.35)
    # basse (La mineur), croches
    bass_seq = [A1, A1, A1 * 1.5, A1, A1 * 1.189, A1, A1 * 1.335, A1 * 1.5]  # A A E A C A D E
    for s in range(int(15.0 / (BEAT / 2))):
        t0 = t_start + s * BEAT / 2
        f = bass_seq[s % 8] * (1.0 if (s // 16) % 2 == 0 else 0.891)
        d = BEAT / 2 * 0.9
        t = tt(d)
        x = lp(saw(f * 2, t), 380 + 900 * (s / 60.0), 2) * np.exp(-t * 4.5) + np.sin(2 * np.pi * f * t) * 0.6
        place(dry, x * 0.38, t0, 1.0 * (0.5 + 0.5 * min(1, s / 30)))
    # arpège sombre (pluck) qui s'ouvre
    arp_notes = [220.0, 261.63, 329.63, 392.0, 329.63, 261.63, 440.0, 329.63]
    for s in range(int(15.0 / (BEAT / 2))):
        if s < 8:
            continue
        t0 = t_start + s * BEAT / 2
        f = arp_notes[s % 8] * (2 if s > 40 else 1)
        t = tt(0.4)
        x = (saw(f, t) + saw(f * 1.004, t)) * 0.5
        x = lp(x, 900 + 3500 * min(1, s / 60), 2) * np.exp(-t * 9)
        place(dry, x * 0.2, t0, 1.0, np.sin(s) * 0.6)
        place(wetbus, x * 0.2, t0, 0.5)
    # nappe qui monte
    pd = pad([A2, A2 * 1.5, A2 * 1.189, A2 * 2], 15.2, lambda u: 220 + 3600 * u ** 1.5)
    pe = np.interp(tt(15.2), [0, 2, 15.0, 15.2], [0.0, 0.5, 0.95, 0.0])
    place(dry, pd * pe * 0.25 * np.concatenate([sidechain[int(t_start * SR):int(t_start * SR) + len(pd)]]
                                              + [np.ones(max(0, len(pd) - (N - int(t_start * SR))))])[:len(pd)], t_start, 0.8)
    # montée finale : roulement de caisse claire + riser
    rr = riser(4.0, 400, 12500, 1.4)
    place(dry, rr, 18.0, 0.55)
    place(wetbus, rr, 18.0, 0.2)
    k = 0.0
    step = 0.25
    t0 = 19.5
    while t0 < 22.0 - 0.01:
        c = clap(0.2)
        gain = 0.25 + 0.55 * (t0 - 19.5) / 2.5
        place(dry, c, t0, gain)
        step = max(0.0625, step * 0.86)
        t0 += step
    # -------- 22 -> 27 s
    # coupure sèche à 22,0 s : tout est tronqué, puis silence
    cut = int(22.0 * SR)
    fade = np.ones(N)
    fade[cut - int(0.004 * SR):cut] = np.linspace(1, 0, int(0.004 * SR))
    fade[cut:int(22.28 * SR)] = 0.0
    # on applique sur l'ensemble avant la section finale
    dry_pre = dry.copy()
    wet_pre = wetbus.copy()
    dry_pre[:, cut:] = 0
    wet_pre[:, cut:] = 0
    # ----- section finale (ajoutée après la coupure)
    dryF = np.zeros_like(dry)
    wetF = np.zeros_like(dry)
    place(dryF, sub_hit(4.2, 34), 22.28, 0.95)
    place(wetF, sub_hit(2.0, 70) * 0.35, 22.28, 0.5)
    m1 = metal(1000, 3.5)
    place(dryF, m1, 22.32, 0.38, -0.2)
    place(wetF, m1, 22.32, 0.9)
    pd2 = pad([A2, A2 * 1.5, A2 * 1.189, A2 * 3.0], 5.0, lambda u: 500 + 900 * np.sin(u * np.pi))
    pe2 = np.interp(tt(5.0), [0, 1.0, 2.8, 5.0], [0, 0.6, 0.55, 0.0])
    place(dryF, pd2 * pe2 * 0.28, 22.3, 0.8)
    place(wetF, pd2 * pe2 * 0.28, 22.3, 0.5)
    for tc, fq in ((23.75, 1560), (24.95, 2100), (25.3, 3150)):
        mm = metal(fq, 2.2)
        place(dryF, mm, tc, 0.14, 0.3 if fq < 3000 else -0.3)
        place(wetF, mm, tc, 0.55)
    # souffle aérien sur la lueur finale
    ri = riser(1.3, 2000, 12000, 1.4)
    place(dryF, ri, 24.0, 0.14)

    dry_all = dry_pre * sidechain[None, :] * 0 + dry_pre  # sidechain déjà appliqué aux nappes
    # sidechain léger sur la nappe seulement ; kick garde sa dynamique
    wet_in = wet_pre + wetF
    rev = reverb(wet_in, 3.2, 1.0)
    mix = dry_pre + dryF + rev * 0.55
    # le reverb d'avant 22 s ne doit pas déborder sur le noir sec
    mix[:, cut:int(22.28 * SR)] = 0.0
    # fondu final doux
    out_fade = np.interp(np.arange(N) / SR, [0, 0.05, 25.6, 27.0], [0, 1, 1, 0])
    mix *= out_fade[None, :]
    # master : filtre infra, compression douce, limiteur
    mix = np.array([hp(mix[0], 28, 2), hp(mix[1], 28, 2)])
    pk = np.max(np.abs(mix))
    mix = mix / pk * 1.6
    mix = np.tanh(mix * 0.95) / np.tanh(1.6 * 0.95)
    mix = mix / np.max(np.abs(mix)) * 0.89
    wavfile.write("build/lyxor_audio.wav", SR, (mix.T * 32767).astype(np.int16))
    print("peak", np.max(np.abs(mix)), "rms", np.sqrt(np.mean(mix ** 2)))


if __name__ == "__main__":
    main()
