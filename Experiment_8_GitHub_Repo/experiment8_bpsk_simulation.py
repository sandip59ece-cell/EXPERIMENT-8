"""
Experiment 8 — Matched Filtering, ISI and Eye Diagrams
BPSK + Root Raised Cosine (RRC) pulse shaping + AWGN

No communications toolbox is required.
Dependencies: numpy, scipy, matplotlib
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import lfilter
from pathlib import Path

OUTDIR = Path(__file__).resolve().parent

def rrc_filter(beta=0.35, span=8, sps=8):
    """Generate a unit-energy Root Raised Cosine (RRC) FIR pulse."""
    N = span * sps
    t = np.arange(-N/2, N/2 + 1) / sps
    h = np.zeros_like(t, dtype=float)

    for i, ti in enumerate(t):
        if abs(ti) < 1e-12:
            h[i] = 1.0 - beta + 4.0 * beta / np.pi
        elif beta > 0 and abs(abs(ti) - 1.0/(4.0*beta)) < 1e-10:
            h[i] = (beta/np.sqrt(2.0)) * (
                (1.0 + 2.0/np.pi) * np.sin(np.pi/(4.0*beta))
                + (1.0 - 2.0/np.pi) * np.cos(np.pi/(4.0*beta))
            )
        else:
            num = (
                np.sin(np.pi * ti * (1.0 - beta))
                + 4.0 * beta * ti * np.cos(np.pi * ti * (1.0 + beta))
            )
            den = np.pi * ti * (1.0 - (4.0 * beta * ti)**2)
            h[i] = num / den

    h /= np.sqrt(np.sum(h**2))
    return h


def add_awgn(x, snr_db, rng):
    """Add real AWGN for the requested signal-to-noise ratio."""
    p = np.mean(x**2)
    snr = 10.0 ** (snr_db / 10.0)
    noise_var = p / snr
    return x + rng.normal(0.0, np.sqrt(noise_var), size=x.shape)


def tx_rx(bits, beta=0.35, span=8, sps=8, snr_db=8.0, seed=7):
    """
    BPSK transmit -> RRC shaping -> AWGN -> RRC matched filter.
    Total cascade delay = span*sps samples.
    """
    rng = np.random.default_rng(seed)
    symbols = 2.0 * bits.astype(float) - 1.0
    up = np.zeros(len(symbols) * sps)
    up[::sps] = symbols

    h = rrc_filter(beta, span, sps)
    tx = np.convolve(up, h, mode="full")
    rx = add_awgn(tx, snr_db, rng)
    mf = np.convolve(rx, h, mode="full")

    # Each RRC filter has group delay span*sps/2.
    # Therefore total cascade delay is span*sps samples.
    total_delay = span * sps
    sample_index = total_delay + np.arange(len(symbols)) * sps
    sample_index = sample_index[sample_index < len(mf)].astype(int)

    detected = np.where(mf[sample_index] >= 0, 1, 0)
    valid = len(detected)
    ber = np.mean(detected != bits[:valid])
    return {
        "bits": bits[:valid],
        "symbols": symbols[:valid],
        "h": h,
        "tx": tx,
        "mf": mf,
        "sample_index": sample_index,
        "detected": detected,
        "ber": ber,
        "total_delay": total_delay,
        "sps": sps,
    }


def eye_segments(signal, sps, samples_per_trace=2, start=0, count=100):
    """Return overlapping 2-symbol eye traces."""
    L = samples_per_trace * sps
    segs = []
    for k in range(start, start + count):
        a = k * sps
        b = a + L
        if b <= len(signal):
            segs.append(signal[a:b])
    return np.asarray(segs)


def measure_eye_height(mf, sps, delay, offset=0, n_symbols=400):
    """Approximate eye opening at the chosen sampling phase."""
    idx = delay + offset + np.arange(n_symbols) * sps
    idx = idx[(idx >= 0) & (idx < len(mf))]
    vals = mf[idx]
    if len(vals) == 0:
        return np.nan
    pos = vals[vals >= 0]
    neg = vals[vals < 0]
    if len(pos) == 0 or len(neg) == 0:
        return np.nan
    # Robust eye height: lower positive level minus upper negative level.
    return np.percentile(pos, 10) - np.percentile(neg, 90)


def run_experiment():
    rng = np.random.default_rng(10)
    n_bits = 3000
    bits = rng.integers(0, 2, n_bits)

    # 1) Demonstration waveform
    demo = tx_rx(bits[:500], beta=0.35, span=8, sps=8, snr_db=8, seed=10)

    # 2) BER vs SNR
    snrs = np.arange(0, 13, 2)
    bers = []
    for snr in snrs:
        r = tx_rx(bits, beta=0.35, span=8, sps=8, snr_db=float(snr), seed=20 + snr)
        bers.append(r["ber"])

    # 3) Eye height vs SNR
    eye_h = []
    for snr in snrs:
        r = tx_rx(bits, beta=0.35, span=8, sps=8, snr_db=float(snr), seed=50 + snr)
        eye_h.append(measure_eye_height(r["mf"], 8, r["total_delay"]))

    # 4) BER vs sampling offset
    base = tx_rx(bits, beta=0.35, span=8, sps=8, snr_db=6, seed=99)
    offsets = np.arange(-3, 4)
    ber_offset = []
    for off in offsets:
        idx = base["total_delay"] + off + np.arange(len(base["bits"])) * 8
        valid = (idx >= 0) & (idx < len(base["mf"]))
        idx = idx[valid].astype(int)
        d = (base["mf"][idx] >= 0).astype(int)
        ber_offset.append(np.mean(d != base["bits"][:len(d)]))

    # 5) Roll-off comparison for eye diagrams
    rolloffs = [0.2, 0.35, 0.7]
    eye_data = {}
    for beta in rolloffs:
        r = tx_rx(bits[:1200], beta=beta, span=8, sps=8, snr_db=8, seed=123)
        eye_data[beta] = eye_segments(r["mf"], 8, 2, count=120)

    # ---- Plot 1: TX and matched filter output ----
    fig, ax = plt.subplots(figsize=(9, 4.8))
    t1 = np.arange(min(1000, len(demo["tx"]))) / 8
    t2 = np.arange(min(1000, len(demo["mf"]))) / 8
    ax.plot(t1, demo["tx"][:len(t1)], label="RRC-shaped transmit")
    ax.plot(t2, demo["mf"][:len(t2)], label="Matched-filter output", alpha=0.85)
    ax.set_title("BPSK through RRC shaping and matched filtering")
    ax.set_xlabel("Symbol periods")
    ax.set_ylabel("Amplitude")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUTDIR / "01_tx_matched_filter.png", dpi=180)
    plt.close(fig)

    # ---- Plot 2: Eye diagram ----
    fig, ax = plt.subplots(figsize=(8, 4.8))
    seg = eye_data[0.35]
    tt = np.arange(seg.shape[1]) / 8
    for row in seg:
        ax.plot(tt, row, alpha=0.15)
    ax.axvline(1.0, linestyle="--", linewidth=1)
    ax.set_title("Eye diagram after matched filtering (β = 0.35, SNR = 8 dB)")
    ax.set_xlabel("Symbol periods")
    ax.set_ylabel("Amplitude")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUTDIR / "02_eye_diagram.png", dpi=180)
    plt.close(fig)

    # ---- Plot 3: BER vs SNR ----
    fig, ax = plt.subplots(figsize=(7.8, 4.7))
    ax.semilogy(snrs, np.maximum(bers, 1e-5), "o-")
    ax.set_title("BER versus SNR")
    ax.set_xlabel("SNR (dB)")
    ax.set_ylabel("BER")
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUTDIR / "03_ber_vs_snr.png", dpi=180)
    plt.close(fig)

    # ---- Plot 4: Eye height vs SNR ----
    fig, ax = plt.subplots(figsize=(7.8, 4.7))
    ax.plot(snrs, eye_h, "o-")
    ax.set_title("Eye height versus SNR")
    ax.set_xlabel("SNR (dB)")
    ax.set_ylabel("Eye height")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUTDIR / "04_eye_height_vs_snr.png", dpi=180)
    plt.close(fig)

    # ---- Plot 5: BER vs sampling offset ----
    fig, ax = plt.subplots(figsize=(7.8, 4.7))
    ax.plot(offsets, ber_offset, "o-")
    ax.set_title("BER versus sampling offset")
    ax.set_xlabel("Offset (samples)")
    ax.set_ylabel("BER")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUTDIR / "05_ber_vs_sampling_offset.png", dpi=180)
    plt.close(fig)

    # ---- Plot 6: Roll-off eye diagrams, separate panels ----
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), sharey=True)
    for ax, beta in zip(axes, rolloffs):
        seg = eye_data[beta]
        tt = np.arange(seg.shape[1]) / 8
        for row in seg:
            ax.plot(tt, row, alpha=0.12)
        ax.axvline(1.0, linestyle="--", linewidth=1)
        ax.set_title(f"β = {beta}")
        ax.set_xlabel("Symbol periods")
        ax.grid(True, alpha=0.25)
    axes[0].set_ylabel("Amplitude")
    fig.suptitle("Effect of RRC roll-off on eye opening")
    fig.tight_layout()
    fig.savefig(OUTDIR / "06_rolloff_eye_comparison.png", dpi=180)
    plt.close(fig)

    # ---- Save numerical summary ----
    summary = [
        "Experiment 8 simulation summary",
        f"RRC roll-off beta = 0.35",
        f"Span = 8 symbols",
        f"Samples/symbol = 8",
        f"Each RRC filter group delay = span*sps/2 = {8*8//2} samples",
        f"Total cascade delay = span*sps = {8*8} samples",
        f"Detection uses samples only after total delay compensation.",
        "",
        "BER vs SNR:",
    ]
    summary += [f"  {s:>2} dB : {b:.6g}" for s, b in zip(snrs, bers)]
    summary += ["", "BER vs sampling offset at 6 dB:"]
    summary += [f"  {o:+d} samples : {b:.6g}" for o, b in zip(offsets, ber_offset)]
    (OUTDIR / "results.txt").write_text("\n".join(summary), encoding="utf-8")

    return demo, snrs, bers, eye_h, offsets, ber_offset


if __name__ == "__main__":
    run_experiment()
    print("Experiment 8 simulation completed.")
    print("Total cascade delay = span*sps samples.")
