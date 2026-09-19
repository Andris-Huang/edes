#!/usr/bin/env python3
"""
rfsoc_spectrum.py
=================
Frequency-domain view of the RFSoC DDR4 I/Q captures, mapped back to RF.

The DDR4 file holds a COMPLEX baseband stream: ``iq[:, 0]`` = I, ``iq[:, 1]``
= Q, sampled at ``fs`` (``fs_msps`` in the .npz, else ``F_OUT_MHZ`` =
552.96 Msps), after the RF was down-converted with an LO at
``rfsoc_read_freq`` / config ``read_freq`` = 175.7 MHz on this bench. Because
the stream is complex, the two sidebands are NOT folded on top of each other —
the spectrum runs over the full ``[LO - fs/2, LO + fs/2]`` and every bin maps
back to a unique RF frequency::

    f_RF = LO + sideband * f_baseband        (sideband = +1 by default)

Usage — ONE call does everything (paste this into a notebook cell)::

    import sys
    sys.path.insert(0, "/home/electron/edes/edes/experiments/analysis")
    import rfsoc_spectrum as S

    out = S.analyze(11345,                    # RID (or .h5 path, or a list)
                    band_mhz=(175.3, 175.5),  # band to keep; omit -> spectrum only
                    rbw_khz=10,               # resolution bandwidth
                    span_mhz=4,               # plotted span around the LO
                    time_unit="ms")

It draws the RF spectrum with the band shaded, then the band-filtered
``|v(t)|``, prints a summary (peak / floor per capture, in-band rms and peak),
and returns the figures and the data:
``out["spectra"][0]["f"], ["p"]`` and ``out["traces"][0]["t"], ["mean"]``.
``save="~/plots"`` also writes the PNGs. Further examples are in
:func:`analyze`'s docstring.

The pieces are callable on their own too::

    S.plot_rf_spectrum([11344, 11345], noise=11338, events=[1],
                       rbw_khz=20, span_mhz=10, band_mhz=(175.3, 175.5))
    S.plot_band_filtered(11345, band_mhz=(175.3, 175.5), time_unit="ms")

A level that MOVES during the capture — a gated drive, a spur that changes
state, a slow drift — is exactly what an averaged spectrum destroys, so there
is a heatmap that keeps the time axis instead::

    S.plot_spectrogram(11333, events=[1, 8], rbw_khz=50, nframes=300,
                       center_mhz=352, span_mhz=1.5, sideband=-1,
                       time_unit="ms", tlim=(0, 10), pulse=True)

``nframes`` sets the time resolution and ``rbw_khz`` the frequency one (they
trade off through the record length), ``navg=`` caps the samples read per
frame, and ``relative="median"`` subtracts each frequency's own time-median so
that only what CHANGED during the capture is left — the view for a trend.
:func:`spectrogram` returns the same thing as an array.

The same frames read as CURVES rather than as a fill — one spectrum per frame,
coloured by when it was taken — which is what you want once the heatmap has
shown you where to look, because a curve has a dB axis and a heatmap only has a
colour::

    S.plot_spectra_vs_time(11333, events=[1, 8], rbw_khz=20, nframes=160,
                           navg=16, center_mhz=175.4, span_mhz=3.4,
                           sideband=-1, time_unit="ms", phase="on")

``phase="on"``/``"off"`` keeps only the frames inside a drive-on (or -off)
span, so a gated run does not interleave two different spectra along one colour
ramp, and ``offset_db=-4`` fans the curves down the page as a stacked
waterfall.

Analysing only part of the capture: ``tlim`` takes one window or several. With
several, every window is transformed with the same ``nperseg`` and their
segments are POOLED into one average, so the gaps contribute nothing and
introduce no edge artefact \u2014 the way to get a spectrum of, say, only the
drive-off gaps::

    gaps = [(1.5 * k + 1.0, 1.5 * k + 1.5) for k in range(40)]   # ms
    S.analyze(11361, tlim=gaps, time_unit="ms", rbw_khz=10)
    S.windows_from_pulse(run, phase="off")      # the same list, from the config
    S.drive_windows(run, 1, 355.0, phase="off")  # ... or measured off the data,
                                                # which is the only safe one when
                                                # the board re-fires mid-record

Or straight from a shell — the same two figures, written as PNGs::

    python rfsoc_spectrum.py 11345 --rbw 10 --span 4 --band 175.3 175.5
    python rfsoc_spectrum.py 11344 11345 --noise 11338 --events 1 --span 6 --show
    python rfsoc_spectrum.py --help

Units: I/Q are raw ADC/DDC counts, so the y axes are dB relative to 1 count
unless a calibration offset is supplied (``dbm_offset=``, e.g. from a
cross-calibration against the spectrum analyser at the same frequency).
"""
from __future__ import annotations

import os

import numpy as np

import edes.experiments.analysis.sa_rfsoc_corr as C

LO_MHZ_DEFAULT = 175.7        # bench LO / RFSoC read_freq

# dB to ADD to 10*log10(counts**2) to get dBm at the RFSoC input. The DDC
# scaling, the digital gain and the whole receive chain sit between counts and
# an absolute power, so this cannot be known from the file alone: it is a
# one-off cross-calibration against the spectrum analyser. Until it is set the
# dBm axis is a relative scale with 0 dBm == 0 dB re 1 count, and the plots say
# so. Set it once per chain configuration::
#
#     import rfsoc_spectrum as S
#     S.DBM_CAL_DB = -37.4                      # if you already know it
#     S.set_dbm_calibration(peak_dbm=-52.0, peak_db_counts=-7.8)   # or derive it
DBM_CAL_DB = 0.0
_ENBW = {"hann": 1.5, "hamming": 1.3628, "blackman": 1.7269,
         "blackmanharris": 2.0044, "flattop": 3.7702, "boxcar": 1.0}


# --------------------------------------------------------------------------- #
#  loading
# --------------------------------------------------------------------------- #
def _as_run(run):
    return run if C._is_run(run) else C.load_run(run)


def lo_hz(run, lo_mhz=None):
    """the down-conversion frequency for this run, in Hz.

    ``lo_mhz`` wins; otherwise the run's own metadata is used (``read_freq``
    from the RFSoC config, or ``rfsoc_read_freq`` / ``SSA_freq_center`` from
    the ARTIQ arguments), and only then the bench default 175.7 MHz."""
    if lo_mhz is not None:
        return float(lo_mhz) * 1e6
    cfg = (run["rfsoc"].get("rfsoc_status") or {}).get("config", {}) or {}
    for k in ("read_freq", "readout_freq", "adc_freq"):
        if cfg.get(k):
            return float(cfg[k]) * 1e6          # config is in MHz
    v = run["args"].get("rfsoc_read_freq")
    if v:
        return float(v) * 1e6
    v = run["args"].get("SSA_freq_center")
    if v:
        return float(v)                          # ARTIQ argument is in Hz
    return LO_MHZ_DEFAULT * 1e6


def _cli_tlim(vals):
    """``--tlim`` -> what the API wants: a flat list of numbers is one window
    when it is a pair, and several windows when it is several pairs."""
    if not vals:
        return None
    if len(vals) % 2:
        raise SystemExit(f"--tlim needs an even number of values (got {len(vals)})")
    return tuple(vals) if len(vals) == 2 else [tuple(vals[i:i + 2])
                                               for i in range(0, len(vals), 2)]


def _norm_windows(tlim):
    """``tlim`` -> a list of ``(t0, t1)`` pairs, or ``None``.

    One window may be given as a bare ``(t0, t1)``; several disjoint windows as
    ``[(a0, a1), (b0, b1), ...]``. ``None`` at either end of a window means "to
    that edge of the record", so ``(None, 2.0)`` is "everything up to 2".
    The two forms are told apart by whether the first element is itself a pair,
    which makes ``[(0, 1)]`` a one-window list and ``(0, 1)`` a single window."""
    if tlim is None:
        return None
    seq = list(tlim)
    if not seq:
        return None
    if isinstance(seq[0], (tuple, list, np.ndarray)):
        wins = [tuple(w) for w in seq]
    else:
        wins = [tuple(seq)]
    for w in wins:
        if len(w) != 2:
            raise ValueError(f"tlim window {w!r} is not a (t0, t1) pair")
    return wins


def _stitch(parts):
    """concatenate per-window arrays with a NaN between them.

    The NaN is what makes a plotted line BREAK at a gap instead of drawing a
    straight segment across time that was never analysed — a fake trace that
    would otherwise look like real, quiet data."""
    if len(parts) == 1:
        return np.asarray(parts[0])
    out = []
    for i, a in enumerate(parts):
        if i:
            out.append(np.array([np.nan]))
        out.append(np.asarray(a, float))
    return np.concatenate(out)


def load_iq_windows(run, m, tlim=None, time_unit="us"):
    """``(zs, fs, t0s)``: one complex64 array per ``tlim`` window.

    ``tlim`` is a single ``(t0, t1)`` or a list of disjoint windows
    ``[(a0, a1), (b0, b1), ...]``, in ``time_unit``; ``None`` analyses the whole
    capture. The .npz is opened once and only the requested spans are turned
    into complex64, so pulling three 1 ms windows out of a 100 ms capture costs
    a thirtieth of the memory of loading it whole. Windows come back in the
    order given (they are NOT sorted or merged — overlapping ones would be
    counted twice, which is occasionally what you want), and ``t0s`` holds each
    one's start in seconds into the capture, so the pieces stay locatable on
    the original time axis after slicing."""
    p = os.path.join(run["rfsoc"]["fetch_dir"], m["npz"] if isinstance(m, dict) else m)
    if not os.path.exists(p):
        raise FileNotFoundError(f"{p} — rerun with rfsoc_fetch='full'")
    d = np.load(p)
    iq = d["iq"]
    fs = float(d["fs_msps"]) * 1e6 if "fs_msps" in getattr(d, "files", []) \
        else C.F_OUT_MHZ * 1e6
    to_s = C._unit_scale(time_unit, "us") * 1e-6
    zs, t0s = [], []
    for w in (_norm_windows(tlim) or [(None, None)]):
        i0, i1 = 0, iq.shape[0]
        if w[0] is not None:
            i0 = max(0, int(float(w[0]) * to_s * fs))
        if w[1] is not None:
            i1 = min(i1, int(np.ceil(float(w[1]) * to_s * fs)))
        if i1 <= i0:
            raise ValueError(f"tlim window {w} selects no samples")
        # assign the parts instead of `I + 1j*Q`, which would build a complex128
        # temporary (double the memory) on a 55 M-sample record
        z = np.empty(i1 - i0, dtype=np.complex64)
        z.real = iq[i0:i1, 0]
        z.imag = iq[i0:i1, 1]
        zs.append(z)
        t0s.append(i0 / fs)
    return zs, fs, t0s


def load_iq(run, m, tlim=None, time_unit="us"):
    """``(z, fs, t0)`` for one capture: complex64 baseband, sample rate in Hz,
    and the time of the first returned sample (seconds into the capture).

    ``tlim`` crops the record to ``(t0, t1)`` in ``time_unit`` BEFORE anything
    else — worth using on the 100 ms / 55 M-sample captures when only a slice
    matters, since the FFT cost and the memory both scale with the length.
    For several disjoint windows use :func:`load_iq_windows`, which is what
    this one wraps."""
    wins = _norm_windows(tlim)
    if wins is not None and len(wins) > 1:
        raise ValueError("load_iq takes a single (t0, t1); use "
                         "load_iq_windows() for several windows")
    zs, fs, t0s = load_iq_windows(run, m, tlim=tlim, time_unit=time_unit)
    return zs[0], fs, t0s[0]


def windows_from_pulse(run, phase="off", guard_us=0.0, time_unit="us",
                       offset_us=None, include_edges=False, tmax_us=None):
    """The pulse-ON or pulse-OFF spans of a run, ready to hand to ``tlim``.

    Building the gap list by hand for a 40-pulse train is tedious and easy to
    get wrong by one ``crossing_to_pulse_us``, so this reads the run's own
    ``rfsoc_pulse_spec`` and returns the spans in ``time_unit``, measured on
    the DDR4 capture axis (t=0 = the crossing) like every other time argument
    here::

        gaps = S.windows_from_pulse(run, phase="off", guard_us=20)
        S.analyze(run, tlim=gaps, time_unit="us", rbw_khz=10)

    The spans are measured from t=0 of the capture, which IS the pulse trigger
    — the first drive lands at ``pulse_initial_delay_us``, not at
    ``crossing_to_pulse_us`` past it. ``offset_us=`` shifts them all if a run
    needs it.

    ``guard_us`` trims that much off BOTH ends of every span. Use it: the
    switching edges are broadband and will otherwise smear across the spectrum
    you were trying to keep clean. ``phase="on"`` selects the drive bursts
    instead, ``include_edges`` adds the quiet run-in before the first pulse
    (and, with ``tmax_us``, the tail after the last one — left out by default
    because on a 100 ms capture that tail is longer than the whole train and
    would dominate the average).

    Only the ``const`` pulse shape has genuinely rectangular spans; for a ramp
    these are still the on/off intervals, but the drive is not flat inside them."""
    run = _as_run(run)
    spec = run["rfsoc"].get("rfsoc_pulse_spec")
    if not isinstance(spec, dict):
        raise ValueError("this run has no rfsoc_pulse_spec to build windows from")
    # The DDR4 record starts at the PULSE trigger, not at the crossing, so the
    # crossing->pulse latency must NOT be added here. Measured on the 355 MHz
    # line of RID 11168 and the 352 MHz line of RID 11333: the drive turns on
    # at 499.5 / 500.3 us against a pulse_initial_delay_us of 500.0, i.e. at
    # the programmed delay with no extra offset, while adding
    # crossing_to_pulse_us put every window 7.1 us late. ``offset_us=`` still
    # overrides for a run where the two clocks really are separated.
    off0 = 0.0 if offset_us is None else float(offset_us)
    on = float(spec.get("pulse_on_us", spec.get("pulse_length_us", 0.0)) or 0.0)
    gap = float(spec.get("pulse_off_us", 0.0) or 0.0)
    reps = max(1, int(spec.get("pulse_repeats", 1) or 1))
    if on <= 0:
        raise ValueError("pulse_on_us / pulse_length_us is 0 — no pulse to gate on")
    start = off0 + float(spec.get("pulse_initial_delay_us", 0.0) or 0.0)
    ons, offs, t = [], [], start
    for r in range(reps):
        ons.append((t, t + on))
        t += on
        if gap > 0 and (r < reps - 1 or spec.get("pulse_count_trailing_off")):
            offs.append((t, t + gap))
            t += gap
    if include_edges:
        if start > 0:
            offs.insert(0, (0.0, start))
        if tmax_us is not None and float(tmax_us) > t:
            offs.append((t, float(tmax_us)))
    wins = ons if str(phase).lower().startswith("on") else offs
    if not wins:
        raise ValueError(f"no {phase!r} windows in this pulse spec "
                         f"(pulse_off_us={gap}, pulse_repeats={reps})")
    g, k = float(guard_us), C._unit_scale("us", time_unit)
    out = [((a + g) * k, (b - g) * k) for a, b in wins if b - a > 2 * g]
    if not out:
        raise ValueError(f"guard_us={guard_us} is wider than every {phase} span")
    if len(out) < len(wins):
        print(f"[windows] {len(wins) - len(out)} span(s) dropped: narrower "
              f"than 2*guard_us")
    return out


def _drive_envelope(run, m, f_bb, nb):
    """``|drive|`` averaged in ``nb``-sample blocks, from one open of the .npz.

    Mixing by ``exp(-2j pi f_bb t)`` puts the drive at DC and the box-car mean
    over each block is then a one-tap low-pass whose sinc nulls sit at every
    multiple of ``fs/nb``: at nb = 1 us everything more than a few hundred kHz
    away — the electron band 179 MHz off, the f/2 spur, the noise floor — is
    down by tens of dB and what survives is the drive's own envelope. The
    phase ramp restarts in every block, which multiplies each block by a
    constant of unit modulus and so leaves ``|mean|`` untouched."""
    p = os.path.join(run["rfsoc"]["fetch_dir"],
                     m["npz"] if isinstance(m, dict) else m)
    if not os.path.exists(p):
        raise FileNotFoundError(f"{p} — rerun with rfsoc_fetch='full'")
    ph = np.exp(-2j * np.pi * f_bb * np.arange(nb) / _capture_fs(run, m)
                ).astype(np.complex64)
    out = []
    with np.load(p) as d:
        iq = d["iq"]
        n = (iq.shape[0] // nb) * nb
        step = max(nb, (1 << 22) // nb * nb)     # ~4 M samples per pass
        for i0 in range(0, n, step):
            i1 = min(i0 + step, n)
            z = np.empty(i1 - i0, dtype=np.complex64)
            z.real = iq[i0:i1, 0]
            z.imag = iq[i0:i1, 1]
            out.append(np.abs((z.reshape(-1, nb) * ph).mean(1)))
    if not out:
        raise ValueError("capture is shorter than one averaging block")
    return np.concatenate(out).astype(float)


def _schmitt(a, t_dn, t_up):
    """boolean "is on" with hysteresis: latch high at ``t_up``, low at
    ``t_dn``, hold the last decision in between (so envelope ripple around one
    threshold cannot chop a burst into fifty)."""
    hi, lo = a >= t_up, a <= t_dn
    idx = np.where(hi | lo, np.arange(a.size), -1)
    np.maximum.accumulate(idx, out=idx)          # most recent decisive sample
    return np.where(idx >= 0, hi[np.maximum(idx, 0)], False)


def _edge_time(a, k, lvl, step_s, rising, span=8):
    """time [s] of the ``a == lvl`` crossing near block ``k``.

    The Schmitt state flips at ``t_up``/``t_dn``, which is deliberately off the
    middle of the edge, so the state change alone is biased by however long the
    envelope takes to climb. Interpolating the real mid-level crossing between
    the two blocks that straddle it removes that bias and gets the edge to well
    under one block."""
    j0, j1 = max(0, k - span), min(a.size - 2, k + span)
    best = None
    for j in range(j0, j1 + 1):
        up = a[j] < lvl <= a[j + 1]
        if up if rising else (a[j] >= lvl > a[j + 1]):
            if best is None or abs(j - k) < abs(best - k):
                best = j
    if best is None:
        return (k + 1.0) * step_s               # no clean crossing: the flip
    frac = (lvl - a[best]) / (a[best + 1] - a[best])
    return (best + 0.5 + frac) * step_s         # +0.5: block centres


def drive_windows(run, m=None, f_drive_mhz=None, lo=None, sideband=+1,
                  phase="on", guard_us=0.0, time_unit="us", step_us=1.0,
                  thresh=0.5, hyst=0.15, min_us=None, merge_us=None,
                  tmax_us=None, quiet=False, return_envelope=False):
    """The drive spans MEASURED from one capture, not predicted from the config.

    :func:`windows_from_pulse` reads ``rfsoc_pulse_spec``, so it describes the
    one sequence the host programmed. When the DDR4 window outlives that
    sequence the board re-arms inside the record, reads the still-high signal
    as a fresh below->above crossing and fires the whole sequence again — RID
    11166 holds three sequences in a 15 ms window against a 4.5 ms program —
    and the predicted "off" gaps are then full of drive. Gating an OFF spectrum
    on them measures exactly what it was meant to exclude. This finds the
    bursts in the data instead, and is otherwise a drop-in::

        off = S.drive_windows(11166, 1, 355.0, phase="off", guard_us=20)
        S.analyze(11166, events=[1], tlim=off, time_unit="us", rbw_khz=10)

    The capture is mixed to the drive frequency and averaged in ``step_us``
    blocks, leaving the drive's envelope and little else; a Schmitt trigger
    ``thresh`` of the way from the off level to the on level (both robust
    percentiles, so any duty cycle from a few % to ~95 % works) cuts it into
    spans and each edge is interpolated between the straddling blocks.

    ``phase="off"`` returns the complement within the record. ``guard_us``
    trims both ends of every span, as in :func:`windows_from_pulse` — keep
    using it, the switching edges are broadband. ``min_us`` / ``merge_us``
    drop hair and heal one-block dropouts; ``return_envelope`` also hands back
    ``(t_us, amp, level)`` for when a run refuses to split cleanly.

    ``f_drive_mhz`` defaults to the run's ``pulse_freq``. It must be right to
    about ``1/(2*step_us)`` MHz or the box-car filters the drive away with
    everything else; if the two levels come back within 6 dB of each other this
    raises rather than return windows cut out of noise, and a flipped
    ``sideband`` is the usual reason."""
    run = _as_run(run)
    if not isinstance(m, dict):
        evs = _events(run, m)
        if not evs:
            raise ValueError(f"no capture {m!r} in this run")
        m = evs[0]
    cfg = (run["rfsoc"].get("rfsoc_status") or {}).get("config", {}) or {}
    if f_drive_mhz is None:
        f_drive_mhz = cfg.get("pulse_freq")
        if not f_drive_mhz:
            raise ValueError("no f_drive_mhz, and the run's config has no "
                             "pulse_freq to fall back on")
    fs = _capture_fs(run, m)
    f_bb = float(sideband) * (float(f_drive_mhz) * 1e6 - lo_hz(run, lo))
    nb = max(8, int(round(float(step_us) * 1e-6 * fs)))
    step_s, step_us = nb / fs, nb / fs * 1e6
    a = _drive_envelope(run, m, f_bb, nb)

    off_lvl, on_lvl = (float(np.percentile(a, 1.0)),
                       float(np.percentile(a, 99.0)))
    contrast = 20 * np.log10(on_lvl / off_lvl) if off_lvl > 0 else np.inf
    if contrast < 6.0:
        raise ValueError(
            f"the {f_drive_mhz} MHz envelope only swings {contrast:.1f} dB "
            f"({off_lvl:.3g} -> {on_lvl:.3g} counts) — there is no on/off to "
            f"cut on. Check sideband= (the drive lands at +f_bb for +1, -f_bb "
            f"for -1), f_drive_mhz, and that this capture contains drive at all")
    swing = on_lvl - off_lvl
    mid = off_lvl + float(thresh) * swing
    on = _schmitt(a, mid - float(hyst) * swing, mid + float(hyst) * swing)

    T = a.size * step_s
    d = np.diff(on.astype(np.int8))
    ups = [_edge_time(a, k, mid, step_s, True) for k in np.flatnonzero(d == 1)]
    dns = [_edge_time(a, k, mid, step_s, False) for k in np.flatnonzero(d == -1)]
    if on[0]:
        ups.insert(0, 0.0)                      # burst already running at t=0
    if on[-1]:
        dns.append(T)                           # ... or still running at the end
    wins = [(u, v) for u, v in zip(ups, dns) if v > u]

    mn = (5 * step_us if min_us is None else float(min_us)) * 1e-6
    mg = (2 * step_us if merge_us is None else float(merge_us)) * 1e-6
    merged = []
    for w in wins:
        if merged and w[0] - merged[-1][1] < mg:
            merged[-1] = (merged[-1][0], w[1])
        else:
            merged.append(w)
    ons = [w for w in merged if w[1] - w[0] >= mn]
    if not ons:
        raise ValueError(f"every span found was shorter than min_us={mn*1e6:.1f}")

    tmax = T if tmax_us is None else float(tmax_us) * 1e-6
    offs, t = [], 0.0
    for u, v in ons:
        if u - t > mn:
            offs.append((t, u))
        t = v
    if tmax - t > mn:
        offs.append((t, tmax))

    if not quiet:
        dur = np.array([v - u for u, v in ons]) * 1e6
        per = np.diff([u for u, _ in ons]) * 1e6
        spec = run["rfsoc"].get("rfsoc_pulse_spec") or {}
        prog = int(spec.get("pulse_repeats", 0) or 0)
        print(f"[drive_windows] {m.get('npz', '?')}: {len(ons)} burst(s) at "
              f"{f_drive_mhz} MHz, {dur.mean():.1f} us long"
              + (f" (spread {np.ptp(dur):.1f})" if np.ptp(dur) > 2 * step_us else "")
              + (f", every {per.mean():.1f} us" if per.size else "")
              + f"; first at {ons[0][0]*1e6:.1f} us of {T*1e6:.0f} us, "
              f"on/off contrast {contrast:.0f} dB")
        if prog and len(ons) != prog:
            print(f"                the config programs {prog} — this record "
                  f"holds {len(ons) / prog:.3g}x that, so the board re-fired "
                  f"inside the window and windows_from_pulse() is wrong here")
        if ons[-1][1] >= T - step_s:
            print("                the last burst is still on when the record "
                  "ends, so its span is cut short by the capture, not by the drive")
        if np.ptp(dur) > 0.5 * dur.mean():
            print("                burst lengths vary by >50%, which a gated "
                  "drive should not: check sideband= (a mirrored image gates "
                  "too, just 20-30 dB down) and f_drive_mhz=, and raise step_us "
                  "if the envelope is noisy")

    wins = ons if str(phase).lower().startswith("on") else offs
    if not wins:
        raise ValueError(f"no {phase!r} spans in this capture")
    g, k = float(guard_us) * 1e-6, C._unit_scale("s", time_unit)
    out = [((u + g) * k, (v - g) * k) for u, v in wins if v - u > 2 * g]
    if not out:
        raise ValueError(f"guard_us={guard_us} is wider than every {phase} span")
    if len(out) < len(wins) and not quiet:
        print(f"[drive_windows] {len(wins) - len(out)} span(s) dropped: "
              f"narrower than 2*guard_us")
    if return_envelope:
        return out, (np.arange(a.size) + 0.5) * step_us, a, mid
    return out


def _events(run, events=None):
    return C._select_events(run, events)


def _resolve_events(specs, noise_events=None, what="spectrum"):
    """pick each spec's captures IN PLACE and drop the empty ones.

    A background run is NOT subject to the ``events`` filter meant for the
    signal runs — ``noise=11335`` with ``events=8`` must still give the whole
    background, not "RID 11335 has no capture 8" and a silently missing trace.
    ``noise_events`` restricts the background instead (default: all of it)."""
    keep = []
    for sp in specs:
        sp["ev"] = _events(sp["run"], noise_events if sp["noise"] else sp["events"])
        if sp["ev"]:
            keep.append(sp)
        else:
            which = ("background " if sp["noise"] else "")
            want = noise_events if sp["noise"] else sp["events"]
            avail = [C._event_number(m["npz"]) for m in _events(sp["run"])]
            print(f"[{what}] RID {sp['rid']}: no {which}DDR4 capture"
                  + ("" if want is None else f" matches events={want}")
                  + (f" — this run has {avail}" if avail else
                     " — this run has no fetched full capture (MR snapshots "
                     "only? rerun with rfsoc_fetch='full')"))
    specs[:] = keep
    return specs


def _envelope_fs(y, fs, t0, n_points=4000, step=None):
    """like :func:`sa_rfsoc_corr._envelope`, but the time axis is built from
    ``fs`` instead of being passed in — a full ``t`` array for a 55 M-sample
    record is 440 MB that is thrown away immediately after binning."""
    y = np.asarray(y)
    if step is None:
        step = 1 if len(y) <= 2 * n_points else int(np.ceil(len(y) / n_points))
    if step <= 1 or len(y) <= step:
        return t0 + np.arange(len(y)) / fs, y, y, y
    k = (len(y) // step) * step
    m = y[:k].reshape(-1, step)
    return (t0 + np.arange(m.shape[0]) * step / fs,
            m.min(1), m.max(1), m.mean(1))


# --------------------------------------------------------------------------- #
#  spectrum
# --------------------------------------------------------------------------- #
def spectrum(z, fs, lo=None, rbw_hz=1e4, window="hann", sideband=+1,
             average="mean", overlap=0.5, max_samples=None, quiet=False):
    """Welch spectrum of a complex baseband record, on an RF frequency axis.

    ``z`` is one complex array, or a LIST of them — several disjoint time
    windows of the same capture (see :func:`load_iq_windows`). Every window is
    transformed with the same ``nperseg``, so they share a frequency grid, and
    their segments are pooled into one average: the result is the spectrum of
    the selected time only, with no contribution from, and no discontinuity at,
    the gaps between windows.

    ``rbw_hz`` is the NOISE resolution bandwidth actually wanted: the segment
    length is ``nperseg = ENBW(window) * fs / rbw``, rounded up to an
    FFT-friendly length, so the returned ``rbw`` is the value achieved (printed
    in the plot title). It is the ONE knob — bin spacing is ``fs/nperseg``,
    finer than the RBW by the ENBW factor. Longer records simply give more
    segments to average: the noise floor gets smoother, the RBW does not change.
    The RBW a gated spectrum can reach is capped by the SHORTEST window:
    ``rbw >= ENBW(window) / T_window``, so 0.5 ms gaps cannot resolve better
    than ~3 kHz no matter what you ask for. The request is coarsened (with a
    message) rather than windows being dropped, so every window you asked for
    is actually used; ``nwin`` reports how many were.

    Returns ``{"f": RF frequencies [Hz], "p": power per RBW bin [counts²],
    "rbw": achieved RBW [Hz], "nperseg", "nseg", "nwin", "fs", "lo"}``."""
    from scipy import signal as ss
    from scipy.fft import next_fast_len
    enbw = _ENBW.get(str(window).lower(), 1.5)
    pieces = ([np.asarray(z)] if isinstance(z, np.ndarray)
              else [np.asarray(x) for x in z])
    if max_samples:                        # BEFORE nperseg is chosen, so the
        pieces = [x[:int(max_samples)]     # reported rbw/nperseg stay truthful
                  for x in pieces]
    # clamp by the SHORTEST window, not the longest: a window one sample short
    # of the clamp would otherwise be dropped, so asking for a finer RBW than
    # the windows can support would silently discard all but the longest of
    # them. Coarsening everything is the predictable failure, and it says so.
    shortest = min(len(x) for x in pieces)
    # an FFT-friendly length just long enough to reach the RBW asked for, so
    # the achieved RBW lands on (or a hair below) the request instead of on the
    # next 5-smooth length up
    n = min(int(next_fast_len(int(np.ceil(max(8.0, enbw * fs / float(rbw_hz)))))),
            shortest, 1 << 23)
    nov = int(n * overlap)
    if n < enbw * fs / float(rbw_hz) - 1 and not quiet:
        lim = ("the 2**23-sample cap" if n >= shortest > (1 << 23) or n == (1 << 23)
               else f"the shortest window ({shortest} samples, "
                    f"{shortest / fs * 1e3:.3f} ms)")
        print(f"[spectrum] RBW coarsened to {enbw * fs / n / 1e3:.3f} kHz "
              f"(asked {rbw_hz / 1e3:.3f} kHz): reaching it needs nperseg="
              f"{int(np.ceil(enbw * fs / float(rbw_hz)))} samples, limited by "
              f"{lim}. A gated spectrum cannot resolve finer than "
              f"~ENBW/T_window.")
    keep = [x for x in pieces if len(x) >= n]
    if not keep:
        raise ValueError(f"every window is shorter than nperseg={n} "
                         f"({n / fs * 1e3:.4f} ms) — raise rbw_khz or widen them")
    if len(keep) < len(pieces) and not quiet:
        print(f"[spectrum] {len(pieces) - len(keep)} of {len(pieces)} windows "
              f"are shorter than nperseg={n} ({n / fs * 1e3:.4f} ms) and were "
              f"dropped — raise rbw_khz to keep them")
    # one Welch per window, then pool. Weighting each window's result by ITS
    # OWN segment count is what makes the mean identical to running Welch over
    # the concatenation, without a long window quietly counting the same as a
    # short one
    acc, wts, f = [], [], None
    for x in keep:
        f, pw = ss.welch(x, fs=fs, window=window, nperseg=n, noverlap=nov,
                         detrend=False, return_onesided=False,
                         scaling="spectrum", average=average)
        acc.append(pw)
        wts.append(max(1, (len(x) - nov) // (n - nov)))
    P, wts = np.asarray(acc, float), np.asarray(wts, float)
    if str(average).startswith("med") and len(P) > 1:
        # NOTE: this is the median ACROSS windows of each window's own
        # (already median-averaged) spectrum, not the median over the pooled
        # segments — close, but not the same estimator. average="mean" is exact.
        p = np.median(P, 0)
    else:
        p = (P * wts[:, None]).sum(0) / wts.sum()
    f, p = np.fft.fftshift(f), np.fft.fftshift(p)
    lo = float(lo if lo is not None else 0.0)
    if sideband < 0:                       # conjugate DDC: RF = LO - f_bb
        f, p = -f[::-1], p[::-1]
    return {"f": lo + f, "p": np.asarray(p, float), "rbw": enbw * fs / n,
            "nperseg": n, "nseg": int(wts.sum()), "nwin": len(keep),
            "fs": fs, "lo": lo}


# --------------------------------------------------------------------------- #
#  spectrogram: the same spectrum, but resolved in time instead of averaged
# --------------------------------------------------------------------------- #
def _capture_fs(run, m):
    """the capture's sample rate [Hz], read WITHOUT materialising ``iq``."""
    p = os.path.join(run["rfsoc"]["fetch_dir"], m["npz"] if isinstance(m, dict) else m)
    with np.load(p) as d:
        return (float(d["fs_msps"]) * 1e6 if "fs_msps" in d.files
                else C.F_OUT_MHZ * 1e6)


def _capture_nsamples(run, m, fs):
    """length of a capture in samples, from the manifest / status rather than
    from the array itself — reading ``iq.shape`` would load all 55 M samples."""
    if isinstance(m, dict) and m.get("n_samples"):
        return int(m["n_samples"])
    st = run["rfsoc"].get("rfsoc_status") or {}
    if st.get("t_capture_s"):
        return int(float(st["t_capture_s"]) * fs)
    p = os.path.join(run["rfsoc"]["fetch_dir"],
                     m["npz"] if isinstance(m, dict) else m)
    with np.load(p) as d:
        return int(d["iq"].shape[0])


def _load_iq_frames(run, m, starts, nsamp):
    """``nsamp``-long complex64 frames starting at the sample indices
    ``starts``, from one open of the .npz.

    :func:`load_iq_windows` takes times and rounds each end independently, so
    two frames can come back one sample different in length — and a differing
    length makes :func:`spectrum` choose a different ``nperseg``, which would
    put the frames of one spectrogram on different frequency grids. Indexing in
    samples keeps every frame exactly the same size, so the grid is shared and
    the result is a rectangular array."""
    p = os.path.join(run["rfsoc"]["fetch_dir"],
                     m["npz"] if isinstance(m, dict) else m)
    if not os.path.exists(p):
        raise FileNotFoundError(f"{p} — rerun with rfsoc_fetch='full'")
    d = np.load(p)
    iq = d["iq"]
    out = []
    for i0 in starts:
        i0 = int(i0)
        if i0 + nsamp > iq.shape[0]:
            break
        z = np.empty(nsamp, dtype=np.complex64)
        z.real = iq[i0:i0 + nsamp, 0]
        z.imag = iq[i0:i0 + nsamp, 1]
        out.append(z)
    return out


def _warn_frame_aliasing(run, dt_s, span_s, navg):
    """note when the frames are too long to resolve the run's pulse train.

    Each frame is a box-car average over its whole length, so ``nframes`` is a
    sampling rate and the pulse train is the signal being sampled. Once a frame
    spans a sizeable part of a pulse period, every frame contains nearly the
    same amount of drive and the gating flattens out; at an EXACT multiple of
    the period it disappears completely and the drive reads as a steady line
    that was never steady. It is the wagon-wheel effect, and it is silent —
    the plot looks clean and is wrong — so it is worth saying out loud."""
    spec = (run["rfsoc"] or {}).get("rfsoc_pulse_spec")
    if not isinstance(spec, dict):
        return
    on = float(spec.get("pulse_on_us", spec.get("pulse_length_us", 0.0)) or 0.0)
    off = float(spec.get("pulse_off_us", 0.0) or 0.0)
    if on <= 0 or off <= 0:
        return
    short_us, per_us, dt_us = min(on, off), on + off, dt_s * 1e6
    if dt_us <= short_us / 3:              # >=3 frames across the shortest
        return                             # phase: the train is resolved
    want = int(np.ceil(span_s / (short_us / 4 * 1e-6)))
    ratio = dt_us / per_us
    head = (f"[spectrogram] frames are {dt_us:.0f} us long against a "
            f"{per_us:.0f} us pulse period ({on:.0f} on / {off:.0f} off)")
    if dt_us < short_us:                   # resolved, but only just
        print(f"{head}: the {short_us:.0f} us phase spans only "
              f"{short_us / dt_us:.1f} frames, so its edges will be soft. "
              f"nframes>={want} puts 4 frames across it.")
        return
    flat = abs(ratio - round(ratio)) < 0.06 and ratio >= 0.94
    print(f"{head} — a frame is longer than the {short_us:.0f} us phase, so the "
          f"gating is ALIASED"
          + (f" and, at {ratio:.2f} of a period, flattened away entirely: a "
             f"pulsed drive will read as a steady line" if flat else "")
          + f". Use nframes>={want}."
          + (f" (navg={navg} also narrows each frame to its first segments, "
             f"which strobes the train harder — navg=0 averages the whole "
             f"frame.)" if navg else ""))


def spectrogram(run, m, lo=None, rbw_hz=5e4, nframes=200, navg=0,
                window="hann", sideband=+1, average="mean", overlap=0.5,
                tlim=None, time_unit="us"):
    """Spectrum of one capture as a function of time: a ``(nt, nf)`` array.

    :func:`spectrum` pools every segment of the record into ONE average, which
    is what you want for a noise floor and exactly wrong for a level that
    moves — a drive that gates on and off, or a spur that changes state partway
    through, averages into a single number that describes neither phase. This
    slices the capture into ``nframes`` consecutive frames and transforms each
    one on its own, so the time axis survives.

    ``nframes`` is a SAMPLING RATE, not just a cosmetic setting: each frame is
    a box-car average over its own length, so a frame comparable to a pulse
    period contains nearly the same amount of drive as its neighbours and the
    gating flattens out — at an exact multiple of the period it vanishes and a
    pulsed drive reads as a steady line. Keep frames under about a quarter of
    the shortest on/off phase; the function says so when they are not.

    The two resolutions trade off against each other through the record: the
    frame length is ``span / nframes`` and the segment length is
    ``ENBW(window) * fs / rbw_hz``, so asking for many frames AND a fine RBW
    eventually leaves under one segment per frame and :func:`spectrum`
    coarsens the RBW (it says so). ``nframes=200`` over a 100 ms capture gives
    0.5 ms frames, which resolve a 1 ms / 0.5 ms pulse train.

    ``navg`` caps how much of each frame is actually transformed: ``0`` (the
    default) uses all of it, and a small number — ``navg=4`` averages four
    Welch segments and reads only the samples those need — turns a 55 M-sample
    capture into a few hundred MB less work for the same picture. The samples
    come from the START of each frame, so with a frame straddling a pulse edge
    a capped ``navg`` sees only the earlier phase.

    ``tlim=(t0, t1)`` restricts the whole thing to part of the capture (one
    window only — the frames must be consecutive for the time axis to mean
    anything). Everything else is :func:`spectrum`'s.

    Returns ``{"t": frame-centre times [s into the capture], "f": RF
    frequencies [Hz], "P": power per RBW bin [counts²], shape (len(t), len(f)),
    "rbw", "nperseg", "nseg", "dt", "fs", "lo"}``."""
    from scipy.fft import next_fast_len
    run = _as_run(run)
    fs = _capture_fs(run, m)
    ntot = _capture_nsamples(run, m, fs)
    wins = _norm_windows(tlim)
    if wins is not None and len(wins) > 1:
        raise ValueError("spectrogram takes a single (t0, t1) — its frames have "
                         "to be consecutive for the time axis to mean anything")
    to_s = C._unit_scale(time_unit, "us") * 1e-6
    i0, i1 = 0, ntot
    if wins:
        if wins[0][0] is not None:
            i0 = max(0, int(float(wins[0][0]) * to_s * fs))
        if wins[0][1] is not None:
            i1 = min(i1, int(np.ceil(float(wins[0][1]) * to_s * fs)))
    if i1 - i0 < 16:
        raise ValueError(f"tlim {tlim} selects {i1 - i0} samples")

    nframes = max(1, int(nframes))
    step = (i1 - i0) / nframes                     # frame pitch, in samples
    nsamp = int(step)                              # frame length, in samples
    enbw = _ENBW.get(str(window).lower(), 1.5)
    if navg:                                       # only load what navg needs
        n = min(int(next_fast_len(int(np.ceil(max(8.0, enbw * fs / float(rbw_hz)))))),
                nsamp, 1 << 23)
        nov = int(n * overlap)
        nsamp = min(nsamp, n + max(0, int(navg) - 1) * max(1, n - nov))
    if nsamp < 16:
        raise ValueError(f"nframes={nframes} leaves {nsamp} samples per frame — "
                         f"lower it or widen tlim")

    starts = [int(i0 + k * step) for k in range(nframes)]
    zs = _load_iq_frames(run, m, starts, nsamp)
    if not zs:
        raise ValueError("no frame fits in the capture")
    lo = lo_hz(run, None) if lo is None else float(lo)
    rows, f, sx = [], None, None
    for k, z in enumerate(zs):
        # every frame is the same length, so the RBW note would be identical
        # for all of them — let the first frame speak and silence the rest
        sx = spectrum(z, fs, lo=lo, rbw_hz=rbw_hz, window=window,
                      sideband=sideband, average=average, overlap=overlap,
                      quiet=bool(k))
        f = sx["f"]
        rows.append(sx["p"])
    t = np.array([(s + nsamp / 2) / fs for s in starts[:len(rows)]])
    _warn_frame_aliasing(run, step / fs, (i1 - i0) / fs, navg)
    return {"t": t, "f": f, "P": np.vstack(rows), "rbw": sx["rbw"],
            "nperseg": sx["nperseg"], "nseg": sx["nseg"], "dt": step / fs,
            "fs": fs, "lo": lo}


def set_dbm_calibration(peak_dbm, peak_db_counts):
    """Pin the dBm scale from ONE measurement: the level a tone reads on the
    spectrum analyser (``peak_dbm``) and what the same tone reads here
    (``peak_db_counts``, the "peak … dB" the summary prints with the
    calibration still at 0). Sets and returns :data:`DBM_CAL_DB`."""
    global DBM_CAL_DB
    DBM_CAL_DB = float(peak_dbm) - float(peak_db_counts)
    print(f"[dBm] DBM_CAL_DB = {DBM_CAL_DB:+.2f} dB "
          f"({peak_db_counts:+.1f} dB re 1 count  ==  {peak_dbm:+.1f} dBm)")
    return DBM_CAL_DB


def _offset(dbm_offset):
    """the dB offset actually applied: an explicit ``dbm_offset`` wins, else the
    module-wide :data:`DBM_CAL_DB`."""
    return float(DBM_CAL_DB if dbm_offset is None else dbm_offset)


def _db(p, db_ref=1.0, dbm_offset=0.0):
    with np.errstate(divide="ignore", invalid="ignore"):
        return (10 * np.log10(np.where(p > 0, p, np.nan) / db_ref ** 2)
                + _offset(dbm_offset))


def _crop(f, p, center, span):
    if span is None:
        return f, p
    c = float(center if center is not None else np.mean(f))
    w = (f >= c - span / 2) & (f <= c + span / 2)
    return (f[w], p[w]) if w.any() else (f, p)


# --------------------------------------------------------------------------- #
#  colours: the background is grey, every signal curve gets its own colour
# --------------------------------------------------------------------------- #
def _noise_alpha(noise_alpha, noise_on_top):
    """how opaque the background curve should be.

    ``None`` means "pick for me": a curve drawn BEHIND the signal may as well
    be opaque, but one drawn in FRONT has to be see-through or it hides the
    very traces it is there to be compared against."""
    if noise_alpha is not None:
        return float(noise_alpha)
    return 0.70 if noise_on_top else 0.95


NOISE_COLOR = "k"          # the background (noise) curve: black, in the back
NOISE_BAND_COLOR = "0.45"  # grey spread band around an averaged background


def _is_grey(c):
    import matplotlib.colors as mcolors
    r, g, b = mcolors.to_rgb(c)
    return max(r, g, b) - min(r, g, b) < 0.10


def _sig_colors(n, cmap=None):
    """``n`` distinct colours for the SIGNAL curves, with the greys removed.

    tab10/tab20 both contain a grey of their own (``tab:gray``), and the
    background is drawn achromatic (black line, grey spread band) — a signal
    curve in that grey would read as a background one. Ask the palette for a
    few extra colours instead and drop the achromatic ones."""
    if n <= 0:
        return []
    for k in range(n, n + 13):
        keep = [c for c in C._event_colors(k, cmap) if not _is_grey(c)]
        if len(keep) >= n:
            return keep[:n]
    return C._event_colors(n, cmap)


def _bg_colors(base, k):
    """``k`` shades of the background colour — grey shades for a grey base.

    ``C._shades`` rebuilds its tints in HLS at a fixed hue, which for an
    achromatic base (``"k"``, ``"0.45"``, …) is hue 0: the family it returns
    drifts towards pink. Ramp the lightness instead and keep r = g = b."""
    import matplotlib.colors as mcolors
    r, g, b = mcolors.to_rgb(base)
    if not _is_grey(base):
        return C._shades(base, k)
    if k <= 1:
        return [(r, g, b)]
    l0 = (r + g + b) / 3
    return [(x, x, x) for x in np.linspace(max(0.12, l0 - 0.22),
                                           min(0.78, l0 + 0.22), k)]


def plot_rf_spectrum(runs, noise=None, events=None, lo_mhz=None, rbw_khz=10.0,
                     span_mhz=None, center_mhz=None, window="hann", sideband=+1,
                     average="mean", tlim=None, time_unit="us", db_ref=1.0,
                     dbm_offset=None, dbm=True, band_mhz=None, figsize=None,
                     ylim=None, lw=1.2, big_font=True, cmap=None,
                     color_by="auto", noise_color=NOISE_COLOR, noise_lw=None,
                     noise_alpha=None, noise_on_top=False, noise_average=True,
                     noise_stat="mean", noise_events=None, ax=None,
                     legend_loc="best", title=None, return_data=False):
    """RF spectrum of the DDR4 captures — every event overlaid on one axes.

    ``runs``/``noise``/``events`` work exactly as in
    :func:`sa_rfsoc_corr.plot_ddr4_events_multi`: RIDs, ``.h5`` paths, loaded
    runs, ``(rid, events)`` tuples or spec dicts, and a run named in ``noise``
    is drawn in grey as the background spectrum (it does not have to be listed
    in ``runs``).

    ``rbw_khz`` sets the resolution bandwidth and ``span_mhz``/``center_mhz``
    the plotted window (the span defaults to everything the sample rate
    covers, ``fs`` = 552.96 MHz wide, and the centre to the LO).
    ``lo_mhz`` overrides the LO taken from the run metadata, and
    ``sideband=-1`` flips the mapping for a conjugated DDC.
    ``band_mhz=(f1, f2)`` shades a band of interest — the same band you would
    then pass to :func:`plot_band_filtered`.

    ``tlim=(t0, t1)`` (in ``time_unit``) restricts the analysis to part of the
    capture, e.g. the few hundred µs around an event. Several disjoint windows
    ``tlim=[(a0, a1), (b0, b1), ...]`` are pooled into ONE spectrum, which is
    how you analyse only the drive-off gaps, or only the pulses, without the
    other phase contributing to the average.

    y is ``10·log10`` of the power in each RBW bin, labelled **dBm** via
    :data:`DBM_CAL_DB` (or an explicit ``dbm_offset``). With the calibration
    still at its 0 dB default the scale is relative — 0 dBm ≡ 0 dB re 1 count —
    and the plot says so in a corner note; :func:`set_dbm_calibration` pins it
    to a spectrum-analyser reading. ``dbm=False`` labels the axis
    "dB re 1 count" instead.

    ``noise=<rid>`` adds the background spectrum of that run: by default every
    capture in it is averaged into ONE black dashed trace
    (``RID 11338 · mean of 11 captures [bg]``), drawn BEHIND the signal traces
    so that it reads as the floor the coloured curves stand on rather than as
    one more curve (``noise_on_top=True`` puts it in front). The average
    is taken in POWER (linear), not in dB, which is the meaningful average for
    a noise floor; ``noise_stat="median"`` is the robust alternative when one
    capture of the noise run caught a real event.

    ``noise_events`` selects which of the noise run's captures go into that
    average (default: all of them, independent of the ``events`` filter used
    for the signal runs). ``noise_average=False`` goes back to one trace per
    background capture. ``noise_color`` (default black, :data:`NOISE_COLOR`),
    ``noise_lw``, ``noise_alpha`` and ``noise_on_top`` control how loud it is
    and whether it sits behind or in front of the signal."""
    import matplotlib.pyplot as plt
    specs = C._norm_run_specs(runs, noise=noise, events=events)
    _resolve_events(specs, noise_events)
    if not specs:
        print("no DDR4 capture matches")
        return
    sig = [sp for sp in specs if not sp["noise"]]
    per_event = (color_by == "event" or (color_by == "auto" and len(sig) <= 1))
    if per_event:
        pal = _sig_colors(sum(len(sp["ev"]) for sp in sig), cmap)
        i = 0
        for sp in sig:
            sp["colors"] = pal[i:i + len(sp["ev"])]
            i += len(sp["ev"])
    else:
        for sp, col in zip(sig, _sig_colors(len(sig), cmap)):
            sp["colors"] = C._shades(sp["color"] or col, len(sp["ev"]))
    for sp in specs:
        if sp["noise"]:
            sp["colors"] = _bg_colors(noise_color, max(1, len(sp["ev"])))

    out, rbw_got, nseg = [], None, None
    with C._rc(big_font=big_font):
        if ax is None:
            fig, ax = plt.subplots(constrained_layout=True,
                                   figsize=figsize or (10.5, 5.8))
        crop = (None if center_mhz is None else center_mhz * 1e6,
                None if span_mhz is None else span_mhz * 1e6)

        def _draw(f, p, lab, col, bg, f0, ev=None, rid=None):
            f, p = _crop(f, p, f0 if crop[0] is None else crop[0], crop[1])
            ax.plot(f / 1e6, _db(p, db_ref, dbm_offset), color=col,
                    lw=(noise_lw or lw * 1.35) if bg else lw,
                    ls=(0, (5, 2.5)) if bg else "-",
                    alpha=_noise_alpha(noise_alpha, noise_on_top) if bg else 1.0,
                    zorder=(3.4 if noise_on_top else 1.8) if bg else 2.5,
                    label=lab)
            out.append({"rid": rid, "event": ev, "label": lab, "color": col,
                        "noise": bg, "f": f, "p": p, "rbw": rbw_got, "lo": f0})

        for sp in specs:
            f0 = lo_hz(sp["run"], lo_mhz)
            bg = sp["noise"]
            # a background run is averaged over ALL its captures (unless
            # noise_events restricts it) into one trace; signal runs keep one
            # trace per capture
            ev_list = sp["ev"]
            stack, fref = [], None
            for m, col in zip(ev_list, list(sp["colors"]) + [noise_color] * len(ev_list)):
                try:
                    zs, fs, _ = load_iq_windows(sp["run"], m, tlim=tlim,
                                                time_unit=time_unit)
                except (FileNotFoundError, ValueError) as e:
                    print(f"[spectrum] RID {sp['rid']} {m['npz']}: {e}")
                    continue
                sx = spectrum(zs, fs, lo=f0, rbw_hz=rbw_khz * 1e3, window=window,
                              sideband=sideband, average=average)
                del zs
                rbw_got, nseg = sx["rbw"], sx["nseg"]
                if bg and noise_average:
                    if fref is None:
                        fref = sx["f"]
                    stack.append(sx["p"] if len(sx["f"]) == len(fref)
                                 else np.interp(fref, sx["f"], sx["p"]))
                    continue
                _draw(sx["f"], sx["p"],
                      f"{sp['label']} · {C._event_label(m['npz'])}"
                      + (" [bg]" if bg else ""), col, bg, f0,
                      C._event_number(m["npz"]), sp["rid"])
            if stack:
                # average the POWER, not the dB — the mean of a noise floor
                rows = np.vstack(stack)
                pm = (np.nanmedian(rows, 0) if str(noise_stat).startswith("med")
                      else np.nanmean(rows, 0))
                _draw(fref, pm,
                      f"{sp['label']} · {noise_stat} of {len(stack)} capture(s) [bg]",
                      noise_color, True, f0, None, sp["rid"])
        if not out:
            print("[spectrum] nothing to plot")
            return
        if band_mhz is not None:
            ax.axvspan(band_mhz[0], band_mhz[1], color="0.45", alpha=.12, lw=0,
                       zorder=0.5, label=f"band {band_mhz[0]:g}–{band_mhz[1]:g} MHz")
            for fx in band_mhz:
                ax.axvline(fx, color="0.35", lw=1.0, ls=(0, (4, 3)), alpha=.75,
                           zorder=0.6)
        h, l = ax.get_legend_handles_labels()
        fs_leg = max(9, plt.rcParams["legend.fontsize"] - 1)
        if legend_loc == "best":
            ax.legend(h, l, loc="upper center", bbox_to_anchor=(0.5, -0.16),
                      ncol=min(3, len(h)), frameon=False, fontsize=fs_leg)
        else:
            ax.legend(h, l, loc=legend_loc, fontsize=fs_leg, framealpha=.9)
        lo_show = out[0]["lo"] / 1e6
        if title is None:
            title = (f"RF spectrum — LO {lo_show:g} MHz, RBW "
                     f"{rbw_got/1e3:.2f} kHz, {nseg} avg"
                     + (f", span {span_mhz:g} MHz" if span_mhz else ""))
        import textwrap
        ax.set_title(textwrap.fill(title, 78))
        ax.set_xlabel("RF frequency [MHz]")
        ax.set_ylabel("Normalized Power Spectrum (dBm)" if dbm else
                      "power in RBW [dB re 1 count]")
        if span_mhz is None:                  # no span asked for -> show it all
            ax.set_xlim(min(t["f"][0] for t in out) / 1e6,
                        max(t["f"][-1] for t in out) / 1e6)
        # if dbm and not _offset(dbm_offset):
        #     # a corner note would land on the traces — put it under the axes,
        #     # left of the legend
        #     ax.figure.text(0.005, 0.004,
        #                    "dBm uncalibrated: 0 dBm ≡ 0 dB re 1 count — set "
        #                    "S.DBM_CAL_DB / S.set_dbm_calibration(sa_dbm, here_db)",
        #                    fontsize=8.5, color="0.45", va="bottom", ha="left")
        if ylim is not None:
            ax.set_ylim(*ylim)
    return (ax, {"spectra": out, "rbw": rbw_got, "nseg": nseg}) if return_data else ax

def _reduce_f(f, P, nf_max, how="max"):
    """block-reduce the frequency axis for display.

    A full-span spectrogram is ``nframes x nperseg`` cells — 200 x 16632 here —
    which pcolormesh will draw one quad at a time and no screen can resolve.
    Reducing in blocks keeps it responsive; ``"max"`` is the spectrum-analyser's
    peak detect and is what keeps a narrow spur visible, where ``"mean"``
    dilutes it into the floor but reports honest levels."""
    nf = len(f)
    if nf_max is None or nf <= int(nf_max):
        return f, P, 1
    k = int(np.ceil(nf / float(nf_max)))
    m = (nf // k) * k
    fb = f[:m].reshape(-1, k).mean(1)
    Pb = P[:, :m].reshape(P.shape[0], -1, k)
    return fb, (Pb.max(2) if how == "max" else Pb.mean(2)), k


def plot_spectrogram(runs, events=None, noise=None, lo_mhz=None, rbw_khz=50.0,
                     nframes=200, navg=0, span_mhz=None, center_mhz=None,
                     band_mhz=None, window="hann", sideband=+1, average="mean",
                     tlim=None, time_unit="ms", relative="none", db_ref=1.0,
                     dbm_offset=None, dbm=True, cmap=None, vmin=None, vmax=None,
                     clip=(1.0, 99.9), nf_max=None, nf_reduce="max",
                     pulse=False, figsize=None, big_font=True, sharec=True,
                     title=None, ax=None, return_data=False):
    """Spectrum vs time as a heatmap — one panel per capture.

    x is time in the capture, y is RF frequency and the colour is the power in
    each RBW bin. Where :func:`plot_rf_spectrum` averages the whole record into
    one curve, this keeps the time axis, so anything that MOVES — the drive
    gating on and off, a spur that changes state partway through the capture,
    a level that drifts over the 100 ms — shows up as structure instead of
    being averaged away.

    ``runs``/``events``/``noise`` are :func:`plot_rf_spectrum`'s, and every
    selected capture gets its own panel on a shared time, frequency and colour
    axis, so panels are directly comparable event to event. ``nframes`` sets
    the time resolution and ``rbw_khz`` the frequency resolution; they trade
    off through the record length (see :func:`spectrogram`). ``navg`` caps the
    samples read per frame — worth setting to 4–8 on a full 100 ms capture.

    ``relative`` picks what the colour means, and with it the colour map:

    * ``"none"`` — absolute level, on a sequential map (dark = quiet).
    * ``"median"`` / ``"mean"`` — dB relative to that panel's own time-median
      (or mean) spectrum, on a diverging map centred at 0 dB. **This is the
      view for a trend**: every static spur and the whole shape of the
      passband subtract away, and only what actually changed during the
      capture is left, red above / blue below its own average.
    * ``"first"`` — dB relative to the first frame, for a level that drifts
      monotonically rather than switching.

    ``span_mhz``/``center_mhz`` crop the frequency axis (do use them — the full
    553 MHz span is mostly empty), ``band_mhz=(f1, f2)`` draws the band edges,
    and ``pulse=True`` marks the programmed drive on/off spans along the top
    from ``rfsoc_pulse_spec``. The colour limits are picked per mode: percentiles of the plotted
    data (``clip``) for an absolute map, and — because a relative map is
    almost all noise, so any percentile would just track the noise and fill
    the picture with speckle — a robust ±6 sigma for a relative one, which
    leaves the noise near white and saturates anything that really moved.
    ``vmin``/``vmax`` override, ``sharec=False`` gives each panel its own.

    Above ``nf_max`` frequency bins the display is block-reduced by peak
    detect (``nf_reduce="mean"`` to average instead) — the title says by how
    much. ``nf_max=None`` (the default) sets it to about one row per PIXEL of
    the panel, which is what keeps a one-bin line visible on a full-span plot:
    reduce less than that and the line is drawn a fifth of a pixel high and
    antialiases into the background, present in the array and invisible on the
    page. The returned data is always the full-resolution array."""
    import matplotlib.pyplot as plt
    specs = C._norm_run_specs(runs, noise=noise, events=events)
    _resolve_events(specs)
    if not specs:
        print("no DDR4 capture matches")
        return
    jobs = [(sp, m) for sp in specs for m in sp["ev"]]
    rel = str(relative).lower()
    if rel not in ("none", "median", "mean", "first"):
        raise ValueError("relative must be 'none', 'median', 'mean' or 'first'")

    out, kred = [], 1
    for sp, m in jobs:
        f0 = lo_hz(sp["run"], lo_mhz)
        try:
            sg = spectrogram(sp["run"], m, lo=f0, rbw_hz=rbw_khz * 1e3,
                             nframes=nframes, navg=navg, window=window,
                             sideband=sideband, average=average, tlim=tlim,
                             time_unit=time_unit)
        except (FileNotFoundError, ValueError) as e:
            print(f"[spectrogram] RID {sp['rid']} {m['npz']}: {e}")
            continue
        lab = (f"{sp['label']} · {C._event_label(m['npz'])}"
               + (" [bg]" if sp["noise"] else ""))
        out.append({"rid": sp["rid"], "event": C._event_number(m["npz"]),
                    "label": lab, "noise": sp["noise"], "run": sp["run"], **sg})
    if not out:
        print("[spectrogram] nothing to plot")
        return
    n = len(out)

    # a full-span capture is nframes x nperseg cells — 200 x 16632 — and the
    # panel is ~230 px tall. Left alone, a one-bin drive line is drawn 0.19 px
    # high and antialiases into the background: the level is in the array and
    # invisible on the page. Reduce to about one row per PIXEL instead, so that
    # the peak detect below has somewhere to put its peak.
    fsz = figsize or (10.5, max(3.0, 2.3 * n))
    if nf_max is None:
        nf_max = int(np.clip(fsz[1] / n * plt.rcParams["figure.dpi"] * 0.78,
                             120, 2000))

    # crop -> reduce -> to dB, in that order: cropping first is what makes the
    # reduction unnecessary on a zoomed plot, and reducing before the dB keeps
    # the peak detect a peak in POWER
    crop_c = f0 if center_mhz is None else center_mhz * 1e6
    for o in out:
        fc, pc = _crop(o["f"], o["P"].T, crop_c,
                       None if span_mhz is None else span_mhz * 1e6)
        fr, Pr, kred = _reduce_f(fc, np.ascontiguousarray(pc.T), nf_max, nf_reduce)
        D = _db(Pr, db_ref, dbm_offset)
        if rel in ("median", "mean"):
            D = D - (np.nanmedian(D, 0) if rel == "median" else np.nanmean(D, 0))
        elif rel == "first":
            D = D - D[0]
        o["f_plot"], o["D"] = fr, D

    if cmap is None:
        cmap = "magma" if rel == "none" else "RdBu_r"
    lo_c, hi_c = (clip or (0, 100))

    def _clim(D):
        """colour limits for one panel's dB array.

        Percentiles are right for an ABSOLUTE map, where the passband shape
        fills the range. They are wrong for a relative one: there almost every
        cell is noise, so any percentile tracks the noise spread and repaints
        the whole map as speckle however clean the data is — the better the
        averaging, the tighter the scale and the louder the speckle. Scale that
        one off a ROBUST sigma instead, so noise stays near white and only a
        real change takes colour."""
        D = D[np.isfinite(D)]
        if rel == "none":
            return np.percentile(D, lo_c), np.percentile(D, hi_c)
        sig = 1.4826 * np.median(np.abs(D - np.median(D)))
        a = max(6.0 * sig, 1.0)
        return -a, a

    if sharec:
        lim = _clim(np.concatenate([o["D"].ravel() for o in out]))
    ts = C._unit_scale("s", time_unit)
    with C._rc(big_font=big_font):
        if ax is None:
            fig, axs = plt.subplots(n, 1, sharex=True, sharey=True,
                                    constrained_layout=True, squeeze=False,
                                    figsize=fsz)
            axs = list(axs[:, 0])
        else:
            fig, axs = ax.figure, [ax] * n
        im = None
        for a, o in zip(axs, out):
            if not sharec:
                lim = _clim(o["D"].ravel())
            im = a.pcolormesh(o["t"] * ts, o["f_plot"] / 1e6, o["D"].T,
                              cmap=cmap, shading="auto", rasterized=True,
                              vmin=lim[0] if vmin is None else vmin,
                              vmax=lim[1] if vmax is None else vmax)
            a.set_ylabel("RF [MHz]")
            # above the panel, not inside it: a label box floating over the
            # data will sooner or later sit exactly on the feature being
            # looked for — as it did on the drive bursts of a 30 ms capture
            a.set_title(o["label"], loc="left", pad=12.0 if pulse else 3.0,
                        fontsize=max(9, plt.rcParams["font.size"] - 3),
                        color="0.25")
            if band_mhz is not None:
                # a band edge outside the cropped span would stretch the y axis
                # and leave a white strip where there is no data — draw only
                # the edges that fall inside it
                ylo, yhi = o["f_plot"][0] / 1e6, o["f_plot"][-1] / 1e6
                for fx in band_mhz:
                    if ylo <= fx <= yhi:
                        a.axhline(fx, color="0.85" if rel == "none" else "0.25",
                                  lw=1.0, ls=(0, (4, 3)), alpha=.8)
            if pulse:
                try:
                    ons = windows_from_pulse(o["run"], phase="on",
                                             time_unit=time_unit)
                except (ValueError, KeyError):
                    ons = []
                # the marks are drawn unclipped so they sit ON the top spine;
                # that also lets a pulse train longer than the plotted slice
                # stretch the x axis, so pin the limits back to the data
                xl = (o["t"][0] * ts, o["t"][-1] * ts)
                for w in ons:
                    if w[1] < xl[0] or w[0] > xl[1]:
                        continue
                    # in axes-fraction y just above the spine, so the marks
                    # neither cover the data nor land on the panel title
                    a.plot([max(w[0], xl[0]), min(w[1], xl[1])], [1.035, 1.035],
                           transform=a.get_xaxis_transform(), color="tab:green",
                           lw=3.5, solid_capstyle="butt", clip_on=False,
                           zorder=5)
                a.set_xlim(*xl)
        axs[-1].set_xlabel(f"time in capture [{C._TIME_UNIT_LABEL[time_unit]}]")
        unit = ("dB re its own " + rel if rel != "none"
                else ("dBm" if dbm else "dB re 1 count"))
        fig.colorbar(im, ax=axs, label=unit, pad=0.015, aspect=max(14, 12 * n))
        if title is None:
            title = (f"Spectrum vs time — RBW {out[0]['rbw']/1e3:.2f} kHz, "
                     f"{len(out[0]['t'])} frames of {out[0]['dt']*1e3:.3f} ms"
                     + (f", {out[0]['nseg']} avg/frame" if out[0]["nseg"] > 1 else "")
                     + (f", {kred}-bin peak detect" if kred > 1 and
                        nf_reduce == "max" else
                        f", {kred}-bin mean" if kred > 1 else ""))
        import textwrap
        fig.suptitle(textwrap.fill(title, 78))
    return (axs, out) if return_data else axs


def _truncate_cmap(cmap, lo=0.06, hi=0.88, n=256):
    """the middle of a colour map, as a colour map.

    Sequential maps are built to end at or near white (magma, inferno) or at a
    bright yellow (viridis, plasma) — fine as a heatmap fill, where the cell
    behind is the reference, and bad as a LINE colour on a white page, where
    the last curves drawn simply vanish. Dropping the extreme ends keeps the
    ramp monotone and every curve visible."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    base = plt.get_cmap(cmap)
    return LinearSegmentedColormap.from_list(
        f"{base.name}_trunc", base(np.linspace(lo, hi, n)))


def _frame_phase_mask(run, t, phase, guard_us=0.0):
    """which frame centre times ``t`` [s] fall in the pulse ON (or OFF) spans."""
    wins = windows_from_pulse(run, phase=phase, guard_us=guard_us, time_unit="s")
    keep = np.zeros(len(t), bool)
    for a, b in wins:
        keep |= (t >= a) & (t <= b)
    return keep


def plot_spectra_vs_time(runs, events=None, noise=None, lo_mhz=None,
                         rbw_khz=50.0, nframes=40, navg=0, span_mhz=None,
                         center_mhz=None, band_mhz=None, window="hann",
                         sideband=+1, average="mean", tlim=None, time_unit="ms",
                         phase=None, guard_us=0.0, relative="none", db_ref=1.0,
                         dbm_offset=None, dbm=True, cmap="viridis",
                         cmap_range=(0.06, 0.88), offset_db=0.0,
                         lw=0.9, alpha=0.9, nf_max=4000, nf_reduce="max",
                         figsize=None, big_font=True, ylim=None, title=None,
                         ax=None, return_data=False):
    """Every frame's spectrum as its own curve, coloured by when it was taken.

    The same data as :func:`plot_spectrogram`, read the other way round: the
    heatmap is better at *finding* the time a thing changed, this is better at
    *reading levels off* it, because a curve has a y axis in dB and a heatmap
    only has a colour. Use it once the heatmap has told you where to look.

    ``nframes`` is deliberately small here (40, against the heatmap's 200): one
    curve per frame, and past ~60 curves on one axes the early ones are buried
    whatever the colour map does. ``offset_db`` fans them out instead — a
    negative value steps each successive curve down the page, the classic
    stacked waterfall, at the cost of the shared y scale.

    ``phase="on"`` / ``"off"`` keeps only the frames whose centre falls inside
    a drive-on (or drive-off) span of ``rfsoc_pulse_spec``, with ``guard_us``
    trimmed off both ends of each span. Without it a gated run interleaves two
    completely different spectra along one colour ramp and neither is legible;
    with it you get the trend WITHIN one phase, which is usually the question.

    The colour map is sequential — time is a magnitude, so it gets a
    light-to-dark ramp and a colourbar in ``time_unit``, never a categorical
    palette. ``cmap_range`` trims its ends (the near-white end of magma, the
    bright yellow end of viridis) so that no curve disappears into the page;
    pass ``(0, 1)`` for the untrimmed map. Everything else (``rbw_khz``, ``navg``, ``relative``,
    ``span_mhz``/``center_mhz``, ``band_mhz``, ``nf_max``) means what it does
    in :func:`plot_spectrogram`."""
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    specs = C._norm_run_specs(runs, noise=noise, events=events)
    _resolve_events(specs)
    if not specs:
        print("no DDR4 capture matches")
        return
    rel = str(relative).lower()
    if rel not in ("none", "median", "mean", "first"):
        raise ValueError("relative must be 'none', 'median', 'mean' or 'first'")

    out, kred, f0 = [], 1, None
    for sp in specs:
        for m in sp["ev"]:
            f0 = lo_hz(sp["run"], lo_mhz)
            try:
                sg = spectrogram(sp["run"], m, lo=f0, rbw_hz=rbw_khz * 1e3,
                                 nframes=nframes, navg=navg, window=window,
                                 sideband=sideband, average=average, tlim=tlim,
                                 time_unit=time_unit)
            except (FileNotFoundError, ValueError) as e:
                print(f"[spectra] RID {sp['rid']} {m['npz']}: {e}")
                continue
            if phase:
                keep = _frame_phase_mask(sp["run"], sg["t"], phase, guard_us)
                if not keep.any():
                    print(f"[spectra] RID {sp['rid']} {C._event_label(m['npz'])}: "
                          f"no frame centre lands in a {phase!r} span — raise "
                          f"nframes so the frames are shorter than the spans")
                    continue
                if keep.sum() < len(keep):
                    sg["t"], sg["P"] = sg["t"][keep], sg["P"][keep]
            out.append({"rid": sp["rid"], "event": C._event_number(m["npz"]),
                        "label": f"{sp['label']} · {C._event_label(m['npz'])}"
                                 + (" [bg]" if sp["noise"] else ""),
                        "noise": sp["noise"], **sg})
    if not out:
        print("[spectra] nothing to plot")
        return

    for o in out:
        fc, pc = _crop(o["f"], o["P"].T, f0 if center_mhz is None else center_mhz * 1e6,
                       None if span_mhz is None else span_mhz * 1e6)
        fr, Pr, kred = _reduce_f(fc, np.ascontiguousarray(pc.T), nf_max, nf_reduce)
        D = _db(Pr, db_ref, dbm_offset)
        if rel in ("median", "mean"):
            D = D - (np.nanmedian(D, 0) if rel == "median" else np.nanmean(D, 0))
        elif rel == "first":
            D = D - D[0]
        o["f_plot"], o["D"] = fr, D

    ts = C._unit_scale("s", time_unit)
    tlo = min(o["t"][0] for o in out) * ts
    thi = max(o["t"][-1] for o in out) * ts
    n = len(out)
    with C._rc(big_font=big_font):
        if ax is None:
            fig, axs = plt.subplots(n, 1, sharex=True, sharey=True,
                                    constrained_layout=True, squeeze=False,
                                    figsize=figsize or (10.5, max(4.2, 3.4 * n)))
            axs = list(axs[:, 0])
        else:
            fig, axs = ax.figure, [ax] * n
        norm = plt.Normalize(tlo, thi)
        sm = plt.cm.ScalarMappable(norm=norm,
                                   cmap=_truncate_cmap(cmap, *cmap_range))
        for a, o in zip(axs, out):
            x = o["f_plot"] / 1e6
            # one LineCollection instead of nframes ax.plot() calls: 40 curves
            # of a few thousand points each is where per-line artists start to
            # cost more than the transform
            segs = [np.column_stack([x, o["D"][k] + k * offset_db])
                    for k in range(len(o["t"]))]
            lc = LineCollection(segs, colors=sm.to_rgba(o["t"] * ts),
                                linewidths=lw, alpha=alpha)
            a.add_collection(lc)
            a.set_xlim(x[0], x[-1])
            a.autoscale_view(scalex=False)
            a.grid(alpha=.3)
            unit = ("dB re its own " + rel if rel != "none"
                    else ("Power (dBm)" if dbm else "power in RBW [dB re 1 count]"))
            a.set_ylabel(unit + (" + offset" if offset_db else ""))
            a.set_title(o["label"], loc="left", pad=3.0,
                        fontsize=max(9, plt.rcParams["font.size"] - 3),
                        color="0.25")
            if band_mhz is not None:
                a.axvspan(band_mhz[0], band_mhz[1], color="0.45", alpha=.10,
                          lw=0, zorder=0.5)
                for fx in band_mhz:
                    a.axvline(fx, color="0.35", lw=1.0, ls=(0, (4, 3)), alpha=.75,
                              zorder=0.6)
            if ylim is not None:
                a.set_ylim(*ylim)
        axs[-1].set_xlabel("RF frequency [MHz]")
        fig.colorbar(sm, ax=axs, pad=0.015, aspect=max(16, 14 * n),
                     label=f"time in capture [{C._TIME_UNIT_LABEL[time_unit]}]")
        if title is None:
            nt = len(out[0]["t"])
            title = (f"Spectrum vs time — {nt} curve(s), RBW "
                     f"{out[0]['rbw']/1e3:.2f} kHz, {out[0]['dt']*1e3:.3f} ms "
                     f"per frame"
                     + (f", {out[0]['nseg']} avg/frame" if out[0]["nseg"] > 1 else "")
                     + (f", drive {phase.upper()} only" if phase else "")
                     + (f", offset {offset_db:g} dB/curve" if offset_db else ""))
        import textwrap
        fig.suptitle(textwrap.fill(title, 78))
    return (axs, out) if return_data else axs


# --------------------------------------------------------------------------- #
#  band filter  ->  back to the time domain
# --------------------------------------------------------------------------- #
def _apply_mask(Z, fs, f1, f2, taper):
    """zero everything outside ``[f1, f2]`` in place, with raised-cosine edges
    ``taper`` Hz wide (a hard rectangle rings badly once transformed back).

    Works on FFT bin indices — negative frequencies simply wrap — so nothing
    the length of the record is allocated besides ``Z`` itself."""
    n = Z.size
    k = lambda f: int(np.round(f * n / fs))
    k1, k2 = k(f1), k(f2)
    kt = max(0, int(round(taper * n / fs)))
    a, b = (k1 - kt) % n, (k2 + kt) % n          # first/last bin kept
    if a <= b:
        Z[:a] = 0
        Z[b + 1:] = 0
    else:                                        # the kept band wraps around
        Z[b + 1:a] = 0
    if kt:
        x = np.arange(1, kt + 1, dtype=np.float32) / (kt + 1)
        ramp = (0.5 * (1 - np.cos(np.pi * x))).astype(np.float32)
        Z[np.arange(k1 - kt, k1) % n] *= ramp
        Z[np.arange(k2 + 1, k2 + kt + 1) % n] *= ramp[::-1]
    return Z


def band_filter(z, fs, band_hz, lo=0.0, sideband=+1, taper_frac=0.05,
                workers=-1, overwrite=False):
    """Zero everything outside ``band_hz`` (an RF ``(f1, f2)``) and transform
    back: ``ifft(fft(z) · mask)``.

    The mask is built on the baseband grid the RF band maps to, with
    raised-cosine edges ``taper_frac`` of the bandwidth wide so the filtered
    time series does not ring. Returns the complex band-limited record, which
    may be a few thousand samples shorter than ``z`` (the length is trimmed to
    one the FFT likes — see the note in the code)."""
    from scipy import fft as sfft
    f1 = (float(band_hz[0]) - lo) * (1 if sideband > 0 else -1)
    f2 = (float(band_hz[1]) - lo) * (1 if sideband > 0 else -1)
    f1, f2 = min(f1, f2), max(f1, f2)
    if f2 <= -fs / 2 or f1 >= fs / 2:
        raise ValueError(f"band {np.array(band_hz)/1e6} MHz is outside "
                         f"{(lo - fs/2)/1e6:g}–{(lo + fs/2)/1e6:g} MHz")
    # a capture length with a large prime factor sends pocketfft down the
    # Bluestein path: several times the memory and the time for the same
    # answer. Trimming the last few thousand samples to an FFT-friendly
    # length costs microseconds of record and avoids all of it.
    n = len(z)
    try:
        nfast = int(sfft.prev_fast_len(n))
    except AttributeError:                     # scipy < 1.15
        nfast = n
        while nfast > 16 and int(sfft.next_fast_len(nfast)) != nfast:
            nfast -= 1
    if nfast < n:
        if (n - nfast) / n > 1e-3:
            print(f"[band] trimming {n - nfast} of {n} samples "
                  f"({(n-nfast)/n*100:.2f} %) for an FFT-friendly length")
        z = z[:nfast]
    Z = sfft.fft(z, workers=workers, overwrite_x=overwrite)
    _apply_mask(Z, fs, f1, f2, taper_frac * max(f2 - f1, 1.0))
    return sfft.ifft(Z, overwrite_x=True, workers=workers)


def plot_band_filtered(runs, band_mhz, noise=None, events=None, lo_mhz=None,
                       sideband=+1, taper_frac=0.05, tlim=None, time_unit="ms",
                       n_points=4000, decimation=None, envelope=True, db=False,
                       db_ref=1.0, dbm_offset=None, dbm=True, show_raw=False,
                       split=True, figsize=None, ylim=None, lw=1.2,
                       big_font=True, cmap=None, color_by="auto",
                       noise_color=NOISE_COLOR, noise_lw=None, noise_alpha=None,
                       noise_on_top=False, noise_average=True,
                       noise_stat="mean", noise_events=None,
                       ax=None, legend_loc="best", title=None,
                       return_data=False):
    """Keep only ``band_mhz=(f1, f2)`` of the RF spectrum, transform back, and
    plot the time-domain magnitude ``|v(t)|`` of what is left.

    This is the companion to :func:`plot_rf_spectrum`: pick the band there,
    filter with it here. Out-of-band noise — which is most of the 552.96 MHz
    the capture covers — disappears, so a burst inside the band stands out
    much further above the floor than it does in the raw ``|I,Q|``.

    Only what survives the band is drawn. ``show_raw=True`` adds the
    unfiltered ``|I,Q|`` back in grey on a twin right axis, for scale — off by
    default, because that trace is an order of magnitude above the in-band one,
    needs an axis of its own and clutters the panel the filter was meant to
    clean up. ``split=True`` (default) gives each capture its own panel, sized
    like the zoom figure of :func:`sa_rfsoc_corr.plot_ddr4_events_bgsub` — one
    panel per event, each autoscaled to itself; ``split=False`` overlays them
    on one axes.

    ``noise=<rid>`` is a REFERENCE, not one more event: by default every
    capture of the background run is filtered through the same band and
    collapsed into ONE curve (``noise_stat="mean"``, ``"median"`` for the
    robust version) with a grey ±1σ band around it, and that same curve is
    repeated in EVERY panel, behind the signal. Otherwise each background
    capture becomes its own panel and the "background" changes from event to
    event, which makes the panels impossible to compare — the whole point of
    the background is that it is the same level in all of them.
    ``noise_average=False`` goes back to one panel per background capture.

    Display sampling (``n_points``/``decimation``/``envelope``) works as in the
    other DDR4 plots: the filtered magnitude is binned to a min–max envelope
    with the mean drawn on top, so a 55 M-sample record renders honestly."""
    import matplotlib.pyplot as plt
    specs = C._norm_run_specs(runs, noise=noise, events=events)
    _resolve_events(specs, noise_events, what="band")
    if not specs:
        print("no DDR4 capture matches")
        return
    for sp in specs:
        if sp["noise"] and len(sp["ev"]) > 3 and noise_events is None:
            print(f"[band] RID {sp['rid']}: filtering {len(sp['ev'])} background "
                  f"capture(s) — pass noise_events=[…] to use fewer")
    sig = [sp for sp in specs if not sp["noise"]]
    per_event = (color_by == "event" or (color_by == "auto" and len(sig) <= 1))
    if per_event:
        pal = _sig_colors(sum(len(sp["ev"]) for sp in sig), cmap)
        i = 0
        for sp in sig:
            sp["colors"] = pal[i:i + len(sp["ev"])]
            i += len(sp["ev"])
    else:
        for sp, col in zip(sig, _sig_colors(len(sig), cmap)):
            sp["colors"] = C._shades(sp["color"] or col, len(sp["ev"]))
    for sp in specs:
        if sp["noise"]:
            sp["colors"] = _bg_colors(noise_color, max(1, len(sp["ev"])))

    def _y(a):
        if not db:
            return np.asarray(a)
        with np.errstate(divide="ignore", invalid="ignore"):
            return (20 * np.log10(np.where(np.asarray(a) > 0, a, np.nan) / db_ref)
                    + _offset(dbm_offset))      # |v|^2 scales like the spectrum

    s_to_disp = 1e6 * C._unit_scale("us", time_unit)     # seconds -> display unit
    band_hz = (float(band_mhz[0]) * 1e6, float(band_mhz[1]) * 1e6)

    traces, bg_stack, bg_t, bg_runs = [], [], None, []
    for sp in specs:
        f0 = lo_hz(sp["run"], lo_mhz)
        ref_only = sp["noise"] and noise_average   # folded into the reference
        if ref_only:
            bg_runs.append(sp["label"])
        for m, col in zip(sp["ev"], sp["colors"]):
            try:
                zs, fs, t0s = load_iq_windows(sp["run"], m, tlim=tlim,
                                              time_unit=time_unit)
            except (FileNotFoundError, ValueError) as e:
                print(f"[band] RID {sp['rid']} {m['npz']}: {e}")
                continue
            # share the point budget across the windows, then stitch the
            # envelopes with a NaN so the line breaks over the skipped time
            npw = max(200, n_points // len(zs))
            rawp, tp, lop, hip, mp = [], [], [], [], []
            ssq, nsamp, peak = 0.0, 0, 0.0
            for z, t0 in zip(zs, t0s):
                if show_raw and not ref_only:   # bin it first, then free the copy
                    raw = np.abs(z)
                    rawp.append(_envelope_fs(raw, fs, t0, npw, step=decimation))
                    del raw
                y = np.abs(band_filter(z, fs, band_hz, lo=f0, sideband=sideband,
                                       taper_frac=taper_frac, overwrite=True))
                e = _envelope_fs(y, fs, t0, npw, step=decimation)
                tp.append(e[0]); lop.append(e[1]); hip.append(e[2]); mp.append(e[3])
                ssq += float(np.sum(y.astype(np.float64) ** 2))
                nsamp += y.size
                peak = max(peak, float(y.max()))
                del y
            del zs
            rawb = None if not rawp else tuple(
                _stitch([r[i] for r in rawp]) for i in range(4))
            tb, lo_e, hi_e, mean = (_stitch(tp), _stitch(lop),
                                    _stitch(hip), _stitch(mp))
            if ref_only:
                # one reference for the whole background run: keep this
                # capture's mean, average across captures once they are all in
                if bg_t is None:
                    bg_t = tb
                bg_stack.append(mean if len(tb) == len(bg_t)
                                else np.interp(bg_t, tb, mean))
                continue
            traces.append({"rid": sp["rid"], "event": C._event_number(m["npz"]),
                           "label": f"{sp['label']} · {C._event_label(m['npz'])}"
                                    + (" [bg]" if sp["noise"] else ""),
                           "color": col, "noise": sp["noise"],
                           "t": tb * s_to_disp, "lo": lo_e, "hi": hi_e,
                           "mean": mean, "rms": float(np.sqrt(ssq / nsamp)),
                           "peak": peak,
                           "raw": None if rawb is None
                                  else (rawb[0] * s_to_disp, rawb[3], rawb[2])})

    bgref = None
    if bg_stack:
        rows = np.vstack(bg_stack)
        mid = (np.nanmedian(rows, 0) if str(noise_stat).startswith("med")
               else np.nanmean(rows, 0))
        bgref = {"t": bg_t * s_to_disp, "y": mid, "sd": np.nanstd(rows, 0),
                 "n": int(rows.shape[0]), "stat": noise_stat,
                 "label": f"{' + '.join(dict.fromkeys(bg_runs))} · {noise_stat}"
                          f" ± 1σ of {rows.shape[0]} capture(s) [bg]"}
    if not traces:
        if bgref is None:
            print("[band] nothing to plot")
            return
        # only background runs were given — draw the reference as the trace
        traces.append({"rid": None, "event": None, "label": bgref["label"],
                       "color": noise_color, "noise": True, "t": bgref["t"],
                       "lo": bgref["y"] - bgref["sd"],
                       "hi": bgref["y"] + bgref["sd"], "mean": bgref["y"],
                       "rms": float(np.sqrt(np.nanmean(bgref["y"] ** 2))),
                       "peak": float(np.nanmax(bgref["y"])), "raw": None})
        bgref = None

    n = len(traces)
    with C._rc(big_font=big_font):
        if ax is not None:
            axes = [ax] * n
            fig = ax.figure
        elif split and n > 1:
            fig, axs = plt.subplots(n, 1, sharex=True, sharey=False,
                                    constrained_layout=True,
                                    figsize=figsize or (10.5, max(2.6, 1.55 * n)))
            axes = list(np.atleast_1d(axs))
        else:
            fig, a0 = plt.subplots(constrained_layout=True,
                                   figsize=figsize or (10.5, 5.8))
            axes = [a0] * n
        one = len({id(a) for a in axes}) == 1
        raw_ax = []
        for i, (a, tr) in enumerate(zip(axes, traces)):
            if bgref is not None and (i == 0 or not one):
                # the SAME reference in every panel, drawn first so the signal
                # sits on top of it: a background that changes from panel to
                # panel is not a background, it is another event
                lo_b = bgref["y"] - bgref["sd"]
                if db:            # a negative edge has no dB — floor the band
                    lo_b = np.maximum(lo_b, bgref["y"] * 1e-2)
                a.fill_between(bgref["t"], _y(lo_b), _y(bgref["y"] + bgref["sd"]),
                               color=NOISE_BAND_COLOR,
                               alpha=.18 if noise_on_top else .30, lw=0,
                               zorder=3.3 if noise_on_top else 1.4)
                a.plot(bgref["t"], _y(bgref["y"]), color=noise_color,
                       lw=noise_lw or lw * 1.35, ls=(0, (5, 2.5)),
                       alpha=_noise_alpha(noise_alpha, noise_on_top),
                       zorder=3.4 if noise_on_top else 1.6,
                       label=bgref["label"] if i == 0 else None)
            if tr["raw"] is not None and (i == 0 or not one):
                # the raw level is ~15x the in-band one here, so it goes on its
                # own right-hand axis instead of flattening the filtered trace
                rt, rmean, rhi = tr["raw"]
                axr = a.twinx()
                axr.grid(False)
                axr.plot(rt, _y(rmean), color="0.72", lw=max(0.8, lw * 0.8),
                         zorder=1.2,
                         label="raw |I,Q| (unfiltered, right axis)" if i == 0 else None)
                axr.tick_params(axis="y", colors="0.55",
                                labelsize=max(8, plt.rcParams["ytick.labelsize"] - 3))
                axr.yaxis.set_major_locator(plt.MaxNLocator(3))
                raw_ax.append(axr)
            if envelope:
                a.fill_between(tr["t"], _y(tr["lo"]), _y(tr["hi"]),
                               color=tr["color"], alpha=.22, lw=0, zorder=2)
            a.plot(tr["t"], _y(tr["mean"]),
                   color=tr["color"],
                   lw=(noise_lw or lw * 1.35) if tr["noise"] else lw,
                   ls=(0, (5, 2.5)) if tr["noise"] else "-",
                   alpha=(_noise_alpha(noise_alpha, noise_on_top)
                          if tr["noise"] else 1.0),
                   zorder=(3.4 if noise_on_top else 3.2) if tr["noise"] else 2.5,
                   label=tr["label"] if one else None)
            if not one:
                a.text(0.988, 0.94, tr["label"], transform=a.transAxes,
                       ha="right", va="top", color=tr["color"], fontweight="bold",
                       fontsize=max(9, plt.rcParams["legend.fontsize"] - 2),
                       bbox=dict(boxstyle="round,pad=0.22", fc="white",
                                 ec="0.8", alpha=.85))
                a.yaxis.set_major_locator(plt.MaxNLocator(4))
            if ylim is not None:
                a.set_ylim(*ylim)
        for a in (axes[:1] if one else axes):
            a.grid(True)
        h, l = axes[0].get_legend_handles_labels()
        if raw_ax:
            h2, l2 = raw_ax[0].get_legend_handles_labels()
            h, l = h + h2, l + l2
        if h:
            fs_leg = max(9, plt.rcParams["legend.fontsize"] - 1)
            if one and legend_loc == "best":
                axes[0].legend(h, l, loc="upper center", bbox_to_anchor=(0.5, -0.16),
                               ncol=min(3, len(h)), frameon=False, fontsize=fs_leg)
            else:
                axes[0].legend(h, l, loc="upper left", fontsize=fs_leg, framealpha=.85)
        if title is None:
            title = (f"band {band_mhz[0]:g}–{band_mhz[1]:g} MHz kept "
                     f"({(band_hz[1]-band_hz[0])/1e3:.0f} kHz wide), "
                     f"back to the time domain")
        import textwrap
        (fig.suptitle if not one else axes[0].set_title)(textwrap.fill(title, 78),
                                                        fontsize=plt.rcParams["axes.titlesize"])
        axes[-1].set_xlabel(f"time in capture window [{C._TIME_UNIT_LABEL[time_unit]}]")
        ylab = (("|v| in band [dBm]" if dbm else "20·log10|v| [dB re 1 count]")
                if db else "In-band voltage magnitude (code)")
        if one:
            axes[0].set_ylabel(ylab)
        else:
            try:
                fig.supylabel(ylab)
            except AttributeError:
                axes[len(axes) // 2].set_ylabel(ylab)
    return (axes[0], {"traces": traces, "band_hz": band_hz, "background": bgref,
                      "axes": axes, "fig": fig}) if return_data else axes[0]


def analyze(rid, band_mhz=None, events=None, noise=None, lo_mhz=None,
            rbw_khz=10.0, span_mhz=None, center_mhz=None, window="hann",
            sideband=+1, average="mean", dbm_offset=None, dbm=True, db_ref=1.0,
            tlim=None, time_unit="ms", taper_frac=0.05, n_points=4000,
            decimation=None, db=False, show_raw=False, split=True,
            big_font=True, lw=1.2, cmap=None, color_by="auto",
            noise_color=NOISE_COLOR, noise_lw=None, noise_alpha=None,
            noise_on_top=False, noise_average=True, noise_stat="mean",
            noise_events=None,
            spectrum_ylim=None, band_ylim=None, figsize=None,
            band_figsize=None, save=None, prefix=None, dpi=150, quiet=False):
    """**The one call that does everything** — spectrum, band filter, summary.

    Paste-ready notebook example::

        import sys
        sys.path.insert(0, "/home/electron/edes/edes/experiments/analysis")
        import rfsoc_spectrum as S

        out = S.analyze(11345,                    # RID (or .h5 path, or a list)
                        band_mhz=(175.3, 175.5),  # band to keep; omit -> spectrum only
                        rbw_khz=10,               # resolution bandwidth
                        span_mhz=4,               # plotted span around the LO
                        time_unit="ms")

        out["spectra"][0]["f"], out["spectra"][0]["p"]   # RF axis [Hz], power
        out["traces"][0]["t"],  out["traces"][0]["mean"] # filtered |v(t)|

    More of the same::

        # only certain captures: one, a few, or a different set per run
        S.analyze(11345, events=1)                      # just capture 1
        S.analyze(11344, events=[1, 3, 5])              # a handful
        S.analyze([(11344, [1, 2]), (11345, [1])])      # per run
        S.analyze([11344, 11345], events=[1])           # capture 1 of each

        # several runs, one capture each, plus the background spectrum of
        # 11338 averaged over all of ITS captures, in grey on top
        S.analyze([11344, 11345], noise=11338, events=[1],
                  band_mhz=(175.3, 175.5), span_mhz=6, rbw_khz=20)

        # dBm: pin the scale once against the spectrum analyser
        S.set_dbm_calibration(peak_dbm=-52.0, peak_db_counts=-7.8)
        S.analyze(11345, span_mhz=4)                    # y axis now true dBm

        # zoom onto 16.0-16.5 ms of the capture and save the figures
        S.analyze(11345, tlim=(16.0, 16.5), time_unit="ms", rbw_khz=2,
                  span_mhz=1, band_mhz=(175.35, 175.45), save="~/plots")

        # conjugated DDC (spectrum mirrored about the LO), explicit LO
        S.analyze(11345, lo_mhz=175.7, sideband=-1, span_mhz=4)

    Arguments
    ---------
    ``rid``             an RID, a ``.h5`` path, a loaded run, or a list of them
                        (also ``(rid, events)`` tuples / spec dicts, as in
                        :func:`sa_rfsoc_corr.plot_ddr4_events_multi`).
    ``band_mhz``        ``(f1, f2)`` in MHz. Given, it is shaded on the spectrum
                        AND produces the second figure — the band-filtered
                        ``|v(t)|``. Left out, only the spectrum is drawn.
    ``events``          capture numbers to use (default: every capture).
    ``noise``           RID(s) whose captures give the background. On the
                        spectrum they are averaged (in power) into ONE black
                        dashed trace over every capture of that run —
                        ``noise_events=[…]`` to restrict which,
                        ``noise_stat="median"`` for the robust average,
                        ``noise_average=False`` for one trace per capture.
                        The run need not appear in ``rid``.
    ``events``          which captures to plot: an int, a list, or per run via
                        ``(rid, [events])`` tuples. Default: all of them.
    ``rbw_khz``         resolution bandwidth. ``span_mhz``/``center_mhz`` set
                        the plotted window; leave ``span_mhz`` out and the
                        WHOLE span the sample rate covers is shown, LO ± fs/2
                        (≈ -100.8 … 452.1 MHz here), with the x limits pinned
                        to the data.
    ``dbm``             y axis in dBm (default), offset by :data:`DBM_CAL_DB`
                        or an explicit ``dbm_offset``; ``dbm=False`` labels it
                        dB re 1 count.
    ``noise_on_top``    draw the background curve IN FRONT of the signal
                        traces instead of behind them — what you want when the
                        two nearly coincide and the background is buried.
                        ``noise_alpha`` then defaults to 0.70 so the traces
                        underneath still read through it; set it yourself to
                        override (1.0 = opaque).
    ``noise_color`` / ``noise_lw`` / ``noise_alpha``
                        how loud the background spectra are (they are grey,
                        dashed and drawn on top by default). The signal curves
                        keep the coloured palette, one colour each.
    ``lo_mhz``          override the LO; by default the run's own ``read_freq``
                        (175.7 MHz on this bench). ``sideband=-1`` mirrors the
                        RF mapping for a conjugated DDC.
    ``tlim``            analyse only this slice of the capture, in ``time_unit``:
                        one ``(t0, t1)``, or several disjoint windows
                        ``[(a0, a1), (b0, b1), ...]`` pooled into one spectrum.
    ``save``            a directory: writes ``rf_spectrum_<tag>.png`` and
                        ``band_<f1>-<f2>MHz_<tag>.png`` (``prefix``/``dpi``).
    ``quiet``           suppress the printed summary.

    Everything else is passed through to :func:`plot_rf_spectrum` and
    :func:`plot_band_filtered`.

    Returns a dict with ``ax_spectrum``/``fig_spectrum``, ``ax_band``/
    ``fig_band``, the ``spectra`` and ``traces`` data, ``rbw``, ``nseg`` and
    the list of files ``saved``."""
    out = {"ax_spectrum": None, "fig_spectrum": None, "ax_band": None,
           "fig_band": None, "spectra": [], "traces": [], "rbw": None,
           "nseg": None, "band_hz": None, "band_background": None, "saved": []}
    tag = prefix or _tag(rid)

    r = plot_rf_spectrum(rid, noise=noise, events=events, lo_mhz=lo_mhz,
                         rbw_khz=rbw_khz, span_mhz=span_mhz,
                         center_mhz=center_mhz, window=window, sideband=sideband,
                         average=average, tlim=tlim, time_unit=time_unit,
                         db_ref=db_ref, dbm_offset=dbm_offset, dbm=dbm,
                         band_mhz=band_mhz, figsize=figsize, ylim=spectrum_ylim,
                         lw=lw, big_font=big_font, cmap=cmap, color_by=color_by,
                         noise_color=noise_color, noise_lw=noise_lw,
                         noise_alpha=noise_alpha, noise_on_top=noise_on_top,
                         noise_average=noise_average,
                         noise_stat=noise_stat, noise_events=noise_events,
                         return_data=True)
    if r is None:
        return out
    ax, d = r
    out.update(ax_spectrum=ax, fig_spectrum=ax.figure, spectra=d["spectra"],
               rbw=d["rbw"], nseg=d["nseg"])
    if not quiet:
        nw = len(_norm_windows(tlim) or [1])
        print(f"\nRF spectrum — RBW {d['rbw']/1e3:.2f} kHz, {d['nseg']} averages"
              + (f" over {nw} time windows" if nw > 1 else ""))
        for s in d["spectra"]:
            p_db = _db(s["p"], db_ref, dbm_offset)
            k = int(np.nanargmax(p_db))
            u = "dBm" if dbm else "dB"
            print(f"  {s['label']:28s} peak {s['f'][k]/1e6:10.4f} MHz "
                  f"{p_db[k]:7.1f} {u}   floor {np.nanmedian(p_db):7.1f} {u}")

    if band_mhz is not None:
        r2 = plot_band_filtered(rid, band_mhz, noise=noise, events=events,
                                lo_mhz=lo_mhz, sideband=sideband,
                                taper_frac=taper_frac, tlim=tlim,
                                time_unit=time_unit, n_points=n_points,
                                decimation=decimation, envelope=True, db=db,
                                db_ref=db_ref, dbm_offset=dbm_offset, dbm=dbm,
                                show_raw=show_raw, split=split,
                                figsize=band_figsize, ylim=band_ylim, lw=lw,
                                big_font=big_font, cmap=cmap, color_by=color_by,
                                noise_color=noise_color, noise_lw=noise_lw,
                                noise_alpha=noise_alpha,
                                noise_on_top=noise_on_top,
                                noise_average=noise_average,
                                noise_stat=noise_stat, noise_events=noise_events,
                                return_data=True)
        if r2 is not None:
            ax2, d2 = r2
            out.update(ax_band=ax2, fig_band=d2["fig"], traces=d2["traces"],
                       band_hz=d2["band_hz"], band_background=d2["background"])
            if not quiet:
                print(f"\nBand {band_mhz[0]:g}–{band_mhz[1]:g} MHz kept "
                      f"({(band_mhz[1]-band_mhz[0])*1e3:.0f} kHz wide)")
                b = d2["background"]
                if b is not None:
                    print(f"  {b['label']:28s} level {np.nanmean(b['y']):8.3f}"
                          f" ± {np.nanmean(b['sd']):.3f}   (counts)")
                for t in d2["traces"]:
                    print(f"  {t['label']:28s} rms {t['rms']:8.3f}   "
                          f"peak {t['peak']:8.2f}   (counts)")

    if save:
        d_out = os.path.expanduser(str(save))
        os.makedirs(d_out, exist_ok=True)
        for fig, name in ((out["fig_spectrum"], f"rf_spectrum_{tag}.png"),
                          (out["fig_band"], None if band_mhz is None else
                           f"band_{band_mhz[0]:g}-{band_mhz[1]:g}MHz_{tag}.png")):
            if fig is not None and name:
                path = os.path.join(d_out, name)
                fig.savefig(path, dpi=dpi)
                out["saved"].append(path)
                if not quiet:
                    print(f"\nsaved {path}")
    return out


def _tag(rid):
    """file-name tag for whatever was passed as ``rid``"""
    items = rid if isinstance(rid, (list, tuple)) else [rid]
    parts = []
    for it in items:
        if C._is_run(it):
            parts.append(str(it["rid"]))
        elif isinstance(it, tuple) and it:
            parts.append(str(it[0]))
        elif isinstance(it, dict):
            parts.append(str(it.get("rid", it.get("run", "run"))))
        else:
            parts.append(os.path.splitext(os.path.basename(str(it)))[0])
    return "_".join(parts) or "run"


def analyze_band(runs, band_mhz, **kw):
    """Back-compatible wrapper around :func:`analyze`; returns
    ``(ax_spectrum, ax_band)``."""
    out = analyze(runs, band_mhz=band_mhz, **kw)
    return out["ax_spectrum"], out["ax_band"]


# --------------------------------------------------------------------------- #
#  command line:  python rfsoc_spectrum.py <rid> [...]
# --------------------------------------------------------------------------- #
_EXAMPLES = """\
examples:
  # RF spectrum of every capture in run 11345, 10 kHz RBW, 4 MHz span
  python rfsoc_spectrum.py 11345 --rbw 10 --span 4

  # ... plus the band-filtered |v(t)| for 175.3-175.5 MHz
  python rfsoc_spectrum.py 11345 --rbw 10 --span 4 --band 175.3 175.5

  # one event of two runs, with a noise run as the grey background spectrum
  python rfsoc_spectrum.py 11344 11345 --noise 11338 --events 1 \\
      --span 6 --band 175.3 175.5 --outdir ~/plots

  # zoom the analysis onto 16.0-16.5 ms of the capture, show instead of save
  python rfsoc_spectrum.py 11345 --tlim 16.0 16.5 --time-unit ms --rbw 2 \\
      --span 1 --show
"""


def _rid(s):
    """an RID if it looks like one, else a path to the results .h5"""
    return int(s) if str(s).isdigit() else s


def _evlist(s):
    """``--events 1 3-5 8`` -> [1, 3, 4, 5, 8]; also accepts ``1,2``"""
    out = []
    for part in str(s).replace(",", " ").split():
        if "-" in part.strip("-"):
            a, b = part.split("-", 1)
            out.extend(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return out


def _parse_args(argv=None):
    import argparse
    p = argparse.ArgumentParser(
        prog="rfsoc_spectrum.py", formatter_class=argparse.RawDescriptionHelpFormatter,
        description="RF spectrum of the RFSoC DDR4 I/Q captures, and the "
                    "band-filtered magnitude back in the time domain.",
        epilog=_EXAMPLES)
    p.add_argument("rid", nargs="+", type=_rid,
                   help="RID(s) or path(s) to the results .h5")
    p.add_argument("--events", nargs="+", type=_evlist, default=None,
                   help="capture numbers to plot, e.g. --events 1 3-5 "
                        "(default: every capture in the run)")
    p.add_argument("--noise", nargs="+", type=_rid, default=None,
                   help="RID(s) for the background spectrum — averaged over all "
                        "their captures into one black dashed trace")
    p.add_argument("--noise-events", nargs="+", type=_evlist, default=None,
                   help="restrict which captures of the noise run(s) are averaged")
    p.add_argument("--noise-stat", default="mean", choices=("mean", "median"),
                   help="how the background captures are combined (default: mean)")
    p.add_argument("--no-noise-average", action="store_true",
                   help="draw every background capture instead of their average")

    g = p.add_argument_group("frequency")
    g.add_argument("--lo", type=float, default=None, metavar="MHz",
                   help="down-conversion LO (default: the run's read_freq, "
                        f"else {LO_MHZ_DEFAULT} MHz)")
    g.add_argument("--rbw", type=float, default=10.0, metavar="kHz",
                   help="resolution bandwidth (default: 10)")
    g.add_argument("--span", type=float, default=None, metavar="MHz",
                   help="plotted span (default: the whole sample rate)")
    g.add_argument("--center", type=float, default=None, metavar="MHz",
                   help="centre of the span (default: the LO)")
    g.add_argument("--window", default="hann", help="FFT window (default: hann)")
    g.add_argument("--sideband", type=int, choices=(1, -1), default=1,
                   help="+1: f_RF = LO + f_baseband (default); -1 for a "
                        "conjugated DDC")
    g.add_argument("--dbm-offset", type=float, default=None, metavar="dB",
                   help="dB added to 10*log10(counts^2) to get dBm "
                        "(default: the module's DBM_CAL_DB)")
    g.add_argument("--no-dbm", action="store_true",
                   help="label the axis 'dB re 1 count' instead of dBm")

    g = p.add_argument_group("band filter (the second figure)")
    g.add_argument("--band", nargs=2, type=float, default=None,
                   metavar=("F1", "F2"),
                   help="RF band to keep, in MHz — without it only the "
                        "spectrum is produced")
    g.add_argument("--taper", type=float, default=0.05, metavar="FRAC",
                   help="raised-cosine band-edge width, as a fraction of the "
                        "bandwidth (default: 0.05)")
    g.add_argument("--n-points", type=int, default=4000,
                   help="display points per trace (default: 4000)")
    g.add_argument("--decimation", type=int, default=None,
                   help="fixed samples per display bin (overrides --n-points)")
    g.add_argument("--db", action="store_true",
                   help="plot 20*log10|v| instead of counts")
    g.add_argument("--raw", action="store_true",
                   help="also draw the unfiltered |I,Q| on a twin right axis")
    g.add_argument("--no-split", action="store_true",
                   help="overlay the captures on one axes instead of one panel each")

    g = p.add_argument_group("window / output")
    g.add_argument("--tlim", nargs="+", type=float, default=None,
                   metavar="T",
                   help="analyse only this slice of the capture (in "
                        "--time-unit). Give several PAIRS -- T0 T1 T2 T3 ... -- "
                        "to use several disjoint windows; their segments are "
                        "pooled into one spectrum")
    g.add_argument("--time-unit", default="ms", choices=("ns", "us", "ms", "s"),
                   help="unit for --tlim and the time axis (default: ms)")
    g.add_argument("--outdir", default=".", help="where to save the PNGs (default: .)")
    g.add_argument("--prefix", default=None, help="file name prefix (default: from the RIDs)")
    g.add_argument("--dpi", type=int, default=150)
    g.add_argument("--show", action="store_true", help="open the figures instead of only saving")
    g.add_argument("--no-save", action="store_true", help="do not write PNGs")
    g.add_argument("--no-big-font", action="store_true",
                   help="use the compact module style instead of big_plt_font")
    return p.parse_args(argv)


def main(argv=None):
    """Entry point for ``python rfsoc_spectrum.py …`` — a thin wrapper around
    :func:`analyze`, which is the same thing callable from a notebook."""
    a = _parse_args(argv)
    import matplotlib
    if not a.show:
        matplotlib.use("Agg")                  # headless: only write files
    import matplotlib.pyplot as plt

    events = None if a.events is None else [e for grp in a.events for e in grp]
    out = analyze(a.rid if len(a.rid) > 1 else a.rid[0],
                  band_mhz=a.band, events=events, noise=a.noise, lo_mhz=a.lo,
                  rbw_khz=a.rbw, span_mhz=a.span, center_mhz=a.center,
                  window=a.window, sideband=a.sideband, dbm_offset=a.dbm_offset,
                  dbm=not a.no_dbm,
                  tlim=_cli_tlim(a.tlim), time_unit=a.time_unit,
                  taper_frac=a.taper,
                  n_points=a.n_points, decimation=a.decimation, db=a.db,
                  show_raw=a.raw, split=not a.no_split,
                  noise_average=not a.no_noise_average, noise_stat=a.noise_stat,
                  noise_events=(None if a.noise_events is None
                                else [e for g in a.noise_events for e in g]),
                  big_font=not a.no_big_font, prefix=a.prefix, dpi=a.dpi,
                  save=None if a.no_save else a.outdir)
    if out["ax_spectrum"] is None:
        return 1
    if a.show:
        plt.show()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
