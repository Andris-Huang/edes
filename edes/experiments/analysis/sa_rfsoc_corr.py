"""
sa_rfsoc_corr.py
================
Put the spectrum-analyser (SA) traces from a ``FeedbackWithRFSoC`` ARTIQ run and
the on-FPGA RFSoC standby-detector results on ONE wall-clock timeline.

Usage (notebook)::

    import sa_rfsoc_corr as C
    run = C.load_run(11109)          # or a path to the .h5
    C.summary(run)
    C.plot_timeline(run)                       # everything on one wall-clock axis
    C.plot_sa_traces(run)
    C.plot_ddr4_events(run)                    # mean = solid, min/max = band
    C.plot_ddr4_events_overlay(run)            # every event on ONE axes,
                                               # colour-coded + labelled 'event N'
    C.plot_ddr4_events_multi([11333, 11330], noise=11330)   # several RIDs on ONE
                                               # axes; noise runs grey + dashed
    C.plot_ddr4_events_bgsub([11333, 11330], noise=11330)   # same, but each event
                                               # MINUS the mean of the noise run's
                                               # captures (subtract='ratio', db=True
                                               # -> plain dB above background)
    C.plot_sa_vs_ddr4(run, sa=0, event=-1)    # ONE SA trace + ONE DDR4 window on a
                                              # shared ms axis. The SA's within-sweep
                                              # timing lags ~10-40 ms (free-run sweep +
                                              # *OPC?/transfer) -- pass sa_latency_ms= to
                                              # compensate; the RFSoC crossing is the
                                              # reliable t=0.

Clocks: the ARTIQ host (electron-ubuntu) and the RFSoC board are both NTP-synced,
so every wall-clock value below is directly comparable to ~10 ms.

  RFSoC crossings   ->  wall = board t_start + t_rel_s     (t_rel_s = iter / pps;
                        absolute SCALE carries the ~0.1-0.5 % error of the
                        startup pps calibration -- a captured crossing's manifest
                        t_wall is exact and is used instead when available)
  RFSoC DDR4/MR     ->  wall = manifest 't_wall'           (exact, host-measured
                        at detection, lags the true crossing by <~ a few ms)
  SA saved trace    ->  wall = dataset 'sa_wall_<rep>[k]'  (exact, added 2026-09)
                        fallback for older runs: an estimate stretched to fill
                        the measured ARTIQ run duration (flagged, ~+/- 10 s)
"""
from __future__ import annotations

import glob
import json
import os
import re

import numpy as np

RESULTS_ROOT = os.environ.get(
    "ARTIQ_RESULTS", "/home/electron/artiq/experiment/artiq-master/results")
RFSOC_DATA_ROOT = os.environ.get("RFSOC_DATA", "/home/electron/data/RFSoc_data")
# for reconstructing the feedback-pulse envelope from rfsoc_pulse_spec --
# pulse_sequence.py (PulseSpec) lives in rfsoc_board_only, not image_current itself
ARTIQ_MODULES = os.environ.get(
    "ARTIQ_MODULES",
    "/home/electron/artiq/experiment/artiq-master/repository/"
    "experiment_sequences/image_current/rfsoc_board_only")
F_OUT_MHZ = 552.96          # DDR4 decimated rate; MR snapshots carry their own fs

_TIME_UNIT_TO_US = {"ns": 1e-3, "us": 1.0, "ms": 1e3, "s": 1e6}
_TIME_UNIT_LABEL = {"ns": "ns", "us": "µs", "ms": "ms", "s": "s"}


def _unit_scale(from_unit, to_unit):
    """multiply a value in ``from_unit`` by this to get it in ``to_unit``"""
    return _TIME_UNIT_TO_US[from_unit] / _TIME_UNIT_TO_US[to_unit]


def _parse_style(linestyle, marker):
    """let ``linestyle`` accept a combined matplotlib fmt shorthand (e.g.
    ``".-"``, ``"o--"``) when ``marker`` isn't given separately."""
    if marker is None and linestyle:
        try:
            from matplotlib.axes._base import _process_plot_format
            ls, mk, _ = _process_plot_format(linestyle)
            if mk != "None":
                return ls, mk
        except Exception:
            pass
    return linestyle, marker


def _pulse_waveform(run, n=6000, offset_us=None):
    """(t_us, drive_frac) of the programmed feedback pulse on the DDR4 capture
    axis (t=0 = crossing), from ``rfsoc_pulse_spec``. Uses
    ``pulse_sequence.PulseSpec`` if importable. ``None`` when there is no pulse
    (voltage 0 / no spec). ``offset_us`` overrides the crossing->RF latency."""
    spec = run["rfsoc"].get("rfsoc_pulse_spec")
    if not isinstance(spec, dict):
        return None
    if float(spec.get("pulse_voltage", 0) or 0) <= 0:
        return None
    st = run["rfsoc"].get("rfsoc_status") or {}
    off = float(st.get("crossing_to_pulse_us", 4.6) if offset_us is None else offset_us)
    try:
        import sys
        if ARTIQ_MODULES not in sys.path:
            sys.path.insert(0, ARTIQ_MODULES)
        from pulse_sequence import PulseSpec
        ps = PulseSpec.from_config_fields(spec)
        t, g = ps.waveform(n=n, pad_frac=0.0)
        return np.asarray(t) + off, np.asarray(g)
    except Exception as e:
        print("[pulse] PulseSpec unavailable (%r) -- falling back to a flat model" % e)
        v = float(spec.get("pulse_voltage", 0))
        on = float(spec.get("pulse_on_us", spec.get("pulse_length_us", 2000.0)))
        gap = float(spec.get("pulse_off_us", 0.0))
        reps = max(1, int(spec.get("pulse_repeats", 1)))
        d0 = float(spec.get("pulse_initial_delay_us", 0.0)) + off
        trail = bool(spec.get("pulse_count_trailing_off", False))
        total = d0 + reps * on + max(0, reps - 1) * gap + (gap if trail else 0)
        t = np.linspace(0, total * 1.02 + 1, n)
        y = np.zeros_like(t); cur = d0
        for r in range(reps):
            y[(t >= cur) & (t < cur + on)] = v
            cur += on + gap
        return t, y


# --------------------------------------------------------------------------- #
#  loading
# --------------------------------------------------------------------------- #
def _find_h5(rid) -> str:
    if isinstance(rid, str) and rid.endswith(".h5"):
        return rid
    rid = int(rid)
    hits = glob.glob(f"{RESULTS_ROOT}/**/{rid:09d}-*.h5", recursive=True)
    if not hits:
        raise FileNotFoundError(f"no results .h5 for RID {rid} under {RESULTS_ROOT}")
    return sorted(hits)[-1]


def _ds(f, key, default=None):
    d = f["datasets"]
    if key not in d:
        return default
    v = d[key][()]
    if isinstance(v, bytes):
        return v.decode()
    return v


def _read_jsonl(path):
    rows = []
    if not os.path.exists(path):
        return rows
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return rows


def load_run(rid) -> dict:
    """Everything needed to correlate, as a dict."""
    import h5py

    h5 = _find_h5(rid)
    run = {"h5": h5, "rid": None}
    with h5py.File(h5, "r") as f:
        run["rid"] = int(f["rid"][()]) if "rid" in f else None
        run["artiq_start_time"] = float(f["start_time"][()]) if "start_time" in f else None
        run["artiq_run_time"] = float(f["run_time"][()]) if "run_time" in f else None
        run["h5_mtime"] = os.path.getmtime(h5)
        try:
            run["args"] = json.loads(f["expid"][()]).get("arguments", {})
        except Exception:
            run["args"] = {}

        # -- RFSoC summary datasets --
        rf = {}
        for k in ("rfsoc_session_id", "rfsoc_fetch_dir", "rfsoc_v_thresh",
                  "rfsoc_event_count", "rfsoc_events_saved", "rfsoc_missed",
                  "rfsoc_mr_snapshots", "rfsoc_crossings_total", "rfsoc_revivals",
                  "rfsoc_pulse_fired", "rfsoc_pps", "rfsoc_last_event_npz"):
            rf[k] = _ds(f, k)
        for k in ("rfsoc_calibration", "rfsoc_status", "rfsoc_pulse_spec"):
            v = _ds(f, k)
            try:
                rf[k] = json.loads(v) if v else None
            except (TypeError, json.JSONDecodeError):
                rf[k] = v
        rf["crossing_t_s"] = np.asarray(_ds(f, "rfsoc_crossing_t_s", []), float)
        rf["crossing_amp"] = np.asarray(_ds(f, "rfsoc_crossing_amp", []), float)
        rf["last_trace_us"] = np.asarray(_ds(f, "rfsoc_last_trace_us", []), float)
        rf["last_trace_mag"] = np.asarray(_ds(f, "rfsoc_last_trace_mag", []), float)
        rf["last_trace_mag_lo"] = np.asarray(_ds(f, "rfsoc_last_trace_mag_lo", []), float)
        rf["last_trace_mag_mean"] = np.asarray(_ds(f, "rfsoc_last_trace_mag_mean", []), float)
        run["rfsoc"] = rf

        # -- SA traces per rep --
        run["sa_swt"] = float(run["args"].get("SSA_SWT", 0.1))
        run["p_thresh"] = float(run["args"].get("P_signal_threshold", -np.inf))
        run["sa_x"] = np.asarray(_ds(f, "t_data", []), float)          # SA sweep axis
        run["sa_run_t0_wall"] = _ds(f, "sa_run_t0_wall")
        reps = []
        n_rep = int(run["args"].get("N_repetition", 0)) or 200
        for N in range(n_rep):
            meas = _ds(f, f"all_meas_{N}")
            if meas is None:
                continue
            meas = np.atleast_2d(np.asarray(meas, float))
            if meas.size == 0 or meas.shape[-1] < 2:
                continue
            ts = np.asarray(_ds(f, f"time_stamps_{N}", []), float)
            wall = np.asarray(_ds(f, f"sa_wall_{N}", []), float)
            wall0 = np.asarray(_ds(f, f"sa_wall_start_{N}", []), float)   # ~= sweep start
            reps.append({"rep": N, "traces": meas, "t_nominal": ts,
                         "wall": wall if wall.size == meas.shape[0] else None,
                         "wall_start": wall0 if wall0.size == meas.shape[0] else None})
        run["sa_reps"] = reps

    # -- RFSoC session files from the fetch dir --
    # rfsoc_fetch_dir is written by the acquisition side only after the final
    # rsync succeeds, so it doubles as a success flag and is simply ABSENT when
    # that rsync tripped (e.g. the live per-capture purge deleting event_*.npz
    # out from under it -> rsync exit 24). The data is still on disk in that
    # case, so fall back to the session id / the last event's own path instead
    # of silently loading an empty run.
    fd = run["rfsoc"].get("rfsoc_fetch_dir") or ""
    if not os.path.isdir(fd):
        cands = []
        if fd:
            cands.append(os.path.join(RFSOC_DATA_ROOT, os.path.basename(fd.rstrip("/"))))
        sid = run["rfsoc"].get("rfsoc_session_id") or ""
        if sid:
            cands.append(os.path.join(RFSOC_DATA_ROOT, sid))
        last = run["rfsoc"].get("rfsoc_last_event_npz") or ""
        if last:
            cands.append(os.path.dirname(last))
        fd = next((c for c in cands if os.path.isdir(c)), fd)
        run["rfsoc"]["fetch_dir_recovered"] = bool(fd) and os.path.isdir(fd)
    run["rfsoc"]["fetch_dir"] = fd
    run["rfsoc"]["crossings"] = _read_jsonl(os.path.join(fd, "crossings.jsonl"))
    run["rfsoc"]["manifest"] = _read_jsonl(os.path.join(fd, "manifest.jsonl"))
    sj = os.path.join(fd, "status.json")
    if os.path.exists(sj) and not run["rfsoc"].get("rfsoc_status"):
        run["rfsoc"]["rfsoc_status"] = json.load(open(sj))
    return run


# --------------------------------------------------------------------------- #
#  wall-clock timeline
# --------------------------------------------------------------------------- #
def _board_t_start(run) -> float:
    st = run["rfsoc"].get("rfsoc_status") or {}
    return float(st.get("t_start") or run["artiq_start_time"] or 0.0)


def timeline(run) -> dict:
    """All events on one axis: seconds since the RFSoC session start (T0).
    Returns dict of arrays; also stores absolute wall times."""
    t0 = _board_t_start(run)
    rf = run["rfsoc"]
    st = rf.get("rfsoc_status") or {}
    pps = float(st.get("passes_per_second") or rf.get("rfsoc_pps") or 1.0)

    # -- RFSoC: crossings (prefer the full jsonl; fall back to the h5 arrays) --
    cr = rf.get("crossings") or []
    if cr:
        c_iter = np.array([c["iter"] for c in cr], float)
        c_amp = np.array([c["amp"] for c in cr], float)
        c_rel = np.array([c.get("t_rel_s", c["iter"] / pps) for c in cr], float)
    else:
        c_rel = rf["crossing_t_s"].copy()
        c_amp = rf["crossing_amp"].copy()
        c_iter = c_rel * pps
    c_wall = t0 + c_rel

    # a captured crossing has an exact manifest t_wall -> use it (matched by iter)
    man = rf.get("manifest") or []
    by_iter = {int(m["iter"]): m for m in man if "iter" in m}
    for k, it in enumerate(c_iter):
        m = by_iter.get(int(it))
        if m and m.get("t_wall"):
            c_wall[k] = float(m["t_wall"])

    # -- RFSoC: captures (DDR4 + MR) straight from the manifest --
    ddr4 = [m for m in man if not str(m.get("npz", "")).startswith("event_mr")
            and m.get("capture") != "mr_snapshot"]
    mr = [m for m in man if str(m.get("npz", "")).startswith("event_mr")
          or m.get("capture") == "mr_snapshot"]

    def _cap(rows):
        return {"wall": np.array([float(m["t_wall"]) for m in rows], float),
                "amp": np.array([float(m.get("amp", np.nan)) for m in rows], float),
                "npz": [m.get("npz") for m in rows],
                "iso": [m.get("iso") for m in rows]}

    # -- SA traces --
    sa_rows = []
    fallback = False
    for rp in run["sa_reps"]:
        peak = rp["traces"].max(axis=1)
        if rp["wall"] is not None and np.all(rp["wall"] > 1e9):
            wall = rp["wall"]
        else:
            fallback = True
            wall = _sa_wall_fallback(run, rp)
        for k in range(rp["traces"].shape[0]):
            sa_rows.append({"rep": rp["rep"], "k": k, "wall": float(wall[k]),
                            "peak": float(peak[k]), "t_nominal": float(
                                rp["t_nominal"][k] if k < len(rp["t_nominal"]) else np.nan)})

    return {
        "t0_wall": t0,
        "pps": pps,
        "crossings": {"wall": c_wall, "amp": c_amp, "iter": c_iter, "t_rel_s": c_rel},
        "ddr4": _cap(ddr4),
        "mr": _cap(mr),
        "sa": sa_rows,
        "sa_wall_is_estimate": fallback,
        "v_thresh": float(rf.get("rfsoc_v_thresh") or st.get("config", {}).get("v_thresh") or np.nan),
        "p_thresh": run["p_thresh"],
    }


def _sa_wall_fallback(run, rp) -> np.ndarray:
    """Older runs without sa_wall_*: estimate. The nominal i*SSA_SWT axis is
    stretched to fill the measured ARTIQ run duration, and each rep is placed at
    its nominal cumulative offset (also stretched). Good to ~+/- 10 s only."""
    args = run["args"]
    swt = run["sa_swt"]
    t_load = float(args.get("t_load", 20.0))
    t_data = float(args.get("t_data", 1.0))
    t_rest = float(args.get("t_rest", 1.0))
    per_rep_nominal = t_load + t_data + t_rest
    t_run0 = run.get("sa_run_t0_wall") or run["artiq_start_time"] or _board_t_start(run)
    t_run1 = run["h5_mtime"]
    n_rep = int(args.get("N_repetition", 1)) or 1
    total_nominal = per_rep_nominal * n_rep
    stretch = (t_run1 - t_run0) / total_nominal if total_nominal > 0 else 1.0
    rep_start = t_run0 + rp["rep"] * per_rep_nominal * stretch
    tn = rp["t_nominal"]
    return rep_start + np.asarray(tn, float) * stretch


# --------------------------------------------------------------------------- #
#  reporting
# --------------------------------------------------------------------------- #
def summary(run) -> None:
    rf = run["rfsoc"]
    st = rf.get("rfsoc_status") or {}
    tl = timeline(run)
    print(f"RID {run['rid']}   {os.path.basename(run['h5'])}")
    print(f"  session   {rf.get('rfsoc_session_id')}   fetch_dir={rf.get('fetch_dir')}")
    if rf.get("fetch_dir_recovered"):
        print("            [!] rfsoc_fetch_dir was not recorded (the run's final "
              "rsync reported a failure);\n"
              "                dir recovered from rfsoc_session_id. The board copy "
              "was almost certainly not purged.")
    print(f"  board t_start {_board_t_start(run):.3f}   elapsed {st.get('elapsed_s', '?')} s"
          f"   pps {tl['pps']:.0f}")
    print(f"  v_thresh {tl['v_thresh']:.0f}   calib={rf.get('rfsoc_calibration')}")
    print(f"  counts: crossings_total={rf.get('rfsoc_crossings_total')}  "
          f"event_count(DDR4)={rf.get('rfsoc_event_count')}  events_saved={rf.get('rfsoc_events_saved')}  "
          f"pulse_fired={rf.get('rfsoc_pulse_fired')}  mr={rf.get('rfsoc_mr_snapshots')}  "
          f"missed={rf.get('rfsoc_missed')}")
    n_sa = len(tl["sa"])
    flat = _sa_flat(run)
    has_start = any(s.get("wall_start") is not None for s in flat)
    ov = [s["overhead_s"] for s in flat if s.get("overhead_s") is not None]
    if tl["sa_wall_is_estimate"]:
        tag = "[wall = ESTIMATE +/- ~10 s]"
    elif has_start and ov:
        o = np.median(ov) * 1e3
        tag = f"[SA anchor = acq-window midpoint, +/-{o/2:.0f} ms  (overhead {o:.0f} ms)]"
    elif has_start:
        tag = "[SA anchor = acq-window midpoint]"
    else:
        tag = "[SA anchor = sweep-END, +*OPC?/transfer lag ~10-40 ms]"
    print(f"  SA: {n_sa} saved trace(s) over {len(run['sa_reps'])} rep(s)"
          f"   P_thresh {run['p_thresh']} dBm   {tag}")
    if tl["ddr4"]["wall"].size:
        print("  DDR4 events:")
        for w, a, n in zip(tl["ddr4"]["wall"], tl["ddr4"]["amp"], tl["ddr4"]["npz"]):
            near = _nearest_sa(tl, w)
            print(f"    {w:.3f}  ({w - tl['t0_wall']:+.1f} s)  amp={a:.0f}  {n}"
                  + (f"   nearest SA trace dt={near:+.1f} s" if near is not None else "   (no SA trace)"))
    if tl["sa"]:
        print("  SA traces:")
        for r in tl["sa"]:
            print(f"    rep {r['rep']:>2d} k{r['k']}  {r['wall']:.3f}  ({r['wall'] - tl['t0_wall']:+.1f} s)"
                  f"  peak={r['peak']:.1f} dBm")


def _nearest_sa(tl, wall):
    if not tl["sa"]:
        return None
    d = [r["wall"] - wall for r in tl["sa"]]
    return min(d, key=abs)


# --------------------------------------------------------------------------- #
#  plots  --  slide / Notion friendly: compact aspect, large fonts
# --------------------------------------------------------------------------- #
# tweak globally with e.g.  sa_rfsoc_corr.STYLE["font.size"] = 15
STYLE = {
    "figure.dpi": 120,
    "savefig.dpi": 120,
    "savefig.bbox": "tight",
    "font.size": 14,
    "axes.titlesize": 15,
    "axes.titleweight": "bold",
    "axes.labelsize": 14,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 12,
    "lines.linewidth": 1.6,
    "axes.grid": True,
    "grid.alpha": 0.25,
}


# the lab-wide "big font" look (edes.utils.plotting.big_plt_font) as a dict we
# can merge into STYLE instead of mutating the global rcParams
_BIG_FONT_FALLBACK = {"font.size": 14, "lines.markersize": 12,
                      "lines.linewidth": 2.5, "xtick.labelsize": 15,
                      "ytick.labelsize": 15, "errorbar.capsize": 2}


def _big_font_style():
    """``edes.utils.plotting.big_plt_font()``'s settings, captured as a dict.

    That helper updates the global ``rcParams`` in place, so it is applied
    inside a throwaway ``rc_context`` and the keys it changed are read back —
    the repo stays the single source of truth for the style, and nothing
    global is left modified. Falls back to a local copy of the same settings
    when ``edes`` isn't importable (e.g. a bare notebook kernel)."""
    import matplotlib.pyplot as plt
    try:
        from edes.utils.plotting import big_plt_font
    except Exception:
        return dict(_BIG_FONT_FALLBACK)
    try:
        with plt.rc_context():
            before = dict(plt.rcParams)
            big_plt_font()
            out = {}
            for k, v in plt.rcParams.items():
                try:
                    if k in before and before[k] != v:
                        out[k] = v
                except Exception:
                    pass
        return out or dict(_BIG_FONT_FALLBACK)
    except Exception:
        return dict(_BIG_FONT_FALLBACK)


def _rc(extra=None, big_font=False):
    """rc context for the module plots: ``STYLE``, optionally on top of the
    lab's big-font rcParams, with ``extra`` overriding both."""
    import matplotlib.pyplot as plt
    rc = dict(STYLE)
    if big_font:
        rc.update(_big_font_style())
        rc.update({k: v for k, v in STYLE.items()
                   if k.startswith(("axes.title", "axes.label", "legend.", "grid."))})
    if extra:
        rc.update(extra)
    return plt.rc_context(rc)


def _envelope(mag, t, n=2000, step=None):
    """per-bin (t, min, max, mean). ``n`` targets a total output point count;
    pass ``step`` (samples per bin) instead to fix the decimation directly,
    which overrides ``n``."""
    mag = np.asarray(mag); t = np.asarray(t)
    if step is None:
        if len(mag) <= 2 * n:
            return t, mag, mag, mag
        step = int(np.ceil(len(mag) / n))
    if step <= 1 or len(mag) <= step:
        return t, mag, mag, mag
    s = step; k = (len(mag) // s) * s
    m = mag[:k].reshape(-1, s)
    return t[:k:s], m.min(1), m.max(1), m.mean(1)


def plot_timeline(run, ax=None, log_y=False, figsize=(9.5, 5.2)):
    """One wall-clock axis for the SA traces and the RFSoC detector.

    A faint full-height guide line marks every DDR4 capture, MR snapshot and SA
    trace so a marker is never fully hidden behind another; markers are
    semi-transparent. ``log_y`` helps when one crossing dwarfs the rest."""
    import matplotlib.pyplot as plt
    tl = timeline(run)
    t0 = tl["t0_wall"]
    with _rc():
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
        else:
            fig = ax.figure
        axr = ax.twinx()
        axr.grid(False)

        cw = tl["crossings"]["wall"] - t0
        ca = tl["crossings"]["amp"]
        dw = tl["ddr4"]["wall"] - t0
        mw = tl["mr"]["wall"] - t0

        for x in dw:
            ax.axvline(x, color="tab:blue", alpha=.25, lw=1.4, zorder=1)
        for x in mw:
            ax.axvline(x, color="tab:orange", alpha=.18, lw=1, zorder=1)
        for r in tl["sa"]:
            ax.axvline(r["wall"] - t0, color="tab:green", alpha=.18, lw=1, zorder=1)

        if cw.size:
            ax.vlines(cw, 0, ca, color="0.6", lw=1, alpha=.4, zorder=2)
            ax.scatter(cw, ca, s=32, color="0.30", alpha=.7, zorder=3, label="RFSoC crossing")
        if np.isfinite(tl["v_thresh"]):
            ax.axhline(tl["v_thresh"], color="crimson", ls="--", lw=1.4, alpha=.85, label="v_thresh")
        if dw.size:
            ax.scatter(dw, tl["ddr4"]["amp"], marker="*", s=340, facecolor="tab:blue",
                       edgecolor="k", lw=.9, alpha=.9, zorder=6, label="DDR4 capture")
        if mw.size:
            ax.scatter(mw, tl["mr"]["amp"], marker="v", s=70, color="tab:orange",
                       alpha=.85, zorder=5, label="MR snapshot")

        if tl["sa"]:
            sx = [r["wall"] - t0 for r in tl["sa"]]
            sy = [r["peak"] for r in tl["sa"]]
            axr.scatter(sx, sy, marker="s", s=95, color="tab:green", edgecolor="k",
                        lw=.7, alpha=.85, zorder=5, label="SA saved trace (peak)")
            for r in tl["sa"]:
                axr.annotate(f"rep{r['rep']}", (r["wall"] - t0, r["peak"]),
                             textcoords="offset points", xytext=(0, 10), ha="center",
                             fontsize=11)
        if np.isfinite(tl["p_thresh"]):
            axr.axhline(tl["p_thresh"], color="tab:green", ls=":", lw=1.4, alpha=.7)
        axr.set_ylabel("SA peak power [dBm]", color="tab:green")
        axr.tick_params(axis="y", colors="tab:green")
        if tl["sa"]:
            lo = min([r["peak"] for r in tl["sa"]]
                     + ([tl["p_thresh"]] if np.isfinite(tl["p_thresh"]) else []))
            hi = max(r["peak"] for r in tl["sa"])
            pad = max(0.5, (hi - lo) * 0.30)
            axr.set_ylim(lo - pad, hi + pad)

        tops = [v for v in [tl["v_thresh"],
                            ca.max() if ca.size else None,
                            tl["ddr4"]["amp"].max() if tl["ddr4"]["amp"].size else None,
                            tl["mr"]["amp"].max() if tl["mr"]["amp"].size else None]
                if v is not None and np.isfinite(v)]
        if tops:
            if log_y:
                ax.set_yscale("log"); ax.set_ylim(max(1, min(tops) * 0.5), max(tops) * 1.7)
            else:
                ax.set_ylim(0, max(tops) * 1.20)

        ax.set_xlabel("time since RFSoC session start [s]")
        ax.set_ylabel("RFSoC detector metric [v_thresh units]")
        ax.set_title(f"RID {run['rid']}  —  SA vs RFSoC standby detector")
        if tl["sa_wall_is_estimate"]:
            ax.text(0.012, 0.97, "SA times ≈ estimate  (±~10 s)",
                    transform=ax.transAxes, ha="left", va="top", fontsize=10.5,
                    color="crimson",
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="crimson", alpha=.85))
        h1, l1 = ax.get_legend_handles_labels()
        h2, l2 = axr.get_legend_handles_labels()
        ax.legend(h1 + h2, l1 + l2, loc="upper center", bbox_to_anchor=(0.5, -0.14),
                  ncol=3, frameon=False, fontsize=11)
        ax.margins(x=0.05)
    return ax


def plot_sa_traces(run, figsize=None):
    import matplotlib.pyplot as plt
    reps = run["sa_reps"]
    n = sum(r["traces"].shape[0] for r in reps)
    if n == 0:
        print("no SA traces saved in this run")
        return
    x = run["sa_x"] if run["sa_x"].size == (reps[0]["traces"].shape[1]) else \
        np.arange(reps[0]["traces"].shape[1])
    with _rc():
        fig, axes = plt.subplots(n, 1, squeeze=False, constrained_layout=True,
                                 figsize=figsize or (7.6, 3.1 * n))
        i = 0
        for rp in reps:
            for k in range(rp["traces"].shape[0]):
                a = axes[i, 0]
                a.plot(x, rp["traces"][k], lw=1.6, color="tab:blue")
                if np.isfinite(run["p_thresh"]):
                    a.axhline(run["p_thresh"], color="tab:green", ls=":", lw=1.4,
                              label="P_signal_threshold")
                tn = rp["t_nominal"][k] if k < len(rp["t_nominal"]) else np.nan
                a.set_title(f"rep {rp['rep']} · trace {k}   "
                            f"t≈{tn:.1f} s   peak {rp['traces'][k].max():.1f} dBm")
                a.set_ylabel("power [dBm]")
                if i == 0:
                    a.legend(loc="best", fontsize=10)
                i += 1
        axes[-1, 0].set_xlabel("SA sweep axis  (dataset 't_data')")
        fig.suptitle(f"RID {run['rid']}  —  saved SA spectra")
    return axes


def _event_number(npz):
    """event number from a manifest filename (``event_000007.npz`` -> 7,
    ``event_mr_000003.npz`` -> 3), with the zero padding dropped. ``None`` when
    the name carries no number."""
    m = re.search(r"(\d+)", os.path.basename(str(npz or "")))
    return int(m.group(1)) if m else None


def _event_label(npz):
    """short human label for a capture: ``event 7`` / ``MR 3`` (no zero pad)."""
    name = os.path.basename(str(npz or ""))
    kind = "MR" if name.startswith("event_mr") else "event"
    n = _event_number(name)
    return f"{kind} {n}" if n is not None else (name or "event")


def _event_trace(m, fd, n_points, decimation, xlim, disp_to_raw, raw_to_disp):
    """load one capture's ``|I,Q|`` and reduce it to ``(t, min, max, mean)`` per
    bin, cropped to ``xlim`` (display units) first and with ``t`` returned in
    display units. ``None`` when the .npz was not fetched."""
    p = os.path.join(fd, m["npz"])
    if not os.path.exists(p):
        return None
    d = np.load(p)
    iq = d["iq"].astype(np.float32)
    mag = np.hypot(iq[:, 0], iq[:, 1])
    fs = float(d["fs_msps"]) if "fs_msps" in getattr(d, "files", []) else F_OUT_MHZ
    t = np.arange(len(mag)) / fs
    if xlim is not None:
        x0, x1 = xlim
        r0 = None if x0 is None else x0 * disp_to_raw
        r1 = None if x1 is None else x1 * disp_to_raw
        i0 = 0 if r0 is None else int(np.searchsorted(t, r0, side="left"))
        i1 = len(t) if r1 is None else int(np.searchsorted(t, r1, side="right"))
        t, mag = t[i0:i1], mag[i0:i1]
    tb, lo, hi, mean = _envelope(mag, t, n_points, step=decimation)
    return tb * raw_to_disp, lo, hi, mean


def _event_colors(n, cmap=None):
    """one distinct colour per event: the qualitative tab10/tab20 cycle for a
    handful of events, a continuous colormap once there are more."""
    import matplotlib.pyplot as plt
    if cmap is None and n <= 20:
        base = plt.get_cmap("tab10" if n <= 10 else "tab20")
        return [base(i % base.N) for i in range(n)]
    cm = plt.get_cmap(cmap or "viridis")
    return [cm(x) for x in (np.linspace(0, 0.88, n) if n > 1 else [0.0])]


def plot_ddr4_events(run, n_points=3000, decimation=None, db=False, db_ref=1.0,
                     envelope=True, pulse=False, pulse_offset_us=None, figsize=None,
                     xlim=None, ylim=None, raw_unit="us", time_unit="us",
                     linestyle="-", marker=None):
    """DDR4 capture windows: |I,Q| per bin (mean solid; min–max band unless
    ``envelope=False``). ``db=True`` plots ``20·log10(|I,Q| / db_ref)`` [dB]
    instead of linear counts; ``db_ref`` sets the 0 dB reference.

    ``pulse=True`` overlays the **programmed feedback pulse** (from
    ``rfsoc_pulse_spec``) on a twin right y-axis — drive fraction vs the same
    time axis, offset by the crossing->RF latency (``crossing_to_pulse_us`` from
    the status; override with ``pulse_offset_us``).

    ``xlim``/``ylim`` set the (left) axes limits, applied to every subplot;
    each is a ``(lo, hi)`` tuple, or ``None`` to leave matplotlib's default.
    When ``xlim`` is given, the raw samples are first cropped to that window
    before binning into ``n_points``, so the envelope is computed only over
    the visible range (cheaper, and higher effective resolution) instead of
    over the full capture.

    ``decimation``, when given, fixes the bin size directly (samples per
    output point) instead of targeting ``n_points`` total points — useful
    once ``xlim`` has already cropped the window and you want a fixed,
    predictable decimation rather than one that depends on the crop width.

    ``raw_unit`` is the time unit the DDR4 sample clock (``fs_msps``) is
    interpreted in — one of ``"ns"``, ``"us"``, ``"ms"``, ``"s"``; it's
    ``"us"`` by default since ``fs_msps`` is Msamples/s. ``time_unit`` is the
    unit everything is shown/accepted in on the plot — the x-axis, its label,
    and ``xlim`` — independent of ``raw_unit``. Both default to ``"us"``, so
    the plot is unchanged unless you set one of them.

    ``linestyle``/``marker`` set the style of the mean |I,Q| trace (and the
    feedback-pulse trace, when ``pulse=True``) — any matplotlib linestyle
    (``"-"``, ``"--"``, ``":"``, ``"-."``, ``""``/``"None"`` for markers only)
    and marker spec (``"o"``, ``"."``, ``None`` for no markers). ``linestyle``
    also accepts a combined fmt shorthand like ``".-"`` or ``"o--"`` (as in
    ``plt.plot(x, y, ".-")``) when ``marker`` is left at its default — it's
    parsed into the separate style/marker automatically."""
    import matplotlib.pyplot as plt
    tl = timeline(run)
    fd = run["rfsoc"]["fetch_dir"]
    ev = [m for m in (run["rfsoc"].get("manifest") or [])
          if m.get("capture") == "ok" and m.get("npz")]
    if not ev:
        print("no DDR4 event_*.npz in the manifest")
        return
    pw = _pulse_waveform(run, offset_us=pulse_offset_us) if pulse else None
    if pulse and pw is None:
        print("[pulse] no feedback pulse in this run (rfsoc_pulse_spec / voltage 0)")

    def _y(a):
        if not db:
            return np.asarray(a)
        with np.errstate(divide="ignore", invalid="ignore"):
            return 20 * np.log10(np.where(np.asarray(a) > 0, a, np.nan) / db_ref)

    raw_to_disp = _unit_scale(raw_unit, time_unit)   # sample-clock time -> display
    us_to_disp = _unit_scale("us", time_unit)        # pulse waveform (always µs) -> display
    disp_to_raw = _unit_scale(time_unit, raw_unit)    # xlim (display) -> sample-clock time
    linestyle, marker = _parse_style(linestyle, marker)

    axes = []
    with _rc():
        for m in ev:
            fig, a = plt.subplots(1, 1, constrained_layout=True,
                                  figsize=figsize or (9.5, 3.6))
            axes.append(a)
            tr = _event_trace(m, fd, n_points, decimation, xlim,
                              disp_to_raw, raw_to_disp)
            if tr is None:
                a.text(0.5, 0.5, f"missing {m['npz']}\n(rerun with rfsoc_fetch='full')",
                       ha="center", va="center", transform=a.transAxes, fontsize=12)
                continue
            tb, lo, hi, mean = tr
            if envelope:
                a.fill_between(tb, _y(lo), _y(hi), color="tab:blue", alpha=.25, lw=0,
                               label="min–max envelope")
            a.plot(tb, _y(mean), color="tab:blue", lw=1.6,
                   linestyle=linestyle, marker=marker, markersize=4,
                   label="mean |I,Q|" if (envelope or pw is not None) else None)

            if pw is not None:
                axr = a.twinx(); axr.grid(False)
                pt, pg = pw
                pt = pt * us_to_disp
                axr.fill_between(pt, 0, pg, color="tab:red", alpha=.12, lw=0)
                axr.plot(pt, pg, color="tab:red", lw=1.4,
                         linestyle=linestyle, marker=marker, markersize=4,
                         label="feedback pulse (programmed)")
                axr.set_ylabel("pulse drive [frac]", color="tab:red")
                axr.tick_params(axis="y", colors="tab:red")
                axr.set_ylim(0, max(pg.max() * 1.15, 1e-6))
                axr.set_xlim(tb[0], tb[-1])

            h, l = a.get_legend_handles_labels()
            if pw is not None:
                h2, l2 = axr.get_legend_handles_labels(); h += h2; l += l2
            if h:
                a.legend(h, l, loc="upper right", fontsize=10)
            w = float(m["t_wall"])
            near = _nearest_sa(tl, w)
            ttl = (f"RID {run['rid']}   {m['npz']}   +{w - tl['t0_wall']:.1f} s"
                   f"   crossing amp {m.get('amp')}")
            if near is not None:
                ttl += f"   |  nearest SA Δt = {near:+.1f} s"
            if db:
                ttl += "   (dB)"
            if pw is not None:
                ttl += "   + feedback pulse"
            a.set_title(ttl)
            a.set_ylabel("20·log10|I,Q| [dB]" if db else "|I,Q|")
            a.set_xlabel(f"time in capture window [{_TIME_UNIT_LABEL[time_unit]}]"
                         + ("" if pw is not None else "  —  pulse plays over the first ~pulse_length_us"))
            if xlim is not None:
                a.set_xlim(xlim)
            if ylim is not None:
                a.set_ylim(ylim)
    return axes


def _shades(base, k):
    """``k`` tints of one base colour — so several events of the SAME run read
    as a family while still being told apart."""
    import colorsys
    import matplotlib.colors as mcolors
    r, g, b = mcolors.to_rgb(base)
    h, li, sat = colorsys.rgb_to_hls(r, g, b)
    if k <= 1:
        return [(r, g, b)]
    ls = np.linspace(0.25, 0.68, k)                       # light->dark, but no
                                                          # tint so pale it vanishes
    ss = np.linspace(min(1.0, sat * 1.15), max(0.25, sat * 0.55), k)
    return [colorsys.hls_to_rgb(h, float(x), float(y)) for x, y in zip(ls, ss)]


def _is_run(o):
    return isinstance(o, dict) and "rfsoc" in o and "sa_reps" in o


def _norm_run_specs(runs, noise=None, events=None, _cache=None):
    """normalise the ``runs`` argument of :func:`plot_ddr4_events_multi` into a
    list of ``{run, rid, events, label, noise, color}`` dicts.

    Accepted items: an RID (int) or ``.h5`` path, an already-loaded ``run``
    dict, a ``(rid, events)`` tuple, or a spec dict with any of
    ``rid``/``run``, ``events``, ``label``, ``noise``, ``color``.

    A run named in ``noise`` that is NOT among ``runs`` is loaded and appended
    as a background spec, so ``noise=`` alone is enough to bring a background
    run in — it does not have to be listed twice."""
    cache = {} if _cache is None else _cache
    if isinstance(runs, (int, str)) or _is_run(runs) or isinstance(runs, dict):
        runs = [runs]
    noise_set = set()
    if noise is not None:
        noise_set = {noise} if isinstance(noise, (int, str)) else set(noise)

    out, seen = [], set()
    for it in runs:
        sp = {}
        if isinstance(it, tuple) and len(it) == 2:
            sp = {"rid": it[0], "events": it[1]}
        elif _is_run(it) or isinstance(it, (int, str)):
            sp = {"run": it} if _is_run(it) else {"rid": it}
        elif isinstance(it, dict):
            sp = dict(it)
        else:
            raise TypeError(f"can't read a run spec out of {it!r}")
        r = sp.get("run", sp.get("rid"))
        if _is_run(r):
            run = r
        else:
            key = str(r)
            if key not in cache:
                cache[key] = load_run(r)
            run = cache[key]
        rid = run["rid"]
        ev = sp.get("events", events)
        out.append({"run": run, "rid": rid,
                    "events": ev,
                    "label": sp.get("label") or f"RID {rid}",
                    "noise": bool(sp.get("noise", rid in noise_set
                                          or str(rid) in noise_set)),
                    "color": sp.get("color")})
        seen.update({rid, str(rid), str(r)})

    # a noise run the caller named but did not list in `runs`: bring it in
    # rather than silently leaving the figure with no background at all
    for nz in noise_set:
        if nz in seen or str(nz) in seen:
            continue
        if _is_run(nz):
            run = nz
        else:
            key = str(nz)
            if key not in cache:
                cache[key] = load_run(nz)
            run = cache[key]
        rid = run["rid"]
        if rid in seen or str(rid) in seen:
            continue
        seen.update({rid, str(rid), str(nz)})
        out.append({"run": run, "rid": rid, "events": events,
                    "label": f"RID {rid}", "noise": True, "color": None})
    return out


def _select_events(run, events=None):
    """the run's usable captures, optionally filtered to the event numbers in
    ``events`` (an int or a list of them, as shown in the plot labels)."""
    ev = [m for m in (run["rfsoc"].get("manifest") or [])
          if m.get("capture") == "ok" and m.get("npz")]
    if events is not None:
        want = {events} if isinstance(events, (int, np.integer)) else set(events)
        ev = [m for m in ev if _event_number(m.get("npz")) in want]
    return ev


def plot_ddr4_events_overlay(run, events=None, n_points=3000, decimation=None,
                             db=False, db_ref=1.0, envelope=False, pulse=False,
                             pulse_offset_us=None, figsize=None,
                             xlim=None, ylim=None, raw_unit="us", time_unit="us",
                             linestyle="-", marker=None, cmap=None, ax=None,
                             legend_loc="best"):
    """Like :func:`plot_ddr4_events`, but ALL capture windows land on ONE axes —
    each event in its own colour, labelled ``event 3`` / ``MR 2`` in the legend
    (the zero padding of ``event_000003.npz`` dropped). Every window shares the
    same in-capture time axis (t=0 = crossing), so this is the view for
    comparing shot-to-shot shape rather than one-figure-per-event.

    ``events`` selects which captures to draw, by the same numbers shown in the
    legend (an int, or a list/tuple of them) — e.g. ``events=[1, 4, 7]``.
    ``None`` (default) draws every capture in the manifest.

    ``envelope`` defaults to **False** here (the min–max bands of several events
    smear into each other); set it True to get each event's band in its own
    colour anyway. ``pulse=True`` draws the programmed feedback pulse once, on a
    twin right y-axis, since it is the same waveform for every event.

    ``cmap`` picks the colour source: by default a qualitative cycle (tab10, or
    tab20 past 10 events) and a continuous colormap past 20 — pass any colormap
    name (``"viridis"``, ``"turbo"``, …) to force a gradient, which reads
    better when the events are a time series.

    ``ax`` draws into an existing axes instead of making a figure. All other
    arguments (``n_points``, ``decimation``, ``db``/``db_ref``, ``xlim``/``ylim``,
    ``raw_unit``/``time_unit``, ``linestyle``/``marker``) mean exactly what they
    mean in :func:`plot_ddr4_events`."""
    import matplotlib.pyplot as plt
    tl = timeline(run)
    fd = run["rfsoc"]["fetch_dir"]
    ev = [m for m in (run["rfsoc"].get("manifest") or [])
          if m.get("capture") == "ok" and m.get("npz")]
    if not ev:
        print("no DDR4 event_*.npz in the manifest")
        return
    if events is not None:
        want = {events} if isinstance(events, (int, np.integer)) else set(events)
        ev = [m for m in ev if _event_number(m.get("npz")) in want]
        if not ev:
            print(f"no capture matches events={sorted(want)}")
            return
    pw = _pulse_waveform(run, offset_us=pulse_offset_us) if pulse else None
    if pulse and pw is None:
        print("[pulse] no feedback pulse in this run (rfsoc_pulse_spec / voltage 0)")

    def _y(a):
        if not db:
            return np.asarray(a)
        with np.errstate(divide="ignore", invalid="ignore"):
            return 20 * np.log10(np.where(np.asarray(a) > 0, a, np.nan) / db_ref)

    raw_to_disp = _unit_scale(raw_unit, time_unit)   # sample-clock time -> display
    us_to_disp = _unit_scale("us", time_unit)        # pulse waveform (always µs) -> display
    disp_to_raw = _unit_scale(time_unit, raw_unit)   # xlim (display) -> sample-clock time
    linestyle, marker = _parse_style(linestyle, marker)
    colors = _event_colors(len(ev), cmap)

    with _rc():
        if ax is None:
            fig, ax = plt.subplots(1, 1, constrained_layout=True,
                                   figsize=figsize or (9.5, 5.2))
        drawn, missing, t_lo, t_hi = 0, [], np.inf, -np.inf
        for m, col in zip(ev, colors):
            tr = _event_trace(m, fd, n_points, decimation, xlim,
                              disp_to_raw, raw_to_disp)
            if tr is None:
                missing.append(m["npz"])
                continue
            tb, lo, hi, mean = tr
            if not len(tb):
                continue
            t_lo, t_hi = min(t_lo, tb[0]), max(t_hi, tb[-1])
            if envelope:
                ax.fill_between(tb, _y(lo), _y(hi), color=col, alpha=.18, lw=0)
            ax.plot(tb, _y(mean), color=col, lw=1.6,
                    linestyle=linestyle, marker=marker, markersize=4,
                    label=_event_label(m["npz"]))
            drawn += 1
        if missing:
            ax.text(0.012, 0.035,
                    f"{len(missing)} capture(s) not fetched\n(rerun with rfsoc_fetch='full')",
                    transform=ax.transAxes, fontsize=9, color="crimson", va="bottom",
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="crimson", alpha=.85))

        if pw is not None:
            axr = ax.twinx(); axr.grid(False)
            pt, pg = pw
            pt = pt * us_to_disp
            axr.fill_between(pt, 0, pg, color="tab:red", alpha=.12, lw=0)
            axr.plot(pt, pg, color="tab:red", lw=1.4,
                     linestyle=linestyle, marker=marker, markersize=4,
                     label="feedback pulse (programmed)")
            axr.set_ylabel("pulse drive [frac]", color="tab:red")
            axr.tick_params(axis="y", colors="tab:red")
            axr.set_ylim(0, max(pg.max() * 1.15, 1e-6))
            if np.isfinite(t_lo) and np.isfinite(t_hi):
                axr.set_xlim(t_lo, t_hi)

        h, l = ax.get_legend_handles_labels()
        if pw is not None:
            h2, l2 = axr.get_legend_handles_labels(); h += h2; l += l2
        if h:
            ax.legend(h, l, loc=legend_loc, fontsize=10,
                      ncol=max(1, int(np.ceil(len(h) / 12))))
        ttl = f"RID {run['rid']}   {drawn} DDR4 capture window(s) overlaid"
        span = [float(m["t_wall"]) - tl["t0_wall"] for m in ev if m.get("t_wall")]
        if span:
            ttl += f"   +{min(span):.1f}–{max(span):.1f} s"
        if db:
            ttl += "   (dB)"
        if pw is not None:
            ttl += "   + feedback pulse"
        ax.set_title(ttl)
        ax.set_ylabel("20·log10|I,Q| [dB]" if db else "|I,Q|")
        ax.set_xlabel(f"time in capture window [{_TIME_UNIT_LABEL[time_unit]}]"
                      + ("" if pw is not None else "  —  pulse plays over the first ~pulse_length_us"))
        if xlim is not None:
            ax.set_xlim(xlim)
        if ylim is not None:
            ax.set_ylim(ylim)
    return ax


def plot_ddr4_events_multi(runs, noise=None, events=None, n_points=3000,
                           decimation=None, db=False, db_ref=1.0, envelope=False,
                           figsize=None, xlim=None, ylim=None, raw_unit="us",
                           time_unit="us", linestyle="-", noise_linestyle="--",
                           marker=None, cmap=None, noise_color="0.15",
                           noise_on_top=True, noise_alpha=0.8, noise_band=False,
                           split=False, share_y=True, color_by="auto", ax=None,
                           legend_loc="best", title=None):
    """DDR4 capture windows from SEVERAL runs on one axes, with some runs marked
    as **noise background**.

    ``runs`` is a list whose items can each be

      * an RID (``11333``) or a path to the results ``.h5``,
      * an already-loaded ``run`` dict (from :func:`load_run`) — handy when you
        have it in the notebook already and don't want to re-read the file,
      * a ``(rid, events)`` tuple to take only some captures of that run,
      * a spec dict ``{"rid": …, "events": […], "label": …, "noise": True,
        "color": …}`` for full control of one entry.

    Two ways to mark background: ``noise=[11330]`` (an RID or list of them), or
    ``"noise": True`` inside a spec dict. Background traces are drawn in grey
    shades, dashed (``noise_linestyle``) and thinner, and their legend entry
    gets a ``[bg]`` tag — so a noise run never competes for a colour with the
    runs you actually care about.

    Keeping the background readable (it is easily buried under a few signal
    traces):

      * ``noise_on_top=True`` (default) draws it ON TOP of the signal at
        ``noise_alpha`` (0.65) so it reads as an overlay you can see through;
        ``noise_on_top=False`` puts it back underneath.
      * ``noise_band=True`` collapses each background run into ONE summary
        band — min–max across its events, with the median as a line — instead
        of one trace per event. With a dozen noise captures this is much
        easier to read than a dozen dashed lines, and it shrinks the legend to
        a single entry. It falls back to per-event traces if the events don't
        share a time grid.
      * ``noise_color`` is the base colour (default near-black ``"0.15"``; any
        matplotlib colour works, e.g. ``"crimson"`` when grey is too quiet).
      * ``noise_alpha`` is the overall background opacity — push it to ``1.0``
        for a background that has to be as loud as the signal.
      * ``split=True`` gives up on overlaying entirely and puts the background
        in its OWN panel under the signal, on a shared x axis (and a shared y
        axis unless ``share_y=False``, so the two levels stay comparable).
        When the background overlaps the signal in both time and amplitude —
        which is the normal case for a noise-trigger run — this is the only
        layout where you can actually see both; it returns
        ``(ax_signal, ax_background)``.

    Every trace is labelled ``RID 11333 · event 2``: run label (override per
    spec with ``"label"``) plus the event number with its zero padding dropped.

    ``color_by`` controls how the signal traces are coloured:

      * ``"event"`` — every event gets its OWN colour from the qualitative
        cycle, so events are maximally separable;
      * ``"run"`` — one colour family per run (tints of the run's base
        colour), so a run reads as a group;
      * ``"auto"`` (default) — ``"event"`` with a single signal run, ``"run"``
        with several (where telling the runs apart matters more).

    All the sampling/axis arguments — ``events`` (a default event filter for
    every run that doesn't set its own), ``n_points``, ``decimation``,
    ``db``/``db_ref``, ``envelope``, ``xlim``/``ylim``, ``raw_unit``/``time_unit``,
    ``linestyle``/``marker`` — mean what they do in :func:`plot_ddr4_events`.
    ``cmap`` picks the signal colour source, ``ax`` draws into an existing axes,
    and ``title`` overrides the auto-generated one.

    Example::

        C.plot_ddr4_events_multi([11333, (11330, [1, 2, 3]), 11277],
                                 noise=11277, noise_band=True,
                                 time_unit="ms", decimation=500)"""
    import matplotlib.pyplot as plt
    specs = _norm_run_specs(runs, noise=noise, events=events)
    if not specs:
        print("no runs given")
        return
    for sp in specs:
        sp["ev"] = _select_events(sp["run"], sp["events"])
        if not sp["ev"]:
            print(f"RID {sp['rid']}: no DDR4 capture matches"
                  + ("" if sp["events"] is None else f" events={sp['events']}"))
    specs = [sp for sp in specs if sp["ev"]]
    if not specs:
        return

    def _y(a):
        if not db:
            return np.asarray(a)
        with np.errstate(divide="ignore", invalid="ignore"):
            return 20 * np.log10(np.where(np.asarray(a) > 0, a, np.nan) / db_ref)

    raw_to_disp = _unit_scale(raw_unit, time_unit)
    disp_to_raw = _unit_scale(time_unit, raw_unit)
    linestyle, marker = _parse_style(linestyle, marker)
    noise_linestyle, _ = _parse_style(noise_linestyle, marker)

    sig = [sp for sp in specs if not sp["noise"]]
    bg = [sp for sp in specs if sp["noise"]]
    per_event = (color_by == "event" or (color_by == "auto" and len(sig) <= 1))
    if per_event:
        # one distinct colour per signal event, cycling across all signal runs
        pal = _event_colors(sum(len(sp["ev"]) for sp in sig), cmap)
        i = 0
        for sp in sig:
            sp["colors"] = ([sp["color"]] * len(sp["ev"]) if sp["color"]
                            else pal[i:i + len(sp["ev"])])
            i += len(sp["ev"])
            sp["color"] = sp["colors"][0]
    else:
        for sp, col in zip(sig, _event_colors(len(sig), cmap)):
            sp["color"] = sp["color"] or col
            sp["colors"] = _shades(sp["color"], len(sp["ev"]))
    for sp, col in zip(bg, _shades(noise_color, len(bg))):
        sp["color"] = sp["color"] or col
        sp["colors"] = _shades(sp["color"], len(sp["ev"]))

    split_on = bool(split and bg and sig)
    if split and not split_on:
        print("[split] nothing to split — every run is signal or every run is background")
    with _rc():
        axb = None
        if ax is None:
            if split_on:
                fig, (ax, axb) = plt.subplots(
                    2, 1, sharex=True, sharey=share_y, constrained_layout=True,
                    figsize=figsize or (9.8, 7.0),
                    gridspec_kw=dict(height_ratios=[2, 1]))
            else:
                fig, ax = plt.subplots(1, 1, constrained_layout=True,
                                       figsize=figsize or (9.8, 5.4))
        elif split_on:
            print("[split] ignored — drawing into the ax you passed")
            split_on = False
        missing = []
        z_bg = 3.0 if noise_on_top else 1.0      # background above or below the signal
        for sp in specs:
            fd = sp["run"]["rfsoc"]["fetch_dir"]
            bgq = sp["noise"]
            a = axb if (bgq and split_on) else ax
            a_bg = 1.0 if split_on and bgq else noise_alpha   # a lone panel can be loud
            z = 2.0 if split_on and bgq else z_bg
            traces = []
            for m, col in zip(sp["ev"], sp["colors"]):
                tr = _event_trace(m, fd, n_points, decimation, xlim,
                                  disp_to_raw, raw_to_disp)
                if tr is None:
                    missing.append(f"RID {sp['rid']} {m['npz']}")
                    continue
                if len(tr[0]):
                    traces.append((m, col, tr))
            if not traces:
                continue

            # one summary band for a whole background run, when asked for and
            # when its events actually share a time grid
            if bgq and noise_band and len(traces) > 1 and len({len(t[2][0]) for t in traces}) == 1:
                tb = traces[0][2][0]
                los = np.vstack([_y(t[2][1] if envelope else t[2][3]) for t in traces])
                his = np.vstack([_y(t[2][2] if envelope else t[2][3]) for t in traces])
                mid = np.vstack([_y(t[2][3]) for t in traces])
                blo, bhi = np.nanmin(los, 0), np.nanmax(his, 0)
                a.fill_between(tb, blo, bhi, color=sp["color"],
                               alpha=min(0.9, a_bg * 0.6), lw=0, zorder=z,
                               label=f"{sp['label']} · min–max of {len(traces)} events [bg]")
                for edge in (blo, bhi):        # crisp band edges, not a soft wash
                    a.plot(tb, edge, color=sp["color"], lw=0.9, alpha=a_bg,
                           zorder=z + 0.3)
                a.plot(tb, np.nanmedian(mid, 0), color=sp["color"], lw=2.2,
                       alpha=a_bg, linestyle=noise_linestyle,
                       zorder=z + 0.5, label=f"{sp['label']} · median [bg]")
                continue

            for m, col, (tb, lo, hi, mean) in traces:
                lab = f"{sp['label']} · {_event_label(m['npz'])}" + (" [bg]" if bgq else "")
                if envelope:
                    a.fill_between(tb, _y(lo), _y(hi), color=col,
                                   alpha=(a_bg * .22) if bgq else .20, lw=0,
                                   zorder=z if bgq else 2)
                a.plot(tb, _y(mean), color=col, lw=1.4 if bgq else 1.6,
                       alpha=a_bg if bgq else 1.0,
                       linestyle=noise_linestyle if bgq else linestyle,
                       marker=marker, markersize=4,
                       zorder=(z + 0.5) if bgq else 2.5, label=lab)
        if missing:
            ax.text(0.012, 0.035,
                    f"{len(missing)} capture(s) not fetched\n(rerun with rfsoc_fetch='full')",
                    transform=ax.transAxes, fontsize=9, color="crimson", va="bottom",
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="crimson", alpha=.85))

        for a in (ax, axb) if axb is not None else (ax,):
            h, l = a.get_legend_handles_labels()
            if h:
                a.legend(h, l, loc=legend_loc, fontsize=9.5,
                         ncol=max(1, int(np.ceil(len(h) / 12))))
        if title is None:
            n_ev = sum(len(sp["ev"]) for sp in specs)
            title = (f"DDR4 (decimation={decimation})— {n_ev} event(s) from "
                     + ", ".join(f"{sp['label']}{' [bg]' if sp['noise'] else ''}"
                                 for sp in specs))
            if db:
                title += "   (dB)"
        ax.set_title(title, fontsize=13)
        xlab = f"time in capture window [{_TIME_UNIT_LABEL[time_unit]}]"
        for a in (ax, axb) if axb is not None else (ax,):
            a.set_ylabel("20·log10|I,Q| [dB]" if db else "|I,Q|")
            if xlim is not None:
                a.set_xlim(xlim)
            if ylim is not None:
                a.set_ylim(ylim)
        (axb or ax).set_xlabel(xlab)
    return (ax, axb) if axb is not None else ax


# --------------------------------------------------------------------------- #
#  background-subtracted events
# --------------------------------------------------------------------------- #
def _bg_reference(specs, n_points, decimation, xlim, disp_to_raw, raw_to_disp,
                  stat="mean"):
    """the per-bin background level, averaged over EVERY capture of the
    background runs.

    Returns ``{"t", "y", "sd", "lo", "hi", "n", "label"}`` on the time grid of
    the longest background capture (any capture on a different grid is
    interpolated onto it first), or ``None`` when not one background capture
    could be read. ``stat`` is ``"mean"`` or ``"median"`` — the median is the
    safer reference when a "noise" run caught a real event or two."""
    ts, ys = [], []
    for sp in specs:
        fd = sp["run"]["rfsoc"]["fetch_dir"]
        for m in sp["ev"]:
            tr = _event_trace(m, fd, n_points, decimation, xlim,
                              disp_to_raw, raw_to_disp)
            if tr is None or not len(tr[0]):
                continue
            ts.append(tr[0])
            ys.append(tr[3])                      # per-bin mean |I,Q|
    if not ys:
        return None
    t = max(ts, key=len)
    rows = np.vstack([y if (len(y) == len(t) and np.allclose(tt, t))
                      else np.interp(t, tt, y) for tt, y in zip(ts, ys)])
    mid = np.nanmedian(rows, 0) if stat == "median" else np.nanmean(rows, 0)
    # two useful widths: the shot-to-shot spread of one bin ACROSS captures,
    # and the pooled scatter of every background bin about the reference (the
    # noise floor a single trace actually wanders over)
    return {"t": t, "y": mid, "sd": np.nanstd(rows, 0),
            "sd_pooled": float(np.nanstd(rows - mid)),
            "lo": np.nanmin(rows, 0), "hi": np.nanmax(rows, 0),
            "n": int(rows.shape[0]), "stat": stat,
            "label": " + ".join(sp["label"] for sp in specs)}


def _as_bg(background, t_hint=None):
    """coerce the ``background=`` argument into a reference dict: a scalar (a
    flat level), an ``(t, y)`` pair, a bare per-bin array, or a dict already in
    :func:`_bg_reference` form."""
    if background is None:
        return None
    if isinstance(background, dict):
        return background
    if np.isscalar(background):
        t = np.asarray([-np.inf, np.inf]) if t_hint is None else np.asarray(t_hint, float)
        y = np.full(len(t), float(background))
        return {"t": t, "y": y, "sd": np.zeros(len(t)), "lo": y, "hi": y,
                "n": 0, "stat": "const", "label": f"level {float(background):g}"}
    if isinstance(background, tuple) and len(background) == 2:
        t, y = (np.asarray(a, float) for a in background)
    else:
        y = np.asarray(background, float)
        if t_hint is None:
            raise ValueError("background as a bare array needs a time grid — "
                             "pass background=(t, y) instead")
        t = np.asarray(t_hint, float)
    return {"t": t, "y": y, "sd": np.zeros_like(y), "lo": y, "hi": y,
            "n": 0, "stat": "given", "label": "given background"}


def _subtract(y, b, how):
    """one trace minus its background, in the sense asked for by ``how``."""
    y = np.asarray(y, float)
    b = np.asarray(b, float)
    if how in ("amp", "amplitude", "linear", "lin"):
        return y - b
    if how in ("power", "quad", "rss"):           # incoherent: |s|² = |y|² − |b|²
        return np.sqrt(np.clip(y * y - b * b, 0.0, None))
    if how in ("ratio", "div"):
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(b > 0, y / b, np.nan)
    raise ValueError(f"subtract={how!r} — use 'amp', 'power' or 'ratio'")


_SUB_YLABEL = {
    "amp": ("|I,Q| − background", "20·log10(|I,Q| − bg) [dB]"),
    "power": ("√(|I,Q|² − bg²)", "10·log10(|I,Q|² − bg²) [dB]"),
    "ratio": ("|I,Q| / background", "20·log10(|I,Q| / bg)  [dB above background]"),
}


def plot_ddr4_events_bgsub(runs, noise=None, events=None, background=None,
                           stat="mean", subtract="amp", baseline=None,
                           n_points=3000, decimation=None, db=False, db_ref=1.0,
                           envelope=False, figsize=None, xlim=None, ylim=None,
                           raw_unit="us", time_unit="us", linestyle="-",
                           marker=None, cmap=None, color_by="auto",
                           show_bg=True, bg_sigma="pooled", bg_color="k",
                           bg_band_color="0.45", bg_alpha=0.30, lw=1.2,
                           bg_lw=2.0, big_font=True,
                           zoom=None, zoom_figsize=None, zoom_ylim=None,
                           zoom_aspect=0.75, zoom_height=None, zoom_span=True,
                           zoom_share_y=False,
                           ax=None, legend_loc="best", title=None,
                           return_data=False):
    """DDR4 capture windows with the BACKGROUND MEAN SUBTRACTED — the companion
    to :func:`plot_ddr4_events_multi`, which overlays signal and background
    instead of differencing them.

    The background reference is the per-bin **mean over every capture of the
    runs marked as noise** (``stat="median"`` to use the median instead, which
    survives a stray real event in the noise run). It is interpolated onto each
    signal event's own time grid before subtracting, so background and signal
    runs need not share ``decimation``/window length.

    ``runs``/``noise``/``events`` are exactly as in
    :func:`plot_ddr4_events_multi`: RIDs, ``.h5`` paths, loaded ``run`` dicts,
    ``(rid, events)`` tuples or spec dicts, with the background marked either by
    ``noise=[11330]`` or ``"noise": True`` in a spec. A run named in ``noise``
    that is not itself in ``runs`` is loaded and used as background anyway.

    Where the background comes from — first of these that is given wins:

      * ``background=`` an explicit reference: a scalar (flat level), an
        ``(t, y)`` pair, or the dict this function returns in ``return_data``
        mode — e.g. to reuse ONE reference across several figures;
      * ``baseline=(t0, t1)`` — a quiet time window (in ``time_unit``, on the
        capture axis) of EACH event itself; that event's own mean over the
        window is its background, and the scatter in that window is its σ. Use
        this when there is no noise run, or when a slow drift makes a run-level
        reference stale;
      * the runs marked as noise (the normal case).

    ``subtract`` picks what "minus the background" means:

      * ``"amp"`` (default) — ``|I,Q| − bg``, a plain subtraction of the mean
        level. Honest about the sign: bins below the background go negative.
      * ``"power"`` — ``√(|I,Q|² − bg²)``, the incoherent-sum answer, i.e. the
        amplitude of a signal ADDING in power to an uncorrelated background.
        This is the physically right one when the background is thermal/noise
        rather than a coherent leakage tone; it is clipped at 0.
      * ``"ratio"`` — ``|I,Q| / bg``; with ``db=True`` the y-axis becomes plain
        **dB above the background**, which is usually the most readable form.

    ``db=True`` converts after subtracting (``20·log10`` of the residual, or
    ``10·log10`` of the residual power for ``subtract="power"``); non-positive
    residuals become NaN and simply leave a gap.

    **The background on the plot** (``show_bg=True``): once subtracted the
    reference IS the zero line, so it is drawn as a **black dashed line**
    (``bg_color``/``bg_lw``) with the **±1σ spread of the background captures
    shaded grey** (``bg_band_color``/``bg_alpha``) — anything poking out of the
    grey is bigger than the background's own scatter. The fill is translucent
    and sits UNDER the traces so it hides nothing, while the band edges and the
    mean line are drawn on top as thin dashes so a narrow band stays readable.
    ``bg_sigma`` chooses which width: ``"pooled"`` (default) is the scatter of
    every background bin about the reference — one flat, clean band, the noise
    floor a single trace actually wanders over and the fair yardstick for a
    single-bin spike; ``"captures"`` is instead the shot-to-shot spread of each
    bin across the background captures, which follows any time structure in the
    background at the cost of a fuzzier band. In dB with ``subtract="amp"``/``"power"``
    the negative half has no dB image, so the σ level is drawn as a single grey
    dashed "noise floor" line instead of a band.

    **Style**: ``big_font=True`` (default) renders with
    ``edes.utils.plotting.big_plt_font()``'s rcParams, applied through a local
    ``rc_context`` — the global ``rcParams`` are left untouched. Its 2.5 pt
    default line width buries these dense traces, so the traces are drawn at
    ``lw`` (1.2) and the background line at ``bg_lw`` (2.0); raise ``lw`` for
    sparse/decimated data.

    **Zooming**: ``zoom=(t0, t1)`` (in ``time_unit``) shades that window on the
    main axes (``zoom_span=False`` to leave the main plot untouched) and draws
    it as a **second figure of stacked panels, one event per panel** — no
    inset, which would sit on top of the data. The panels share x and y so
    amplitudes stay comparable, each carries the same background mean line and
    ±1σ band, and each curve keeps the colour it has on the main figure.

    That second figure is sized for a slide: its height matches the main
    figure's (``zoom_height=`` to override) and its width follows
    ``zoom_aspect`` = height/width, 0.75 by default, i.e. a 3:4 PowerPoint-
    friendly block that stands next to the main figure at the same height.
    Panel fonts shrink automatically as panels are added so the labels keep
    fitting; ``zoom_figsize=(w, h)`` overrides the whole calculation.
    Each panel keeps its OWN y scale, so one event's tall spike cannot flatten
    every other panel; ``zoom_share_y=True`` puts them all back on one shared
    scale when the point is to compare amplitudes directly.

    Every other argument (``n_points``, ``decimation``, ``envelope``,
    ``xlim``/``ylim``, ``raw_unit``/``time_unit``, ``linestyle``/``marker``,
    ``cmap``, ``color_by``, ``ax``, ``legend_loc``, ``title``) means what it
    does in :func:`plot_ddr4_events_multi`.

    Returns the axes, or ``(ax, data)`` with ``return_data=True``, where
    ``data`` is ``{"background": <reference dict>, "subtract": …, "events":
    [{rid, event, label, color, t, y, lo, hi, bg, sd}, …],
    "zoom_axes": [panels], "zoom_fig": <figure>}`` — ``y`` is the subtracted
    residual in LINEAR counts (before any dB conversion), for fitting or for
    reuse as a ``background=`` argument elsewhere.

    Example::

        C.plot_ddr4_events_bgsub([11344, 11345], noise=11338, decimation=500,
                                 time_unit="ms", zoom=(15, 20))"""
    import matplotlib.pyplot as plt
    specs = _norm_run_specs(runs, noise=noise, events=events)
    if not specs:
        print("no runs given")
        return
    for sp in specs:
        sp["ev"] = _select_events(sp["run"], sp["events"])
        if not sp["ev"]:
            print(f"RID {sp['rid']}: no DDR4 capture matches"
                  + ("" if sp["events"] is None else f" events={sp['events']}"))
    specs = [sp for sp in specs if sp["ev"]]
    if not specs:
        return

    how = {"amplitude": "amp", "linear": "amp", "lin": "amp", "quad": "power",
           "rss": "power", "div": "ratio"}.get(subtract, subtract)
    if how not in _SUB_YLABEL:
        raise ValueError(f"subtract={subtract!r} — use 'amp', 'power' or 'ratio'")

    def _y(a):
        """counts -> dB (or straight through). ``power`` mode already carries an
        amplitude, so it is 20·log10 of it = 10·log10 of the residual power."""
        a = np.asarray(a, float)
        if not db:
            return a
        ref_db = 1.0 if how == "ratio" else db_ref
        with np.errstate(divide="ignore", invalid="ignore"):
            return 20 * np.log10(np.where(a > 0, a, np.nan) / ref_db)

    raw_to_disp = _unit_scale(raw_unit, time_unit)
    disp_to_raw = _unit_scale(time_unit, raw_unit)
    linestyle, marker = _parse_style(linestyle, marker)

    sig = [sp for sp in specs if not sp["noise"]]
    bgs = [sp for sp in specs if sp["noise"]]
    if not sig:
        print("every run is marked as background — nothing left to subtract from")
        return

    # ---- the reference -------------------------------------------------- #
    ref = _as_bg(background)
    self_baseline = None
    if ref is None and baseline is not None:
        self_baseline = (float(baseline[0]), float(baseline[1]))
        if bgs:
            print("[bgsub] baseline= given — using each event's own window, "
                  "ignoring the noise run(s) as a reference")
    elif ref is None:
        if not bgs:
            print("[bgsub] no background: mark one with noise=<rid>, or pass "
                  "background=<level> / baseline=(t0, t1)")
            return
        ref = _bg_reference(bgs, n_points, decimation, xlim, disp_to_raw,
                            raw_to_disp, stat=stat)
        if ref is None:
            print("[bgsub] the background run(s) have no fetched capture "
                  "(rerun with rfsoc_fetch='full')")
            return

    # ---- colours: same scheme as plot_ddr4_events_multi ------------------ #
    per_event = (color_by == "event" or (color_by == "auto" and len(sig) <= 1))
    if per_event:
        pal = _event_colors(sum(len(sp["ev"]) for sp in sig), cmap)
        i = 0
        for sp in sig:
            sp["colors"] = ([sp["color"]] * len(sp["ev"]) if sp["color"]
                            else pal[i:i + len(sp["ev"])])
            i += len(sp["ev"])
            sp["color"] = sp["colors"][0]
    else:
        for sp, col in zip(sig, _event_colors(len(sig), cmap)):
            sp["color"] = sp["color"] or col
            sp["colors"] = _shades(sp["color"], len(sp["ev"]))

    # ---- background drawing helpers ------------------------------------- #
    zero = (0.0 if how != "ratio" else 1.0)          # the reference, subtracted
    zero_disp = None if (db and how != "ratio") else _y(zero)
    sig_kind = "pooled" if str(bg_sigma).startswith(("pool", "trace")) else "captures"

    def _band(t, sd, lvl):
        """(low, high) edges of the ±1σ background band in display units, or
        ``None`` when dB leaves the lower half undefined."""
        sd = np.asarray(sd, float)
        if how == "ratio":
            lvl = np.where(np.asarray(lvl, float) > 0, lvl, np.nan)
            return _y(np.clip(1 - sd / lvl, 1e-12, None)), _y(1 + sd / lvl)
        if db:
            return None
        return -sd, sd

    def _draw_bg(a, t, sd, lvl, label=False):
        """black dashed background mean + grey ±1σ. The fill sits UNDER the
        traces (translucent, nothing is hidden) while the band edges and the
        mean line are drawn ON TOP as thin dashed lines — otherwise a band
        narrower than the trace-to-trace scatter is simply invisible."""
        if not show_bg:
            return
        band = _band(t, sd, lvl)
        if band is not None:
            # the fill goes ON TOP of the traces, translucent: under them a band
            # this narrow is simply invisible behind the noise
            a.fill_between(t, band[0], band[1], color=bg_band_color,
                           alpha=bg_alpha, lw=0, zorder=2.9,
                           label="background ±1σ" if label else None)
            for edge in band:
                a.plot(t, edge, color=bg_band_color, lw=max(0.8, bg_lw * 0.5),
                       ls=(0, (5, 3)), alpha=.95, zorder=3.0)
        else:      # dB, one-sided: show the σ level as a noise floor instead
            a.plot(t, _y(sd), color=bg_band_color, lw=max(1.0, bg_lw * 0.6),
                   ls=(0, (5, 3)), alpha=.95, zorder=3.0,
                   label="background 1σ" if label else None)
        if zero_disp is not None:
            a.axhline(zero_disp, color=bg_color, lw=bg_lw, ls="--", alpha=.85,
                      zorder=3.2, label="background mean" if label else None)

    out = []
    zaxes, zfig = [], None
    with _rc(big_font=big_font):
        if ax is None:
            fig, ax = plt.subplots(1, 1, constrained_layout=True,
                                   figsize=figsize or (10.5, 5.8))
        missing, t_lo, t_hi = [], np.inf, -np.inf
        for sp in sig:
            fd = sp["run"]["rfsoc"]["fetch_dir"]
            for m, col in zip(sp["ev"], sp["colors"]):
                tr = _event_trace(m, fd, n_points, decimation, xlim,
                                  disp_to_raw, raw_to_disp)
                if tr is None:
                    missing.append(f"RID {sp['rid']} {m['npz']}")
                    continue
                tb, lo, hi, mean = tr
                if not len(tb):
                    continue
                if self_baseline is not None:
                    w = (tb >= self_baseline[0]) & (tb <= self_baseline[1])
                    if not w.any():
                        print(f"[bgsub] RID {sp['rid']} {m['npz']}: baseline "
                              f"window {self_baseline} is outside the capture")
                        continue
                    b = np.full(len(tb), float(np.nanmean(mean[w])))
                    sd = np.full(len(tb), float(np.nanstd(mean[w])))
                else:
                    b = np.interp(tb, ref["t"], ref["y"])
                    sd = np.interp(tb, ref["t"], ref["sd"])
                ys, ylo, yhi = (_subtract(a, b, how) for a in (mean, lo, hi))
                t_lo, t_hi = min(t_lo, tb[0]), max(t_hi, tb[-1])
                lab = f"{sp['label']} · {_event_label(m['npz'])}"
                if envelope:
                    ax.fill_between(tb, _y(ylo), _y(yhi), color=col, alpha=.18,
                                    lw=0, zorder=2)
                ax.plot(tb, _y(ys), color=col, lw=lw, linestyle=linestyle,
                        marker=marker, markersize=4, zorder=2.5, label=lab)
                out.append({"rid": sp["rid"], "event": _event_number(m["npz"]),
                            "label": lab, "color": col, "t": tb, "y": ys,
                            "lo": ylo, "hi": yhi, "bg": b, "sd": sd})
        if missing:
            ax.text(0.012, 0.035,
                    f"{len(missing)} capture(s) not fetched\n(rerun with rfsoc_fetch='full')",
                    transform=ax.transAxes, fontsize=9, color="crimson", va="bottom",
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="crimson", alpha=.85))
        if not out:
            print("[bgsub] nothing to plot")
            return

        # the background, on the grid the events actually cover
        if self_baseline is not None:
            bt = out[0]["t"]
            bsd = np.mean([e["sd"] for e in out], 0)
            blvl = np.mean([e["bg"] for e in out], 0)
        else:
            w = (ref["t"] >= t_lo) & (ref["t"] <= t_hi)
            w = w if w.any() else slice(None)
            bt, blvl = ref["t"][w], ref["y"][w]
            bsd = (np.full(len(blvl), float(ref["sd_pooled"]))
                   if sig_kind == "pooled" and ref.get("sd_pooled") is not None
                   else ref["sd"][w])
            for e in out:                       # the per-event band matches it
                e["sd"] = np.interp(e["t"], bt, bsd)
        _draw_bg(ax, bt, bsd, blvl, label=True)

        # ---- mark the zoom window (the zoomed view is its own figure) --- #
        if zoom is not None and zoom_span:
            z0, z1 = float(zoom[0]), float(zoom[1])
            ax.axvspan(z0, z1, color="0.45", alpha=.10, lw=0, zorder=0.5)
            for zx in (z0, z1):
                ax.axvline(zx, color="0.35", lw=1.0, ls=(0, (4, 3)),
                           alpha=.75, zorder=0.6)
            ax.annotate(f"zoom {z0:g}–{z1:g} {_TIME_UNIT_LABEL[time_unit]}",
                        (0.5 * (z0 + z1), 0.985), xycoords=("data", "axes fraction"),
                        ha="center", va="top", color="0.3",
                        fontsize=max(9, plt.rcParams["legend.fontsize"] - 2),
                        bbox=dict(boxstyle="round,pad=0.18", fc="white",
                                  ec="0.8", alpha=.8))

        h, l = ax.get_legend_handles_labels()
        if h:
            fs = max(9, plt.rcParams["legend.fontsize"] - 1)
            if legend_loc == "best":
                # these traces fill the axes — park the legend under them
                ax.legend(h, l, loc="upper center", bbox_to_anchor=(0.5, -0.16),
                          ncol=min(3, len(h)), frameon=False, fontsize=fs)
            else:
                ax.legend(h, l, loc=legend_loc, fontsize=fs,
                          ncol=max(1, int(np.ceil(len(h) / 8))), framealpha=.9)
        if title is None:
            src = (f"baseline {self_baseline[0]:g}–{self_baseline[1]:g} "
                   f"{_TIME_UNIT_LABEL[time_unit]} of each event"
                   if self_baseline is not None else
                   f"{ref['label']} · {ref['stat']} of {ref['n']} capture(s)"
                   if ref["n"] else ref["label"])
            title = (f"DDR4 background-subtracted ({how}"
                     f"{', decimation=%s' % decimation if decimation else ''}) — "
                     f"{len(out)} event(s) vs {src}")
            if db:
                title += "   (dB)"
        import textwrap
        ax.set_title(textwrap.fill(title, 78))
        ax.set_ylabel(_SUB_YLABEL[how][1 if db else 0])
        ax.set_xlabel(f"time in capture window [{_TIME_UNIT_LABEL[time_unit]}]")
        if xlim is not None:
            ax.set_xlim(xlim)
        if ylim is not None:
            ax.set_ylim(ylim)

    # ---- second figure: the zoom window, one event per panel ----------- #
    # deliberately NOT an inset: an inset that big sits on top of the data.
    # Sized to stand beside the main figure on a slide — same height, 3:4
    # (height/width = zoom_aspect) block.
    if zoom is not None and out:
        z0, z1 = float(zoom[0]), float(zoom[1])
        n = len(out)
        h = float(zoom_height if zoom_height else ax.figure.get_size_inches()[1])
        zsize = zoom_figsize or (h / max(0.2, float(zoom_aspect)), h)
        # shrink the panel fonts as panels are added, so labels keep fitting
        base = dict(STYLE)
        if big_font:
            base.update(_big_font_style())
        scale = float(np.clip((h / n) / 1.9, 0.55, 1.0))
        keys = {"font.size": 14, "axes.titlesize": 15, "axes.labelsize": 14,
                "xtick.labelsize": 12, "ytick.labelsize": 12, "legend.fontsize": 12}
        small = {k: max(7.0, float(base.get(k, d) if isinstance(base.get(k, d), (int, float))
                                   else d) * scale) for k, d in keys.items()}
        with _rc(extra=small, big_font=big_font):
            zfig, axs = plt.subplots(n, 1, sharex=True, sharey=bool(zoom_share_y),
                                     constrained_layout=True, figsize=zsize)
            zfig.get_layout_engine().set(h_pad=0.02, hspace=0.03)
            zaxes = list(np.atleast_1d(axs))
            for i, (a, e) in enumerate(zip(zaxes, out)):
                _draw_bg(a, bt, bsd, blvl, label=(i == 0))
                a.plot(e["t"], _y(e["y"]), color=e["color"], lw=max(lw, 1.0),
                       linestyle=linestyle, zorder=2.5)
                a.set_xlim(z0, z1)
                a.yaxis.set_major_locator(plt.MaxNLocator(4))
                a.text(0.988, 0.94, e["label"], transform=a.transAxes,
                       ha="right", va="top", color=e["color"],
                       fontsize=small["legend.fontsize"], fontweight="bold",
                       bbox=dict(boxstyle="round,pad=0.22", fc="white",
                                 ec="0.8", alpha=.85))
                if i == 0:
                    hh, ll = a.get_legend_handles_labels()
                    if hh:
                        a.legend(hh, ll, loc="upper left", ncol=2, framealpha=.85,
                                 fontsize=small["legend.fontsize"] * 0.95,
                                 handlelength=1.4, borderpad=0.25,
                                 labelspacing=0.25, columnspacing=1.0)
            if zoom_ylim is not None:
                zaxes[0].set_ylim(*zoom_ylim)
            else:
                # shared y: one scale over every panel; independent y: each
                # panel scaled to its own event, so a small one isn't flattened
                groups = ([(zaxes[0], out)] if zoom_share_y
                          else list(zip(zaxes, ([e] for e in out))))
                for a, es in groups:
                    v = np.concatenate([_y(e["y"][(e["t"] >= z0) & (e["t"] <= z1)])
                                        for e in es])
                    v = v[np.isfinite(v)]
                    if v.size:
                        span = (v.max() - v.min()) or 1.0
                        a.set_ylim(v.min() - 0.10 * span, v.max() + 0.28 * span)
            zaxes[-1].set_xlabel(f"time [{_TIME_UNIT_LABEL[time_unit]}]")
            lab_y = _SUB_YLABEL[how][1 if db else 0]
            try:
                zfig.supylabel(lab_y, fontsize=small["axes.labelsize"])
            except AttributeError:                 # matplotlib < 3.4
                zaxes[len(zaxes) // 2].set_ylabel(lab_y)
            zfig.suptitle(f"zoom {z0:g}–{z1:g} {_TIME_UNIT_LABEL[time_unit]}"
                          f"  ·  background-subtracted ({how})",
                          fontsize=small["axes.titlesize"])

    if return_data:
        return ax, {"background": ref, "baseline": self_baseline,
                    "subtract": how, "events": out, "zoom": zoom,
                    "zoom_axes": zaxes, "zoom_fig": zfig}
    return ax


def _sa_flat(run):
    """flat list of SA traces: {rep, k, wall, wall_exact, trace, x}. ``wall`` is
    the exact per-trace time when available, else the stretched estimate."""
    out, x = [], run["sa_x"]
    swt = run["sa_swt"]
    for rp in run["sa_reps"]:
        xx = x if x.size == rp["traces"].shape[1] else np.arange(rp["traces"].shape[1])
        exact = rp["wall"] is not None and np.all(np.asarray(rp["wall"]) > 1e9)
        has_start = rp["wall_start"] is not None and np.all(np.asarray(rp["wall_start"]) > 1e9)
        est = None if exact else _sa_wall_fallback(run, rp)
        for k in range(rp["traces"].shape[0]):
            w = float(rp["wall"][k]) if exact else (
                float(est[k]) if est is not None and k < len(est) else None)
            w0 = float(rp["wall_start"][k]) if has_start else None
            # the sweep (duration SSA_SWT) sits somewhere in [w0, w]; the leftover
            # time is overhead we can't place (SCPI writes before + *OPC?/transfer
            # after). end-start >= ~2*SSA_SWT means :AVERage ran multiple sweeps.
            overhead = (w - w0 - swt) if (w is not None and w0 is not None) else None
            multi = (w0 is not None and w is not None and (w - w0) > 1.9 * swt)
            out.append({"rep": rp["rep"], "k": k, "trace": rp["traces"][k], "x": xx,
                        "wall": w, "wall_exact": bool(exact), "wall_start": w0,
                        "overhead_s": overhead, "multi_sweep": multi})
    return out


def _ddr4_list(run):
    return [m for m in (run["rfsoc"].get("manifest") or [])
            if m.get("capture") == "ok" and m.get("npz")]


def plot_sa_vs_ddr4(run, sa=0, event=-1, which="mean", n_points=4000,
                    shift_ms=0.0, sa_latency_ms=0.0, sa_anchor="end", sep_axis=False,
                    db_ref="auto", xlim_ms=None, ax=None):
    """Overlay ONE SA trace and ONE DDR4 capture on a shared time axis.

    **Anchoring** (the plot picks automatically):
      * both stamps (``sa_wall_start_*`` + ``sa_wall_*``) → the SSA_SWT sweep sits
        somewhere in ``[start, end - SSA_SWT]``; the SA trace is anchored at the
        MIDPOINT of that window and the residual uncertainty is ``±overhead/2``
        (printed on the plot). On this bench the overhead is large (~80 ms →
        ±40 ms) because ``*OPC?``/transfer dominate — a hardware/level trigger is
        the only way to beat it. ``:AVERage`` over multiple sweeps is flagged.
      * only ``sa_wall_*`` (end stamp) → ``end - SSA_SWT``; add the ~10–40 ms lag
        with ``sa_latency_ms=``.
      * neither (pre-patch) → SA time is a ±10 s estimate; align by eye with
        ``shift_ms`` / ``sa_anchor=<wall>``.
    The SA sweep is *not signal-triggered*, so a burst lands at an arbitrary
    phase of the window — the RFSoC crossing (x=0, manifest ``t_wall``) is the
    reliable timestamp.

    x = ms relative to the DDR4 crossing.
      shift_ms       slide the DDR4 window
      sa_latency_ms  shift the SA trace EARLIER (manual nudge)
      sa_anchor      "start" is used automatically when available; else
                     "end" (default) | "start" | <wall float>

    y: SA in dBm; DDR4 |I,Q| as ``20·log10|I,Q|`` shifted (``db_ref="auto"``
    median-match) to overlay, or its own axis with ``sep_axis=True``.
    """
    import matplotlib.pyplot as plt
    sas, evs = _sa_flat(run), _ddr4_list(run)
    if not sas or not evs:
        print("need at least one SA trace and one DDR4 event"); return
    st = sas[sa % len(sas)]
    m = evs[event % len(evs)]
    p = os.path.join(run["rfsoc"]["fetch_dir"], m["npz"])
    if not os.path.exists(p):
        print(f"{m['npz']} not fetched — rerun with rfsoc_fetch='full'"); return

    d = np.load(p)
    iq = d["iq"].astype(np.float32)
    mag = np.hypot(iq[:, 0], iq[:, 1])
    fs = float(d["fs_msps"]) if "fs_msps" in getattr(d, "files", []) else F_OUT_MHZ
    tb, lo, hi, mean = _envelope(mag, np.arange(len(mag)) / fs, n_points)
    prim = {"mean": mean, "peak": hi}[which]
    with np.errstate(divide="ignore", invalid="ignore"):
        to_db = lambda a: 20 * np.log10(np.where(np.asarray(a) > 0, a, np.nan))
        prim_db, lo_db, hi_db = to_db(prim), to_db(lo), to_db(hi)
    ddr4_x_ms = tb / 1e3 + shift_ms

    cross_wall = float(m.get("t_wall") or 0.0)
    swt = run["sa_swt"]
    x_s = np.asarray(st["x"], float)
    win_lo = win_hi = None                       # sweep-start uncertainty window (rel x, ms)
    if isinstance(sa_anchor, (int, float)) and not isinstance(sa_anchor, bool):
        sa_t0, anchor_kind = float(sa_anchor), "manual"
    elif st.get("wall_start") is not None and st["wall_exact"]:
        # the 100 ms sweep sits somewhere in [wall_start, wall_end - SSA_SWT];
        # anchor on the MIDPOINT, and the residual uncertainty is +/- overhead/2
        lo, hi = st["wall_start"], st["wall"] - swt
        sa_t0, anchor_kind = 0.5 * (lo + hi), "window"
        win_lo, win_hi = (lo - cross_wall) * 1e3, (hi - cross_wall) * 1e3
    elif st["wall_exact"]:                       # only the end stamp: sweep end - SSA_SWT
        sa_t0 = st["wall"] - (swt if sa_anchor == "end" else 0.0)
        anchor_kind = "end"
    else:                                        # estimate is +/-10 s -> useless on a ms
        sa_t0, anchor_kind = cross_wall - swt / 2.0, "estimate"
    sa_t0 -= sa_latency_ms / 1e3                 # manual nudge
    sa_x_ms = (sa_t0 + x_s - cross_wall) * 1e3

    with _rc():
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 5.2), constrained_layout=True)
        fc = run["args"].get("SSA_freq_center", 0) or 0
        ax.plot(sa_x_ms, st["trace"], color="tab:green", lw=1.8,
                label=f"SA rep{st['rep']}  (zero-span @ {fc/1e6:.2f} MHz)")
        if np.isfinite(run["p_thresh"]):
            ax.axhline(run["p_thresh"], color="tab:green", ls=":", lw=1.3, alpha=.7)

        if sep_axis:
            axd = ax.twinx(); axd.grid(False); off = 0.0
        else:
            axd = ax
            if db_ref == "auto":
                off = float(np.nanmedian(st["trace"]) - np.nanmedian(prim_db))
            elif db_ref in (None, "none"):
                off = 0.0
            else:
                off = -20 * np.log10(float(db_ref))
        axd.fill_between(ddr4_x_ms, lo_db + off, hi_db + off, color="tab:blue",
                         alpha=.25, lw=0, label="DDR4 min–max")
        axd.plot(ddr4_x_ms, prim_db + off, color="tab:blue", lw=1.8,
                 label=f"DDR4 {which}  20·log10|I,Q|"
                       + ("" if sep_axis or not off else f"  ({off:+.0f} dB)"))
        if sep_axis:
            axd.set_ylabel("DDR4  20·log10|I,Q|  [dB, arb.]", color="tab:blue")
            axd.tick_params(axis="y", colors="tab:blue")
            finite = prim_db[np.isfinite(prim_db)]
            if finite.size:
                lo_f = lo_db[np.isfinite(lo_db)]; hi_f = hi_db[np.isfinite(hi_db)]
                y0 = (lo_f.min() if lo_f.size else finite.min())
                y1 = (hi_f.max() if hi_f.size else finite.max())
                axd.set_ylim(y0 - 0.15 * (y1 - y0) - 1, y1 + 0.15 * (y1 - y0) + 1)

        ax.axvline(0.0, color="crimson", lw=1.4, alpha=.85)
        ax.annotate("crossing", (0, 1), xycoords=("data", "axes fraction"),
                    xytext=(4, -13), textcoords="offset points", color="crimson", fontsize=10)
        spec = run["rfsoc"].get("rfsoc_pulse_spec") or {}
        cfg = (run["rfsoc"].get("rfsoc_status") or {}).get("config", {})
        plen = float(spec.get("pulse_on_us") or cfg.get("pulse_on_us")
                     or cfg.get("pulse_length_us") or 0.0)
        if plen > 0:
            ax.axvspan(shift_ms, plen / 1e3 + shift_ms, color="crimson", alpha=.06,
                       label=f"feedback pulse ({plen:.0f} µs)")

        # shade the SA sweep span, and (window anchor) the sweep-start uncertainty
        ax.axvspan(sa_x_ms[0], sa_x_ms[-1], color="tab:green", alpha=.05, zorder=0)
        if win_lo is not None:
            ax.axvspan(win_lo, win_hi, color="0.5", alpha=.10, zorder=0)
            ax.axvspan(win_lo + swt * 1e3, win_hi + swt * 1e3, color="0.5", alpha=.10, zorder=0)

        ax.set_xlabel("time relative to the DDR4 crossing [ms]")
        ax.set_ylabel("SA power [dBm]" + ("" if sep_axis else "   /   DDR4 dB (shifted)"))
        ax.set_title(f"RID {run['rid']}:  SA rep{st['rep']}  ↔  {m['npz']}")
        ov_ms = None if st.get("overhead_s") is None else st["overhead_s"] * 1e3
        if anchor_kind == "estimate":
            note = ("SA time unknown (±~10 s) — window centred on the crossing;\n"
                    "align features with shift_ms= / sa_anchor=<wall>")
            col = "crimson"
        elif anchor_kind == "window":
            note = (f"SA anchored at the MIDPOINT of its acquisition window;\n"
                    f"the sweep could start anywhere in the grey band → **±{ov_ms/2:.0f} ms**.\n"
                    f"(readout overhead {ov_ms:.0f} ms — mostly *OPC?/transfer.)\n"
                    f"Nudge with sa_latency_ms= if you can calibrate it.")
            col = "crimson" if ov_ms and ov_ms > 30 else "0.35"
            if st.get("multi_sweep"):
                note += "\n⚠ :AVERage ran multiple sweeps — trace is NOT one snapshot"
        else:  # end anchor
            note = ("anchored on the SA sweep-END stamp — includes the\n"
                    "~10–40 ms *OPC?/transfer lag. Re-run for sa_wall_start_*,\n"
                    "or dial it out with sa_latency_ms=.")
            col = "crimson"
        ax.text(0.012, 0.035, note, transform=ax.transAxes, fontsize=9, color=col,
                va="bottom", bbox=dict(boxstyle="round,pad=0.3", fc="white",
                                       ec=col, alpha=.85))
        if xlim_ms:
            ax.set_xlim(*xlim_ms)
        h, l = [], []
        for a in ([ax, axd] if sep_axis else [ax]):
            hh, ll = a.get_legend_handles_labels(); h += hh; l += ll
        ax.legend(h, l, loc="upper center", bbox_to_anchor=(0.5, -0.13),
                  ncol=2, frameon=False, fontsize=10)
    return ax


def analyze(rid):
    run = load_run(rid)
    summary(run)
    plot_timeline(run)
    plot_sa_traces(run)
    plot_ddr4_events(run)
    if _sa_flat(run) and _ddr4_list(run):
        plot_sa_vs_ddr4(run)
    return run
