import numpy as np

def detect_microsaccades_engbert(
    x, y, fs,
    lam=6.0,                 # lambda (multiplier for adaptive threshold)
    min_dur_ms=6.0,          # minimum event duration
    merge_gap_ms=10.0        # merge events separated by < this gap
):

    """
    Engbert & Kliegl (2003) microsaccade detection.
    x, y : 1D arrays of gaze samples (pixels)
    fs   : sampling rate (Hz), e.g., 200
    Returns dict with:
      - 'saccades': list of dicts (start, end, duration_ms, amplitude_px, peak_speed)
      - 'saccade_mask': boolean mask over samples (True = saccade)
      - 'jitter_mask': boolean mask over samples (True = non-saccadic)
    """

    x = np.asarray(x).astype(float)
    y = np.asarray(y).astype(float)
    assert x.ndim == y.ndim == 1 and x.size == y.size
    N = x.size
    dt = 1.0 / float(fs)

    # 1) velocities (length N-1)
    vx = np.diff(x) / dt
    vy = np.diff(y) / dt

    # 2) robust dispersion (median-based SD as in Engbert)
    def robust_sd(v):
        med = np.median(v)
        return np.sqrt(np.median((v - med) ** 2))

    sigx = robust_sd(vx)
    sigy = robust_sd(vy)
    # avoid divide-by-zero if data are constant
    eps = 1e-12
    tx = lam * max(sigx, eps)
    ty = lam * max(sigy, eps)

    # 3) elliptical 2D threshold in velocity space (over steps)
    above = (vx / tx) ** 2 + (vy / ty) ** 2 > 1.0  # bool, length N-1

    # Helper: contiguous runs in a boolean vector
    def boolean_runs(b):
        if b.size == 0:
            return []
        diff = np.diff(b.astype(int))
        starts = np.flatnonzero(np.r_[b[0], diff == 1])
        ends   = np.flatnonzero(np.r_[diff == -1, b[-1]])  # inclusive ends (index in b)
        return list(zip(starts, ends))

    # initial runs in step-domain
    step_runs = boolean_runs(above)  # each run is on indices of 'above' (0..N-2)

    # 4) merge runs that are separated by small gaps (in time)
    if step_runs:
        merged = [list(step_runs[0])]
        max_gap_steps = int(np.round((merge_gap_ms / 1000.0) / dt))  # gap in steps
        for s, e in step_runs[1:]:
            prev_s, prev_e = merged[-1]
            gap = s - prev_e - 1  # steps between runs
            if gap <= max_gap_steps:
                merged[-1][1] = e  # extend previous
            else:
                merged.append([s, e])
        step_runs = [tuple(r) for r in merged]

    # 5) duration filter & compute stats; map to sample indices
    min_len_steps = max(1, int(np.ceil((min_dur_ms / 1000.0) / dt)))
    saccade_mask = np.zeros(N, dtype=bool)
    events = []
    for s_step, e_step in step_runs:
        length_steps = e_step - s_step + 1
        if length_steps < min_len_steps:
            continue
        # steps are between samples i and i+1
        start_samp = s_step
        end_samp   = e_step + 1
        start_samp = max(0, start_samp)
        end_samp   = min(N - 1, end_samp)

        # mark samples in [start_samp, end_samp]
        saccade_mask[start_samp:end_samp + 1] = True

        # stats
        duration_ms = (end_samp - start_samp) * dt * 1000.0
        amp = np.hypot(x[end_samp] - x[start_samp], y[end_samp] - y[start_samp])

        # peak speed from the velocity steps that belong to this event
        sl = slice(s_step, e_step + 1)  # in velocity domain
        peak_speed = np.max(np.hypot(vx[sl], vy[sl])) if sl.stop > sl.start else 0.0

        events.append(dict(
            start_idx=int(start_samp),
            end_idx=int(end_samp),
            duration_ms=float(duration_ms),
            amplitude_px=float(amp),
            peak_speed=float(peak_speed)
        ))

    jitter_mask = ~saccade_mask
    return {
        "saccades": events,
        "saccade_mask": saccade_mask,
        "jitter_mask": jitter_mask
    }

def stitch_by_increments(x, y, keep_mask, fs, carry_over=True, detrend_segment=False):
    """
    If carry_over=True  -> single long random walk (what you saw).
    If carry_over=False -> each drift segment starts at 0, so the overall
                           path stays bounded; good for per-segment stats.
    detrend_segment=True removes the mean velocity per segment (optional).
    """
    x = np.asarray(x, float); y = np.asarray(y, float)
    keep = np.asarray(keep_mask, bool)

    b = keep.astype(int)
    diff = np.diff(b, prepend=0, append=0)
    starts = np.flatnonzero(diff == 1)
    ends   = np.flatnonzero(diff == -1) - 1

    dt = 1.0/fs
    xs, ys, ts = [], [], []
    cur_x = cur_y = 0.0
    t_cursor = 0.0

    for s, e in zip(starts, ends):
        if e <= s:
            continue
        dx = np.diff(x[s:e+1])
        dy = np.diff(y[s:e+1])

        if detrend_segment:
            # remove mean velocity within this segment
            dx = dx - dx.mean()
            dy = dy - dy.mean()

        seg_x = np.r_[0.0, np.cumsum(dx)]
        seg_y = np.r_[0.0, np.cumsum(dy)]
        seg_t = t_cursor + np.arange(e - s + 1) * dt

        if carry_over:
            seg_x = cur_x + seg_x
            seg_y = cur_y + seg_y
            cur_x, cur_y = seg_x[-1], seg_y[-1]
        else:
            # start every segment at (0,0); don't carry displacement across segments
            cur_x = cur_y = 0.0

        xs.append(seg_x); ys.append(seg_y); ts.append(seg_t)
        t_cursor = seg_t[-1] + dt

    if not xs:
        return np.array([]), np.array([]), np.array([])

    return np.concatenate(xs), np.concatenate(ys), np.concatenate(ts)

def circular(n_frames, radius, pixels_per_frame, center=(0.0, 0.0)):
    cx, cy = center
    # angular step that yields the desired arc-length per frame
    dtheta = pixels_per_frame / radius   # radians
    c, s = np.cos(dtheta), np.sin(dtheta)

    # start at angle 0 (point on +x axis)
    x, y = radius, 0.0
    Xe = np.empty(n_frames, dtype=float)
    Ye = np.empty(n_frames, dtype=float)

    for t in range(n_frames):
        Xe[t] = cx + x
        Ye[t] = cy + y
        # rotate (x, y) by fixed dθ
        x, y = c*x - s*y, s*x + c*y

    return Xe, Ye

def sine_velocity_trajectory(
    number_frames,
    v_base, v_amp, velocity_cycles, v_phase=0.0, v_min=0.0,
    orientation_cycles=1.0, theta0=0.0,
    orient_amp=0.0, orient_cycles=None, orient_phase=0.0,
):
    """
    Returns Xe, Ye (pixels) of length number_frames.
    - Speed per frame: v(t) = v_base + v_amp * sin(2π * velocity_cycles * t/N + v_phase)
      (clipped to >= v_min)
    - Heading (direction):
        theta(t) = theta0 + 2π * orientation_cycles * t/N            # constant angular rate
                    + orient_amp * sin(2π * (orient_cycles or orientation_cycles) * t/N + orient_phase)
                      # optional heading-rate modulation
    Units: pixels/frame.
    """

    N = number_frames
    t = np.arange(N, dtype=float)

    # speed profile (pixels/frame)
    theta_vel = 2*np.pi * velocity_cycles * t / N + v_phase
    v = v_base + v_amp * np.sin(theta_vel)
    if v_min is not None:
        v = np.clip(v, v_min, None)  # ensure non-negative steps

    # heading profile
    if orient_cycles is None:
        orient_cycles = orientation_cycles
    theta = (
        theta0
        + 2*np.pi * orientation_cycles * t / N
        + orient_amp * np.sin(2*np.pi * orient_cycles * t / N + orient_phase)
    )

    # per-frame displacement (so |[dx,dy]| = v(t))
    dx = v * np.cos(theta)
    dy = v * np.sin(theta)

    # integrate to positions
    Xe = np.cumsum(dx)
    Ye = np.cumsum(dy)

    return Xe, Ye
