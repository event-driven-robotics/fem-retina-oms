import numpy as np
import cv2

def compute_object_selectivity(
    spikes, mask, *, ksize=39, min_visibility=0.98,
    exclude_edge_touch=True, return_more=False, vicinity_method="box"
):
    """
    Object selectivity = fraction of spikes that fall within the object's 'vicinity'
    (mask dilated with a square of side `ksize`), pooled over frames where the object
    is fully visible (>= min_visibility of *max non-edge* area when exclude_edge_touch=True,
    otherwise max over all frames) and, optionally, not touching edges.
    With vicinity_method='exact', use the nonzero mask pixels directly without
    expansion; ksize does not affect that scoring region.

    Returns:
        p (float) or (p, info_dict)
        info_dict keys: p_chance, enrichment, N_obj, N_bg, N_tot, r_obj, r_bg,
                        density_ratio, selectivity_index, kept_frames, A_max, A_obj, A_tot, area_frac
    """
    if spikes.shape != mask.shape or spikes.ndim != 3:
        raise ValueError("spikes and mask must have the same shape (T,H,W).")
    if vicinity_method not in {"box", "dilate", "exact"}:
        raise ValueError(f"Unsupported vicinity_method: {vicinity_method}")

    T, H, W = spikes.shape

    # --- normalize ksize to odd (kh, kw) ---
    if isinstance(ksize, int):
        if ksize < 1:
            raise ValueError("ksize must be >= 1.")
        kh = kw = ksize + (ksize % 2 == 0)
    else:
        if len(ksize) != 2 or min(ksize) < 1:
            raise ValueError("ksize must be int or (kh, kw) with both >= 1.")
        kh = int(ksize[0] + (ksize[0] % 2 == 0))
        kw = int(ksize[1] + (ksize[1] % 2 == 0))
    ktuple = (kw, kh)  # OpenCV uses (width, height)

    # --- force boolean mask early (fixes 0/255 cases) ---
    mask_bool = (mask > 0)

    # --- detect edge touch per frame (on boolean) ---
    touch = np.zeros(T, dtype=bool)
    for t in range(T):
        Mt = mask_bool[t]
        if Mt.any():
            touch[t] = Mt[0, :].any() or Mt[-1, :].any() or Mt[:, 0].any() or Mt[:, -1].any()

    # --- areas per frame (in pixels) ---
    A_t = mask_bool.reshape(T, -1).sum(axis=1).astype(np.int64)

    # --- choose the right A_max for visibility thresholding ---
    if exclude_edge_touch:
        # Use the maximum area among NON-edge-touching frames if available,
        # otherwise fall back to the global max.
        if (~touch).any():
            A_max = int(A_t[~touch].max())
        else:
            A_max = int(A_t.max())
    else:
        A_max = int(A_t.max())

    # quick exit if mask empty everywhere
    if A_max == 0:
        out = {"p_chance": np.nan, "enrichment": np.nan, "N_obj": 0.0, "N_bg": 0.0,
               "N_tot": 0.0, "r_obj": np.nan, "r_bg": np.nan, "density_ratio": np.nan,
               "selectivity_index": np.nan, "kept_frames": 0, "A_max": 0,
               "A_obj": 0, "A_tot": 0, "area_frac": np.nan}
        return (np.nan, out) if return_more else np.nan

    # --- build vicinity per-frame ---
    M_vic = np.empty_like(mask_bool, dtype=bool)

    if vicinity_method == "exact":
        M_vic = mask_bool
    elif vicinity_method == "box":
        m = mask_bool.astype(np.float32)
        # loop over T keeps mem down; vectorization is possible but often not worth it
        for t in range(T):
            M_vic[t] = cv2.blur(m[t], ktuple, borderType=cv2.BORDER_REFLECT) > 0.0

    elif vicinity_method == "dilate":
        kernel = np.ones((kh, kw), np.uint8)
        for t in range(T):
            # cv2.dilate expects uint8
            M_vic[t] = cv2.dilate(mask_bool[t].astype(np.uint8), kernel, borderType=cv2.BORDER_CONSTANT) > 0
    else:
        raise ValueError(f"Unsupported vicinity_method: {vicinity_method}")

    # --- select frames: fully in view (+ optionally not touching borders) ---
    fully = A_t >= (min_visibility * A_max)
    keep = fully & (~touch) if exclude_edge_touch else fully

    kept_frames = int(keep.sum())
    if kept_frames == 0:
        out = {"p_chance": np.nan, "enrichment": np.nan, "N_obj": 0.0, "N_bg": 0.0,
               "N_tot": 0.0, "r_obj": np.nan, "r_bg": np.nan, "density_ratio": np.nan,
               "selectivity_index": np.nan, "kept_frames": 0, "A_max": A_max,
               "A_obj": 0, "A_tot": 0, "area_frac": np.nan}
        return (np.nan, out) if return_more else np.nan

    # --- pool counts on kept frames ---
    S = spikes[keep].astype(np.float64, copy=False)
    # Optional: treat NaNs as 0 spikes (robustness)
    if np.isnan(S).any():
        S = np.nan_to_num(S, copy=False)

    V = M_vic[keep]

    N_obj = float(S[V].sum())
    N_bg  = float(S[~V].sum())
    N_tot = N_obj + N_bg
    p = np.nan if N_tot == 0.0 else float(N_obj / N_tot)

    if not return_more:
        return p

    # --- areas over kept frames ---
    # (vectorized per-frame counts/rates for framewise SI)
    K = S.shape[0]                          # number of kept frames
    eps = 1e-12

    A_obj_t = V.reshape(K, -1).sum(axis=1).astype(np.float64)         # pixels in obj vicinity per frame
    A_bg_t  = (~V).reshape(K, -1).sum(axis=1).astype(np.float64)      # pixels in background per frame

    # per-frame spike counts
    N_obj_t = (S * V).reshape(K, -1).sum(axis=1)
    N_bg_t  = (S * (~V)).reshape(K, -1).sum(axis=1)

    # per-frame area-normalized rates
    r_obj_t = N_obj_t / np.maximum(A_obj_t, 1.0)
    r_bg_t  = N_bg_t  / np.maximum(A_bg_t,  1.0)

    # per-frame selectivity index (bounded in [-1,1])
    denom_t = r_obj_t + r_bg_t
    with np.errstate(divide='ignore', invalid='ignore'):
        si_frame = (r_obj_t - r_bg_t) / np.where(denom_t > 0, denom_t, np.nan)

    # robust aggregate (penalizes temporal sparsity)
    si_median = np.nanmedian(si_frame) if np.any(np.isfinite(si_frame)) else np.nan
    si_mean   = np.nanmean(si_frame)   if np.any(np.isfinite(si_frame)) else np.nan


    # chance baseline over the same subset of pixels
    A_obj = int(V.sum())
    A_tot = int(V.size)
    A_bg  = A_tot - A_obj

    # rates (guard zeros)
    r_obj = (N_obj / A_obj) if A_obj > 0 else np.nan
    r_bg  = (N_bg  / A_bg)  if A_bg  > 0 else np.nan

    density_ratio = (r_obj / r_bg) if (np.isfinite(r_obj) and np.isfinite(r_bg) and r_bg > 0) else np.nan
    selectivity_index = ((r_obj - r_bg) / (r_obj + r_bg)) if (np.isfinite(r_obj) and np.isfinite(r_bg) and (r_obj + r_bg) > 0) else np.nan

    p_chance = (A_obj / A_tot) if A_tot > 0 else np.nan
    enrichment = (p / p_chance) if (p_chance > 0 and np.isfinite(p)) else np.nan
    area_frac = (A_obj / A_tot) if A_tot > 0 else np.nan  # same as p_chance; included for clarity

    return {
        "p": p,
        "p_chance": float(p_chance) if np.isfinite(p_chance) else np.nan,
        "enrichment": float(enrichment) if np.isfinite(enrichment) else np.nan,
        "N_obj": N_obj,
        "N_bg":  N_bg,
        "N_tot": N_tot,
        "r_obj": float(r_obj) if np.isfinite(r_obj) else np.nan,
        "r_bg":  float(r_bg)  if np.isfinite(r_bg)  else np.nan,
        "density_ratio": float(density_ratio) if np.isfinite(density_ratio) else np.nan,
        "selectivity_index": float(selectivity_index) if np.isfinite(selectivity_index) else np.nan,
        "kept_frames": kept_frames,
        "A_max": A_max,
        "A_obj": A_obj,
        "A_tot": A_tot,
        "area_frac": float(area_frac) if np.isfinite(area_frac) else np.nan,
        "si_median": float(si_median) if np.isfinite(si_median) else np.nan,
        "si_mean": float(si_mean) if np.isfinite(si_mean) else np.nan,
    }
