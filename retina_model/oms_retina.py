import numpy as np
import cv2
from scipy.special import expit
from common_utils.memory import release_pages


# @profile
def normalize_bipolar_responces(
    binned_events, ksize, dt, tau_bp, tau_am, tau_var,
    squash_scale=0.5, eps=1e-3, return_only_normalized=False, output_path=None
):
    """
    Single-pass online normalization with a constant epsilon (no warm-up, no debiasing).

    Args
    ----
    binned_events : (T,P,H,W) float array  where P=1 if polarity_filtering=False else P=2
    dt            : seconds per bin
    tau_bp        : EMA time constant for 'fast' (bipolar-like)
    tau_am        : EMA time constant for 'slow' (amacrine-like)
    tau_var       : EMA time constant for variance of (fast - slow)
    eps           : small constant added to the divisor to prevent blow-ups
    output_path   : optional .npy file for bounded-memory recording; requires
                    return_only_normalized=True and returns a memory-mapped array

    Returns
    -------
    bp_rec, am_rec, scale_rec, C_rec, R_rec : all (T,H,W) float32
    """

    if output_path is not None and not return_only_normalized:
        raise ValueError('Disk-backed output requires return_only_normalized=True')
    if binned_events.ndim == 4:
        assert binned_events.shape[1] in (1, 2)


    beta_bp = np.exp(-dt / tau_bp);  alpha_bp = 1.0 - beta_bp
    beta_am = np.exp(-dt / tau_am);  alpha_am = 1.0 - beta_am
    beta_var = np.exp(-dt / tau_var); alpha_var = 1.0 - beta_var

    T, H, W = binned_events.shape[0], *binned_events.shape[-2:]
    fast = np.zeros((H, W), np.float32)
    slow = np.zeros((H, W), np.float32)
    var  = np.zeros((H, W), np.float32)  # EMA of (fast - slow)^2

    if output_path is not None:
        R_rec = np.lib.format.open_memmap(output_path, mode='w+', dtype=np.float32, shape=(T, H, W))
    elif return_only_normalized:
        R_rec = []
    else:
        bp_rec, am_rec, scale_rec, C_rec, R_rec = [], [], [], [], []

    for i in range(T):
        # syn   = cv2.blur(binned_events[i].astype(np.float32), (39, 39))
        frame = binned_events[i]
        if frame.ndim == 3:
            frame = frame[0] + frame[1] if len(frame) == 2 else frame[0]
        syn   = cv2.blur(frame.astype(np.float32), (ksize, ksize))

        # fast/slow EMAs
        fast  = beta_bp * fast + alpha_bp * syn
        slow  = beta_am * slow + alpha_am * syn

        # variability estimate
        dev   = fast - slow
        var   = beta_var * var + alpha_var * (dev * dev)
        scale = np.sqrt(var + 1e-12)

        # divisive normalization
        C = dev / (eps + scale)

        ### use sigmoid nonlinearity
        R = expit(C).astype(np.float32)  # 0..1 ### squashing nonlinearity

        if not return_only_normalized:
            bp_rec.append(fast.copy())
            am_rec.append(slow.copy())
            scale_rec.append(scale.copy())
            C_rec.append(C.astype(np.float32))
        if output_path is None:
            R_rec.append(R)
        else:
            R_rec[i] = R
            if i % 32 == 31:
                release_pages(R_rec)
                release_pages(binned_events)

    if output_path is not None:
        release_pages(R_rec)
        return R_rec
    normalized_bp_rec     = np.stack(R_rec,     axis=0)

    if return_only_normalized:
        return normalized_bp_rec

    bp_rec    = np.stack(bp_rec,    axis=0)
    am_rec    = np.stack(am_rec,    axis=0)
    scale_rec = np.stack(scale_rec, axis=0)
    C_rec     = np.stack(C_rec,     axis=0)

    return bp_rec, am_rec, scale_rec, C_rec, normalized_bp_rec



def spike_fn(x: np.ndarray) -> np.ndarray:
    out = np.zeros_like(x, dtype=np.float32)
    out[x > 0] = 1.0
    return out


def run_oms_retina_(
    bp_out_rec,                 # (T,H,W) normalized bipolar output
    threshold=0.3,
    dt=5e-3,
    tau_syn=5e-3,               # EMA time-const for synapses (amacrine & OMS)
    tau_mem=5e-3,               # EMA time-const for OMS membrane
    rf_ksize=(5,5),             # **shared** local RF for both amacrine & bipolar syn
    reduce_op="mean",           # "mean" or "median" for wide-field pooling
    record=True,
    no_supression=False,
    weight_map=False,
    tau_w=50e-3,              # EMA time-const for adaptive weights
    w_smooth_ksize=(3,3),      # spatial smoothing kernel for weight map
    return_spikes_only=False,
    output_path=None,
):
    """
    Global wide-field drive a_t with LOCAL adaptive weights from FILTERED excitation.

      pooled = local blur of bipolar frame (shared RF)
      amc_syn = EMA(pooled)
      bp_syn  = EMA(pooled)          # same RF as amacrine
      a_t     = reduce(amc_syn)      # scalar (mean/median)
      w_map   ~ EMA(bp_syn * a_t) / EMA(a_t^2)
      oms_syn = bp_syn - w_map * a_t
      mem     = EMA(oms_syn), reset on spike

    With output_path, record binary spikes directly to a .npy memory map and
    release file-backed pages periodically. Requires record=True and
    return_spikes_only=True; the recurrence and its one-step spike timing are unchanged.
    """
    T, H, W = bp_out_rec.shape
    bp_out_rec = np.asarray(bp_out_rec, dtype=np.float32)
    if output_path is not None and not (record and return_spikes_only):
        raise ValueError('Disk-backed output requires record=True and return_spikes_only=True')

    # EMA coefficients
    beta_syn_amc = float(np.exp(-dt / tau_syn));  alpha_syn_amc = 1.0 - beta_syn_amc
    beta_syn_oms = float(np.exp(-dt / tau_syn));  alpha_syn_oms = 1.0 - beta_syn_oms
    beta_mem = float(np.exp(-dt / tau_mem));  alpha_mem = 1.0 - beta_mem

    beta_w   = float(np.exp(-dt / tau_w));    alpha_w   = 1.0 - beta_w
    eps      = 1e-6
    a_min    = 1e-5
    # Adaptive weight stats
    w_top = np.zeros((H, W), np.float32)     # per-pixel EMA of bp_syn * a_t
    w_bot = 0.0                               # scalar EMA of a_t^2

    # States
    oms_mem     = np.zeros((H, W), np.float32)
    w_map = np.zeros((H, W), np.float32)

    # Records
    if record:
        rec_bp_syn, rec_oms_syn, rec_mem, rec_spk = [], [], [], []
        rec_a = []
        rec_amc_inh = []
        if output_path is not None:
            rec_spk = np.lib.format.open_memmap(output_path, mode='w+', dtype=np.uint8, shape=(T, H, W))

    k_h, k_w = rf_ksize
    use_blur = (k_h > 1 or k_w > 1)

    for t in range(T):
        oms_thr = oms_mem - threshold
        oms_out = spike_fn(oms_thr)

        bp_t = bp_out_rec[t]                                 # (H,W)

        # ---- Wide-field scalar from amacrine ----
        amc_bp_new_syn = cv2.blur(bp_t, (k_w, k_h)) if use_blur else bp_t
        a_t = float(np.median(amc_bp_new_syn) if reduce_op == "median" else np.mean(amc_bp_new_syn))
        if no_supression:
            a_t = 0.0


        oms_bp_new_syn = cv2.blur(bp_t, (k_w, k_h)) if use_blur else bp_t

        # ################# just trying 24/10/2025 ###########################
        # # ---- Adaptive LOCAL weights from FILTERED excitation ----
        # if abs(a_t) >= a_min:
        #     w_top = beta_w * w_top + alpha_w * (oms_bp_new_syn * a_t)  # (H,W)
        #     w_bot = beta_w * w_bot + alpha_w * (a_t * a_t)     # scalar
        # # else: freeze stats when drive is too small

        # w_map = w_top / (w_bot + eps)                          # (H,W)
        # if w_smooth_ksize is not None:
        #     kh, kw = w_smooth_ksize
        #     w_map = cv2.blur(w_map, (kw, kh))

        # ######################################################################


        # ---- OMS current & membrane ----
        if weight_map:
            amc_inh = w_map * a_t                     #### adaptive inhibition
        else:
            amc_inh = np.full((H, W), a_t, dtype=np.float32)  # (H,W) all with the same value

        oms_syn = oms_bp_new_syn -  amc_inh             #### wide field inhibition

        # integrate membrane
        mem_raw   = beta_mem * oms_mem + alpha_mem * oms_syn
        oms_new_mem = mem_raw - oms_out * threshold

        oms_mem = oms_new_mem


        if record:
            if not return_spikes_only:
                rec_bp_syn.append(oms_bp_new_syn)
                rec_a.append(a_t)
                rec_amc_inh.append(amc_inh)
                rec_oms_syn.append(oms_syn)
                rec_mem.append(oms_new_mem)
            if output_path is None:
                rec_spk.append(oms_out)
            else:
                rec_spk[t] = oms_out
                if t % 32 == 31:
                    release_pages(rec_spk)
                    release_pages(bp_out_rec)


    if not record:
        return None


    if return_spikes_only:
        if output_path is not None:
            release_pages(rec_spk)
            return rec_spk
        return np.stack(rec_spk, axis=0)

    return (
        np.stack(rec_bp_syn, axis=0),                 # 0: bp_syn   (T,H,W)
        np.array(rec_a, dtype=np.float32),            # 1: a_t      (T,)
        np.stack(rec_amc_inh, axis=0),                  # 2: w_map    (T,H,W)
        np.stack(rec_oms_syn, axis=0),                # 3: oms_syn  (T,H,W)
        np.stack(rec_mem, axis=0),                    # 4: mem      (T,H,W)
        np.stack(rec_spk, axis=0)                     # 5: spikes   (T,H,W)
    )
