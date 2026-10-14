"""Physics-guided descriptor dictionary extracted from the source experiment."""

import numpy as np
from scipy.interpolate import interp1d
from scipy.spatial import ConvexHull

def continuum_removal_batch(wls, X):
    N, M = X.shape
    X_cr = np.zeros_like(X)
    for i in range(N):
        x = X[i]
        if M < 3:
            X_cr[i] = np.ones(M)
            continue
        
        # Prepare 2D points: (wavelength, reflectance)
        pts = np.zeros((M + 2, 2))
        pts[:M, 0] = wls
        pts[:M, 1] = x
        
        # Bottom anchors to close the hull at the bottom
        min_val = np.min(x) - 1.0
        pts[M, 0] = wls[0]
        pts[M, 1] = min_val
        pts[M+1, 0] = wls[-1]
        pts[M+1, 1] = min_val
        
        try:
            hull = ConvexHull(pts)
            # Find vertices that are among the original M points
            valid_idx = sorted([v for v in hull.vertices if v < M])
            
            if len(valid_idx) < 2:
                valid_idx = [0, M - 1]
                
            hull_wls = wls[valid_idx]
            hull_vals = x[valid_idx]
            
            # Ensure boundary coverage
            if 0 not in valid_idx:
                hull_wls = np.insert(hull_wls, 0, wls[0])
                hull_vals = np.insert(hull_vals, 0, x[0])
            if (M - 1) not in valid_idx:
                hull_wls = np.append(hull_wls, wls[-1])
                hull_vals = np.append(hull_vals, x[-1])
                
            f = interp1d(hull_wls, hull_vals, kind='linear', bounds_error=False, fill_value="extrapolate")
            continuum = f(wls)
            
            continuum = np.where(continuum <= 0, 1e-8, continuum)
            X_cr[i] = x / continuum
        except Exception:
            X_cr[i] = np.ones(M)
            
    return X_cr

def compute_features(pixels: np.ndarray, wls: np.ndarray) -> np.ndarray:
    eps = 1e-8

    def band(lo, hi):
        return np.where((wls >= lo) & (wls <= hi))[0]

    def m_(idx):
        return pixels[:, idx].mean(1) if len(idx) > 0 else np.zeros(len(pixels))

    def at(wl):
        return pixels[:, np.argmin(np.abs(wls - wl))]

    def cr_depth(lo, hi):
        idx = band(lo, hi)
        if len(idx) < 2:
            return np.zeros(len(pixels))
        return 1.0 - continuum_removal_batch(wls[idx], pixels[:, idx]).min(axis=1)

    def cr_area(lo, hi):
        idx = band(lo, hi)
        if len(idx) < 2:
            return np.zeros(len(pixels))
        return (1.0 - continuum_removal_batch(wls[idx], pixels[:, idx])).sum(axis=1)

    B  = m_(band(0.45, 0.52)); G  = m_(band(0.52, 0.60)); R  = m_(band(0.63, 0.69))
    RE = m_(band(0.70, 0.74)); N  = m_(band(0.76, 0.90))
    S1 = m_(band(1.00, 1.35)); S2 = m_(band(1.45, 1.78)); S3 = m_(band(1.91, 2.51))

    r550 = at(0.55); r670 = at(0.67); r700 = at(0.70); r720 = at(0.72); r750 = at(0.75)
    r970 = at(0.97); r900 = at(0.90); r531 = at(0.531); r570 = at(0.570); r860 = at(0.86)
    r1510 = at(1.51); r1650 = at(1.65); r1680 = at(1.68); r1750 = at(1.75)
    r1690 = at(1.69); r1740 = at(1.74); r1500 = at(1.50); r1710 = at(1.71)
    r2000 = at(2.00); r2100 = at(2.10); r2200 = at(2.20)
    r1600 = at(1.60); r2050 = at(2.05); r2090 = at(2.09); r2150 = at(2.15)
    r2250 = at(2.25); r705 = at(0.705); r1480 = at(1.48); r1560 = at(1.56)
    r1240 = at(1.24); r830 = at(0.83); r1730 = at(1.73)

    d_chl  = cr_depth(0.55, 0.72); d_w970 = cr_depth(0.93, 1.05)
    d_w12  = cr_depth(1.10, 1.30); d_cell = cr_depth(2.05, 2.25)
    d_lig  = cr_depth(2.20, 2.35); d_w145 = cr_depth(1.45, 1.78)
    d_w19  = cr_depth(1.91, 2.05); d_starch  = cr_depth(2.06, 2.12)
    d_cell230 = cr_depth(2.27, 2.35); d_prot151 = cr_depth(1.48, 1.56)
    d_prot173 = cr_depth(1.69, 1.77)
    area_2050_2150 = cr_area(2.05, 2.15); area_2150_2250 = cr_area(2.15, 2.25)
    d_green_peak   = cr_depth(0.50, 0.58); area_nir_canopy = cr_area(0.76, 0.92)
    d_silica = cr_depth(2.12, 2.22)

    rep_idx = band(0.68, 0.78)
    if len(rep_idx) > 1:
        sub_re   = pixels[:, rep_idx]
        d_re     = np.diff(sub_re, axis=1) / np.diff(wls[rep_idx])
        RE_dmax  = d_re.max(axis=1)
        REP      = wls[rep_idx[:-1]][d_re.argmax(axis=1)]
    else:
        RE_dmax = np.zeros(len(pixels))
        REP     = np.zeros(len(pixels))

    all_feats = {
        "NDVI":   (N - R) / (N + R + eps),
        "GNDVI":  (N - G) / (N + G + eps),
        "OSAVI":  (N - R) / (N + R + 0.16),
        "SAVI":   (N - R) * 1.5 / (N + R + 0.5),
        "mSR":    (N / R - 1) / (np.sqrt(N / (R + eps)) + 1 + eps),
        "Cl_green": (N / (G + eps)) - 1,
        "CIRE":   (N / (RE + eps)) - 1,
        "MCARI":  ((r700 - r670) - 0.2 * (r700 - r550)) * (r700 / (r670 + eps)),
        "RE_idx": (RE - R) / (RE + R + eps),
        "RE_slope": (r750 - r670) / (0.75 - 0.67),
        "RE_dmax": RE_dmax,
        "REP":    REP,
        "RE_ratio": at(0.73) / (at(0.71) + eps),
        "NDWI2":  (N - S2) / (N + S2 + eps),
        "NBR":    (N - S3) / (N + S3 + eps),
        "WBI":    r900 / (r970 + eps),
        "d_w970": d_w970,
        "d_w1200": d_w12,
        "area_w12": cr_area(1.10, 1.30),
        "CAI_orig": 0.5 * (r2000 + r2200) - r2100,
        "CAI2":   (S1 - S2) / (S1 + S2 + eps),
        "NDLI":   (np.log(1 / (r1650 + eps)) - np.log(1 / (r1750 + eps))) / (np.log(1 / (r1650 + eps)) + np.log(1 / (r1750 + eps)) + eps),
        "NDNI":   (np.log(1 / (r1510 + eps)) - np.log(1 / (r1680 + eps))) / (np.log(1 / (r1510 + eps)) + np.log(1 / (r1680 + eps)) + eps),
        "NDLI_n": (np.log(1 / (r1690 + eps)) - np.log(1 / (r1740 + eps))) / (np.log(1 / (r1690 + eps)) + np.log(1 / (r1740 + eps)) + eps),
        "NDNI_n": (np.log(1 / (r1500 + eps)) - np.log(1 / (r1710 + eps))) / (np.log(1 / (r1500 + eps)) + np.log(1 / (r1710 + eps)) + eps),
        "d_cell": d_cell,
        "d_lig":  d_lig,
        "d_cell_21":   cr_depth(2.08, 2.13),
        "d_lignin_27": cr_depth(2.25, 2.31),
        "area_2100":   cr_area(2.00, 2.30),
        "SR_2100": r2100 / (r2200 + eps),
        "NIR_S3":  N / (S3 + eps),
        "S1_S2":   S1 / (S2 + eps),
        "R_S2":    R / (S2 + eps),
        "CL_ratio": d_cell / (d_lig + eps),
        "LC_pos":  cr_depth(2.25, 2.31) / (cr_depth(2.08, 2.13) + eps),
        "w145_w12_area": cr_area(1.45, 1.78) / (cr_area(1.10, 1.30) + eps),
        "BSI":     ((S2 + R) - (N + B)) / ((S2 + R) + (N + B) + eps),
        "NDSI":    (G - S2) / (G + S2 + eps),
        "d_700":   cr_depth(0.66, 0.78),
        "VIS":     (B + G + R) / 3,
        "NIR":     N,
        "SWIR1":   S1,
        "SWIR2":   S2,
        "SWIR3":   S3,
        "SW220":   m_(band(2.20, 2.30)),
        "PRI":     (r531 - r570) / (r531 + r570 + eps),
        "d_chl":   d_chl,
        "area_chl": cr_area(0.55, 0.72),
        "asym_21": (r2100 - r2000) - (r2200 - r2100),
        "flatness_vis": pixels[:, band(0.50, 0.66)].std(axis=1) if len(band(0.50, 0.66)) > 0 else np.zeros(len(pixels)),
        "flatness_nir": pixels[:, band(0.78, 0.93)].std(axis=1) if len(band(0.78, 0.93)) > 0 else np.zeros(len(pixels)),
        "d_w145":  d_w145,
        "d_w19":   d_w19,
        "NDII":    (N - S1) / (N + S1 + eps),
        "EVI":     2.5 * (N - R) / (N + 6 * R - 7.5 * B + 1 + eps),
        "CAI3":    2 * r2100 - r2000 - r2200,
        "slope_sw": m_(band(2.10, 2.20)) - m_(band(2.00, 2.10)),
        "area_w19": cr_area(1.91, 2.05),
        "NDRE":    (N - r705) / (N + r705 + eps),
        "CI_rededge": (N / (r705 + eps)) - 1,
        "d_w850":  cr_depth(0.85, 0.92),
        "area_lig": cr_area(2.20, 2.35),
        "slope_vis": (R - B) / (0.67 - 0.48 + eps),
        "slope_re":  (r750 - r670) / 0.08,
        "area_chl2": cr_area(0.60, 0.72),
        "area_w145": cr_area(1.45, 1.78),
        "SWIR_ratio": S1 / (S3 + eps),
        "soil_adj": (N - R - 0.05 * (1 + N + R - 2 * G)) / ((N + R + 0.05 * (1 + N + R - 2 * G)) + eps),
        "d_starch":    d_starch,
        "d_cell230":   d_cell230,
        "d_prot151":   d_prot151,
        "d_prot173":   d_prot173,
        "area_2050_2150": area_2050_2150,
        "area_2150_2250": area_2150_2250,
        "starch_cell_ratio": area_2050_2150 / (area_2150_2250 + eps),
        "CAI_2090":    r2050 - 2 * r2090 + r2150,
        "SWIR_slope_fine": (at(2.25) - at(2.00)) / (0.25 + eps),
        "cell_prot_ratio": d_cell / (d_prot151 + eps),
        "prot_n_idx":  (r1480 - r1560) / (r1480 + r1560 + eps),
        "r2090_2200":  r2090 / (r2200 + eps),
        "r2050_2250":  r2050 / (r2250 + eps),
        "d_OH141":     cr_depth(1.38, 1.45),
        "MSI":         r1600 / (N + eps),
        "NDII_1600":   (N - r1600) / (N + r1600 + eps),
        "R_G_ratio":   R / (G + eps),
        "d_green_peak": d_green_peak,
        "area_green_peak": cr_area(0.50, 0.58),
        "CRI1":        (1.0 / (G + eps)) - (1.0 / (at(0.70) + eps)),
        "CRI2":        (1.0 / (G + eps)) - (1.0 / (at(0.75) + eps)),
        "SWIR1_NIR":   S1 / (N + eps),
        "dryness_idx": (R + S1) / (N + G + eps),
        "NDBI":        (S1 - N) / (S1 + N + eps),
        "MNDWI":       (G - S1) / (G + S1 + eps),
        "UI":          (S2 - N) / (S2 + N + eps),
        "VIS_NIR":     ((B + G + R) / 3) / (N + eps),
        "d_silica":    d_silica,
        "SWIR2_NIR":   S2 / (N + eps),
        "BU_idx":      ((S1 - N) / (S1 + N + eps)) - ((N - R) / (N + R + eps)),
        "area_nir_canopy": area_nir_canopy,
        "EVI2":        2.5 * (N - R) / (N + 2.4 * R + 1 + eps),
        "SWIR_diff":   S1 - S2,
        "NIR_S1_ratio": N / (S1 + eps),
        "canopy_water": (r860 - at(0.87)) / (r860 + at(0.87) + eps),
        "NH_ratio":    r1510 / (r1730 + eps),
        "starch_sig":  2 * r2090 - r2000 - r2200,
        "cellulose_2330": cr_depth(2.28, 2.36),
        "log_NH":      np.log((r1500 + eps) / (r1710 + eps)),
        "tillage_proxy": N / (N + (0.5 * (r2000 + r2200) - r2100) + 0.01 + eps),
        "residue_exp": (R + S1 - N - B) / (R + S1 + N + B + eps),
        "corn_narrow": r2090 / (r2200 + eps),
        "CAI_x_NDVI":  (0.5 * (r2000 + r2200) - r2100) * ((N - R) / (N + R + eps)),
        "S1_NIR_slope": (S1 - N) / (N + eps),
        "spec_var_swir": pixels[:, band(2.00, 2.35)].std(axis=1) if len(band(2.00, 2.35)) > 0 else np.zeros(len(pixels)),
        "chl_x_water": d_chl * d_w970,
        "dry_cover":   0.5 * (r2000 + r2200) - r2100 - ((N - R) / (N + R + eps)),
        "senescence":  R / (G + eps),
        "FMC":         (r860 - r1240) / (r860 + r1240 + eps),
        "WBI_CAI":     r900 / (r970 + eps) - (0.5 * (r2000 + r2200) - r2100),
        "alfalfa_score": ((N - R) / (N + R + eps)) * ((N - S2) / (N + S2 + eps)) * N,
        "hay_score":   S3 / (N + G + eps),
        "NIR_struct":  N / (r830 + eps),
        "dew_proxy":   d_w970 / (d_chl + eps),
        "albedo_vis":  (B + G + R) / 3,
        "IBI":         2 * S1 / (S1 + N + eps) - (N / (N + R + eps) + G / (G + S1 + eps)),
        "swir_vis":    (S1 + S2) / ((B + G + R) + eps),
        "oats_maturity": R / (G + B + eps),
        "concrete":    d_silica,
        "urban_veg":   (1 - (N - R) / (N + R + eps)) * S2,
        "GT_score":    ((N - R) / (N + R + eps)) * d_w970 * N,
        "spec_var_vis": pixels[:, band(0.45, 0.75)].std(axis=1) if len(band(0.45, 0.75)) > 0 else np.zeros(len(pixels)),
        "nir_vis_diff": N - ((B + G + R) / 3),
        "brick_clay":  cr_depth(2.20, 2.32) / (cr_depth(2.08, 2.13) + eps),
        "crown_closure": N ** 2 / (R + eps),
        "LAI_proxy":   np.log(1 + N / (R + eps)),
        "woods_score": ((N - S2) / (N + S2 + eps)) * N * (1 - ((S2 + R - N - B) / (S2 + R + N + B + eps))),
        "soil_veg_mix": ((S2 + R - N - B) / (S2 + R + N + B + eps)) * (1 - (N - R) / (N + R + eps)),
        "nir_s1_diff": N - S1,
        "NDMI":        (N - r1600) / (N + r1600 + eps),
        "forest_shadow": B / (N + eps),
        "grass_ratio": G / (R + eps),

        # ── 신규 지수 2차 배치 (30개) ────────────────────────────────────────
        "d1_720":      (at(0.730) - at(0.710)) / 0.020,
        "d2_re":       r720 - 2 * r700 + at(0.680),
        "SIPI":        (N - B) / (N - R + eps),
        "PSRI":        (R - B) / (RE + eps),
        "MTCI":        (at(0.754) - r705) / (r705 - r670 + eps),
        "ARI2":        N * (1.0 / (G + eps) - 1.0 / (r700 + eps)),
        "CVI":         N * R / (G ** 2 + eps),
        "ARVI":        (N - (2 * R - B)) / (N + (2 * R - B) + eps),
        "kNDVI":       np.tanh(((N - R) / (N + R + eps)) ** 2),
        "NDTI":        (r1600 - r2200) / (r1600 + r2200 + eps),
        "STI":         r1510 / (r1750 + eps),
        "NDSRI":       (r2090 - r2200) / (r2090 + r2200 + eps),
        "NBI":         R * S2 / (N + eps),
        "BCI":         ((S1 + R) / 2 - N) / ((S1 + R) / 2 + N + eps),
        "BI":          np.sqrt((R ** 2 + G ** 2) / 2 + eps),
        "CI_s":        (R - B) / (R + B + eps),
        "RI":          (R - G) / (R + G + eps),
        "nir_plateau": m_(band(0.76, 1.30)),
        "r740_r720":   at(0.740) / (r720 + eps),
        "r550_r670":   r550 / (r670 + eps),
        "SWIR2_SWIR3": S2 / (S3 + eps),
        "SWIR3_VIS":   S3 / ((B + G + R) / 3 + eps),
        "NDSI2":       (S1 - S2) / (S1 + S2 + eps),
        "NDWI_gao":    (G - N) / (G + N + eps),
        "r1100_r900":  m_(band(1.05, 1.15)) / (r900 + eps),
        "B_R":         B / (R + eps),
        "NDVI_CAI":    ((N - R) / (N + R + eps)) * (0.5 * (r2000 + r2200) - r2100),
        "oats_score2": ((N - S1) / (N + S1 + eps)) * d_w970,
        "buildings_idx": (R * S2 - N ** 2) / (R * S2 + N ** 2 + 1e-6),
        "alfalfa2":    N / (R + S1 + eps) * d_chl,

        # ── 신규 지수 3차 배치 (25개) ────────────────────────────────────────
        "ExG":          2 * G - R - B,
        "VARI":         (G - R) / (G + R - B + eps),
        "MGRVI":        (G ** 2 - R ** 2) / (G ** 2 + R ** 2 + eps),
        "REIP_est":     700 + 40 * (((r670 + at(0.780)) / 2) - r700) / (at(0.740) - r700 + eps),
        "CAI_norm":     (r2000 + r2200 - 2 * r2100) / (r2000 + r2200 + eps),
        "r2100_r1650":  r2100 / (r1650 + eps),
        "r1100_r1650":  m_(band(1.05, 1.15)) / (r1650 + eps),
        "r2200_r1500":  r2200 / (r1500 + eps),
        "SWIR_curv2":   r2200 - 0.5 * (r2100 + at(2.30)),
        "d_cellulose2": cr_depth(2.12, 2.18),
        "RCI":          d_cell / (d_w12 + eps),
        "r850_r700":    at(0.850) / (r700 + eps),
        "flatness_swir1": pixels[:, band(1.00, 1.35)].std(axis=1) if len(band(1.00, 1.35)) > 0 else np.zeros(len(pixels)),
        "notill_sig":   (0.5 * (r2000 + r2200) - r2100) * (1.0 - (N - R) / (N + R + eps)),
        "clean_sig":    (N - R) / (N + R + eps) / (np.abs(0.5 * (r2000 + r2200) - r2100) + 0.05),
        "till_ratio":   (0.5 * (r2000 + r2200) - r2100) / ((N - R) / (N + R + eps) + 0.10),
        "SOC_idx":      (R - B) / (N + eps),
        "CCCI":         (RE - R) / (RE + R + eps) / ((N - R) / (N + R + eps) + eps),
        "nir_red_ratio": N / (R + eps),
        "d_lca":        cr_depth(2.30, 2.40),
        "corn_R_SWIR":  R / (S3 + eps),
        "r530_r560":    r531 / (at(0.560) + eps),
        "SWIR1_R":      S1 / (R + eps),
        "NDVI_x_S2":    ((N - R) / (N + R + eps)) * S2,
        "water_NDVI":   d_w970 / ((N - R) / (N + R + eps) + eps),
    }

    X_out = np.stack([all_feats[k] for k in all_feats], axis=1)
    return np.where(np.isfinite(X_out), X_out, 0.0), list(all_feats.keys())

def compute_descriptors(
    spectra: np.ndarray,
    wavelengths_nm: np.ndarray,
    mask: np.ndarray | None = None,
) -> tuple[np.ndarray, list[str]]:
    """Compute the source experiment's 196 named physics descriptors.

    Input spectra may be reflectance fractions or the Indian Pines integer scale.
    A partial wavelength mask conservatively gates the complete descriptor vector;
    this prevents unmeasured wavelengths from being interpreted as reflectance.
    """
    values = np.asarray(spectra, dtype=float)
    wavelengths = np.asarray(wavelengths_nm, dtype=float).reshape(-1)
    if values.ndim != 2 or values.shape[1] != wavelengths.size:
        raise ValueError("spectra columns must align with wavelengths")
    if wavelengths.size < 2 or not np.all(np.diff(wavelengths) > 0):
        raise ValueError("wavelengths must be strictly increasing")
    availability = None
    if mask is not None:
        availability = np.asarray(mask, dtype=bool)
        if availability.shape not in ((wavelengths.size,), values.shape):
            raise ValueError("mask must match the wavelength axis or spectra shape")
    wavelengths_um = wavelengths / 1000.0 if wavelengths.max() > 10 else wavelengths
    reflectance = values / 10000.0 if values.size and np.nanmax(np.abs(values)) > 2 else values
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        descriptors, names = compute_features(reflectance, wavelengths_um)
    if descriptors.shape[1] != 196 or len(names) != 196:
        raise RuntimeError(f"descriptor dictionary must contain 196 features, got {len(names)}")
    if availability is not None:
        if availability.shape == (wavelengths.size,):
            if not availability.all():
                descriptors[:] = 0.0
        else:
            descriptors[~availability.all(axis=1)] = 0.0
    return np.where(np.isfinite(descriptors), descriptors, 0.0), names

