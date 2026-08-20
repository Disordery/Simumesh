"""Electromagnetic path-loss, interference, and PHY-rate models.

All distance args are meters and all power args are dBm unless noted.
Functions accept scalars or numpy arrays interchangeably.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

NOISE_FLOOR_DBM = -95.0
MAX_RSSI_DBM = -30.0
LIGHT_SPEED_M_S = 299_792_458.0
MAX_GRAZING_ANGLE_DEG = 80.0  # cap on wall-incidence angle to avoid 1/cos blowup

# Log-distance path-loss exponents by environment classification.
GAMMA_BY_ENVIRONMENT = {
    "free_space": 2.0,
    "open_office": 2.5,
    "office": 3.0,
    "residential": 3.2,
    "dense_obstruction": 4.2,
}


# --------------------------------------------------------------------------
# Free-space & log-distance path loss
# --------------------------------------------------------------------------

def fspl_db(distance_m, freq_ghz: float):
    """FSPL(dB) = 20log10(d_km) + 20log10(f_MHz) + 32.44."""
    d_km = np.maximum(np.asarray(distance_m, dtype=float), 0.01) / 1000.0
    f_mhz = freq_ghz * 1000.0
    return 20.0 * np.log10(d_km) + 20.0 * np.log10(f_mhz) + 32.44


def log_distance_path_loss(distance_m, freq_ghz: float, gamma: float = 3.0,
                            d0: float = 1.0, shadow_sigma_db: float = 0.0,
                            rng: Optional[np.random.Generator] = None):
    """PL(d) = PL(d0) + 10*gamma*log10(d/d0) [+ lognormal shadowing]."""
    d = np.maximum(np.asarray(distance_m, dtype=float), 0.1)
    pl_d0 = fspl_db(np.asarray(d0, dtype=float), freq_ghz)
    pl = pl_d0 + 10.0 * gamma * np.log10(d / d0)
    if shadow_sigma_db > 0:
        rng = rng or np.random.default_rng()
        pl = pl + rng.normal(0.0, shadow_sigma_db, size=np.shape(pl))
    return pl


# --------------------------------------------------------------------------
# Wall / multi-wall attenuation (COST231 Multi-Wall Model)
# --------------------------------------------------------------------------

def angle_adjusted_wall_loss(base_loss_db, incidence_angle_rad):
    """L_effective = L_material / cos(theta), theta capped at MAX_GRAZING_ANGLE_DEG."""
    angle_deg = np.minimum(np.degrees(np.abs(incidence_angle_rad)), MAX_GRAZING_ANGLE_DEG)
    cos_theta = np.cos(np.radians(angle_deg))
    cos_theta = np.maximum(cos_theta, np.cos(np.radians(MAX_GRAZING_ANGLE_DEG)))
    return base_loss_db / cos_theta


def total_path_loss(distance_m, freq_ghz: float, wall_loss_db=0.0,
                     gamma: float = 3.0, model: str = "log_distance"):
    """COST231 Multi-Wall Model: base propagation loss + summed wall attenuation."""
    if model == "fspl":
        base = fspl_db(distance_m, freq_ghz)
    else:
        base = log_distance_path_loss(distance_m, freq_ghz, gamma=gamma)
    return base + wall_loss_db


# --------------------------------------------------------------------------
# Link budget / RSSI
# --------------------------------------------------------------------------

def rssi_dbm(tx_power_dbm, path_loss_db, tx_gain_dbi=0.0, rx_gain_dbi: float = 0.0):
    return tx_power_dbm + tx_gain_dbi + rx_gain_dbi - path_loss_db


def dbm_to_mw(dbm):
    return 10.0 ** (np.asarray(dbm, dtype=float) / 10.0)


def mw_to_dbm(mw):
    return 10.0 * np.log10(np.maximum(np.asarray(mw, dtype=float), 1e-12))


def path_phase(distance_m, freq_ghz: float):
    """Carrier phase (rad, wrapped to [0, 2pi)) accrued over a travel distance."""
    wavelength_m = LIGHT_SPEED_M_S / (freq_ghz * 1e9)
    return np.mod(2.0 * np.pi * np.asarray(distance_m, dtype=float) / wavelength_m, 2 * np.pi)


# --------------------------------------------------------------------------
# SINR / interference
# --------------------------------------------------------------------------

def sinr_db(signal_dbm, interference_mw_total, noise_floor_dbm: float = NOISE_FLOOR_DBM):
    """SINR(dB) = 10log10( signal_mw / (interference_mw + noise_mw) )."""
    signal_mw = dbm_to_mw(signal_dbm)
    noise_mw = dbm_to_mw(noise_floor_dbm)
    denom = np.asarray(interference_mw_total, dtype=float) + noise_mw
    return mw_to_dbm(signal_mw) - mw_to_dbm(denom)


# --------------------------------------------------------------------------
# Shannon capacity & MCS lookup
# --------------------------------------------------------------------------

def shannon_capacity_mbps(bandwidth_mhz, sinr_db_val):
    sinr_linear = 10.0 ** (np.asarray(sinr_db_val, dtype=float) / 10.0)
    return bandwidth_mhz * np.log2(1.0 + np.maximum(sinr_linear, 0.0))


# (mcs_index, modulation, code_rate, min_snr_db_for_pdr, rate_20MHz_1SS_mbps)
# Representative of 802.11ac/ax rate tables at ~10% PER thresholds.
MCS_TABLE = [
    (0, "BPSK", 0.5, 2.0, 8.6),
    (1, "QPSK", 0.5, 5.0, 17.2),
    (2, "QPSK", 0.75, 9.0, 25.8),
    (3, "16-QAM", 0.5, 11.0, 34.4),
    (4, "16-QAM", 0.75, 15.0, 51.6),
    (5, "64-QAM", 0.667, 18.0, 68.8),
    (6, "64-QAM", 0.75, 20.0, 77.4),
    (7, "64-QAM", 0.833, 25.0, 86.0),
    (8, "256-QAM", 0.75, 29.0, 103.2),
    (9, "256-QAM", 0.833, 31.0, 114.7),
    (10, "1024-QAM", 0.75, 34.0, 129.0),
    (11, "1024-QAM", 0.833, 36.0, 143.4),
]

BANDWIDTH_SCALE = {20: 1.0, 40: 2.1, 80: 4.4, 160: 8.8}


def mcs_lookup(snr_db_val: float, bandwidth_mhz: int = 80, spatial_streams: int = 2) -> dict:
    """Highest sustainable MCS for a given SNR, scaled to bandwidth/streams."""
    best = MCS_TABLE[0]
    for row in MCS_TABLE:
        if snr_db_val >= row[3]:
            best = row
        else:
            break
    scale = BANDWIDTH_SCALE.get(bandwidth_mhz, bandwidth_mhz / 20.0)
    rate = best[4] * scale * spatial_streams
    return {
        "mcs_index": best[0], "modulation": best[1], "code_rate": best[2],
        "phy_rate_mbps": round(float(rate), 1),
    }
