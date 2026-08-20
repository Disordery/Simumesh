"""Vectorized 2D ray-casting engine.

Two complementary propagation paths are modeled:

  * ``direct_path_field``  - fully vectorized AP -> every grid cell path loss,
    i.e. the transmitted/penetrating energy through intervening walls
    (COST231 Multi-Wall Model). This is the dominant coverage contributor.

  * ``cast_ap_rays`` / ``accumulate_multipath_grid``  - angular ray casting
    with specular reflections off wall surfaces, depositing incoherent (or
    optionally phase-coherent) power into the grid. This fills NLOS shadow
    pockets that the direct field alone cannot reach.

A uniform-bucket spatial index (``WallIndex``) accelerates wall lookups for
the sequential ray tracer, and the innermost intersection kernel is
Numba-JIT compiled when numba is available (pure-numpy fallback otherwise).
"""
from __future__ import annotations

import math
from typing import Dict, List, Tuple

import numpy as np

import physics
from models import AccessPoint, SignalGrid, Wall

try:
    from numba import njit
    NUMBA_AVAILABLE = True
except Exception:  # pragma: no cover - numba is optional
    NUMBA_AVAILABLE = False

    def njit(*args, **kwargs):
        def _wrap(fn):
            return fn
        if len(args) == 1 and callable(args[0]):
            return args[0]
        return _wrap


# ==========================================================================
# Vectorized direct-path field (AP -> every grid cell simultaneously)
# ==========================================================================

def _batched_wall_crossing_loss(ap_pos: np.ndarray, d1: np.ndarray,
                                 w1: np.ndarray, w2: np.ndarray,
                                 wnorm: np.ndarray, wloss: np.ndarray) -> np.ndarray:
    """For C ray origins->targets (encoded via d1) vs M walls, sum angle-adjusted
    wall loss for every wall genuinely crossed between the AP and each target.
    Returns shape (C,).
    """
    p1x, p1y = ap_pos
    d1x = d1[:, 0][:, None]
    d1y = d1[:, 1][:, None]                       # (C,1)

    w1x = w1[:, 0][None, :]; w1y = w1[:, 1][None, :]   # (1,M)
    w2x = w2[:, 0][None, :]; w2y = w2[:, 1][None, :]
    d2x = w2x - w1x; d2y = w2y - w1y

    denom = d1x * d2y - d1y * d2x                  # (C,M)
    with np.errstate(divide="ignore", invalid="ignore"):
        denom_safe = np.where(np.abs(denom) < 1e-12, np.nan, denom)
        fx = w1x - p1x; fy = w1y - p1y
        t = (fx * d2y - fy * d2x) / denom_safe      # param along AP->target
        u = (fx * d1y - fy * d1x) / denom_safe      # param along wall segment

    hits = (t > 1e-6) & (t < 1.0 - 1e-6) & (u >= -1e-9) & (u <= 1.0 + 1e-9)
    hits &= ~np.isnan(t)

    ray_norm = np.hypot(d1x, d1y)
    rux = d1x / np.maximum(ray_norm, 1e-12)
    ruy = d1y / np.maximum(ray_norm, 1e-12)
    nx = wnorm[:, 0][None, :]; ny = wnorm[:, 1][None, :]
    cos_i = np.abs(rux * nx + ruy * ny)
    cos_i = np.clip(cos_i, math.cos(math.radians(physics.MAX_GRAZING_ANGLE_DEG)), 1.0)
    incidence = np.arccos(cos_i)

    eff_loss = physics.angle_adjusted_wall_loss(wloss[None, :], incidence)  # (C,M)
    return np.where(hits, eff_loss, 0.0).sum(axis=1)


def direct_path_field(ap: AccessPoint, walls: List[Wall], grid: SignalGrid,
                       gamma: float = 3.0, model: str = "log_distance",
                       chunk_size: int = 4000) -> np.ndarray:
    """RSSI(dBm) from ``ap`` to every grid cell, chunked to bound memory to
    O(chunk_size * n_walls) regardless of grid resolution.
    """
    pts = grid.points
    n = pts.shape[0]
    ap_pos = ap.pos
    rssi_flat = np.empty(n)

    if walls:
        w1 = np.array([[w.x1, w.y1] for w in walls])
        w2 = np.array([[w.x2, w.y2] for w in walls])
        wnorm = np.array([w.normal for w in walls])
        wloss = np.array([w.material.loss_db(ap.band_ghz) for w in walls])

    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        chunk = pts[start:end]
        d1 = chunk - ap_pos
        dist = np.linalg.norm(d1, axis=1)

        if walls:
            wall_loss = _batched_wall_crossing_loss(ap_pos, d1, w1, w2, wnorm, wloss)
        else:
            wall_loss = np.zeros(end - start)

        az = np.degrees(np.arctan2(d1[:, 1], d1[:, 0]))
        tx_gain = ap.antenna_gain_vectorized(az)

        pl = physics.total_path_loss(dist, ap.band_ghz, wall_loss_db=wall_loss,
                                      gamma=gamma, model=model)
        rssi_flat[start:end] = physics.rssi_dbm(ap.tx_power_dbm, pl, tx_gain_dbi=tx_gain)

    return rssi_flat.reshape(grid.shape)


def point_to_point_path_loss(p1: np.ndarray, p2: np.ndarray, walls: List[Wall],
                              band_ghz: float, gamma: float = 2.5,
                              model: str = "log_distance") -> float:
    """Path loss between two arbitrary points (used for mesh backhaul links)."""
    p1 = np.asarray(p1, dtype=float); p2 = np.asarray(p2, dtype=float)
    dist = float(np.hypot(*(p2 - p1)))
    if not walls:
        wall_loss = 0.0
    else:
        w1 = np.array([[w.x1, w.y1] for w in walls])
        w2 = np.array([[w.x2, w.y2] for w in walls])
        wnorm = np.array([w.normal for w in walls])
        wloss = np.array([w.material.loss_db(band_ghz) for w in walls])
        d1 = (p2 - p1)[None, :]
        wall_loss = float(_batched_wall_crossing_loss(p1, d1, w1, w2, wnorm, wloss)[0])
    return physics.total_path_loss(dist, band_ghz, wall_loss_db=wall_loss, gamma=gamma, model=model)


# ==========================================================================
# Spatial index for the sequential multipath ray tracer
# ==========================================================================

class WallIndex:
    """Uniform-bucket spatial index over wall segments (quadtree-equivalent
    candidate lookup for near-uniform wall density) to accelerate per-ray
    intersection queries during multipath tracing.
    """

    def __init__(self, walls: List[Wall], width_m: float, height_m: float,
                 cell_size_m: float = 2.0):
        self.walls = walls
        self.cell = max(cell_size_m, 0.5)
        self.buckets: Dict[Tuple[int, int], List[int]] = {}
        for idx, w in enumerate(walls):
            for c in self._cells_for_segment(w.x1, w.y1, w.x2, w.y2):
                self.buckets.setdefault(c, []).append(idx)

    def _cell_of(self, x: float, y: float) -> Tuple[int, int]:
        return (int(math.floor(x / self.cell)), int(math.floor(y / self.cell)))

    def _cells_for_segment(self, x1, y1, x2, y2):
        c1 = self._cell_of(min(x1, x2), min(y1, y2))
        c2 = self._cell_of(max(x1, x2), max(y1, y2))
        return [(cx, cy) for cx in range(c1[0], c2[0] + 1) for cy in range(c1[1], c2[1] + 1)]

    def candidates_for_ray(self, x1, y1, x2, y2) -> List[int]:
        c1 = self._cell_of(min(x1, x2), min(y1, y2))
        c2 = self._cell_of(max(x1, x2), max(y1, y2))
        ids = set()
        for cx in range(c1[0] - 1, c2[0] + 2):
            for cy in range(c1[1] - 1, c2[1] + 2):
                ids.update(self.buckets.get((cx, cy), []))
        return list(ids)


@njit(cache=True)
def _nearest_intersection(rx1, ry1, rx2, ry2, wx1, wy1, wx2, wy2):
    """Nearest wall crossing along ray segment (rx1,ry1)->(rx2,ry2) among the
    supplied candidate walls. Returns (t, local_wall_index) or (-1.0, -1).
    """
    best_t = 2.0
    best_idx = -1
    dx = rx2 - rx1
    dy = ry2 - ry1
    n = wx1.shape[0]
    for i in range(n):
        ex = wx2[i] - wx1[i]
        ey = wy2[i] - wy1[i]
        denom = dx * ey - dy * ex
        if abs(denom) < 1e-12:
            continue
        fx = wx1[i] - rx1
        fy = wy1[i] - ry1
        t = (fx * ey - fy * ex) / denom
        u = (fx * dy - fy * dx) / denom
        if 1e-6 < t < best_t and -1e-9 <= u <= 1.0 + 1e-9:
            best_t = t
            best_idx = i
    if best_idx == -1:
        return -1.0, -1
    return best_t, best_idx


# ==========================================================================
# Multipath / reflection ray tracing
# ==========================================================================

def cast_ap_rays(ap: AccessPoint, walls: List[Wall], wall_index: WallIndex,
                  n_rays: int = 720, max_bounces: int = 2,
                  max_range_m: float = 40.0) -> List[list]:
    """Cast ``n_rays`` angular rays from the AP; each bounces specularly off
    walls (material reflection coefficient controls energy loss per bounce)
    up to ``max_bounces`` times. Returns a list of ray paths, each a list of
    (p1, p2, cumulative_loss_db_before_segment, bounce_index) tuples.
    """
    paths = []
    angles = np.linspace(0.0, 2 * np.pi, n_rays, endpoint=False)
    for a0 in angles:
        origin = ap.pos.copy()
        direction = np.array([math.cos(a0), math.sin(a0)])
        segments = []
        cumulative_loss = 0.0
        remaining_frac = 1.0

        for bounce in range(max_bounces + 1):
            far = origin + direction * max_range_m
            cand = wall_index.candidates_for_ray(origin[0], origin[1], far[0], far[1])
            if not cand:
                segments.append((origin, far, cumulative_loss, bounce))
                break

            wx1 = np.array([walls[i].x1 for i in cand]); wy1 = np.array([walls[i].y1 for i in cand])
            wx2 = np.array([walls[i].x2 for i in cand]); wy2 = np.array([walls[i].y2 for i in cand])
            t, local_idx = _nearest_intersection(origin[0], origin[1], far[0], far[1], wx1, wy1, wx2, wy2)

            if local_idx == -1:
                segments.append((origin, far, cumulative_loss, bounce))
                break

            wall = walls[cand[local_idx]]
            hit_point = origin + direction * (t * max_range_m)
            segments.append((origin, hit_point, cumulative_loss, bounce))

            reflect_coeff = max(wall.material.reflection_coeff, 1e-4)
            cumulative_loss += -10.0 * math.log10(reflect_coeff)
            remaining_frac *= reflect_coeff

            if remaining_frac < 0.01 or bounce == max_bounces:
                break

            wall_dir = wall.vector / max(wall.length, 1e-9)
            normal = np.array([-wall_dir[1], wall_dir[0]])
            direction = direction - 2.0 * np.dot(direction, normal) * normal
            direction = direction / max(np.linalg.norm(direction), 1e-9)
            origin = hit_point + direction * 1e-4

        paths.append(segments)
    return paths


def accumulate_multipath_grid(grid: SignalGrid, ap: AccessPoint, ray_paths: List[list],
                               coherent: bool = False) -> np.ndarray:
    """March along every ray segment, depositing received power (mW) into the
    nearest grid cell at each sample point. Power sums incoherently by
    default (standard for building-scale multipath); ``coherent=True`` sums
    complex amplitudes using the carrier phase instead.
    """
    ny, nx = grid.shape
    cell = grid.resolution_m
    samples_per_m = 1.0 / cell

    power_accum_mw = np.zeros((ny, nx))
    phase_accum = np.zeros((ny, nx), dtype=complex) if coherent else None

    for segments in ray_paths:
        path_len = 0.0
        for (p1, p2, loss_before, _bounce) in segments:
            seg_len = float(np.hypot(*(p2 - p1)))
            if seg_len < 1e-9:
                continue
            n_samples = max(1, int(seg_len * samples_per_m))
            fracs = np.linspace(0.0, 1.0, n_samples + 1)
            pts = p1[None, :] + (p2 - p1)[None, :] * fracs[:, None]
            dist_along = path_len + seg_len * fracs
            total_dist = np.maximum(dist_along, 0.05)

            pl = physics.fspl_db(total_dist, ap.band_ghz) + loss_before
            rssi = physics.rssi_dbm(ap.tx_power_dbm, pl)
            mw = physics.dbm_to_mw(rssi)

            ix = np.round(pts[:, 0] / cell).astype(int)
            iy = np.round(pts[:, 1] / cell).astype(int)
            valid = (ix >= 0) & (ix < nx) & (iy >= 0) & (iy < ny)

            if coherent:
                ph = physics.path_phase(total_dist, ap.band_ghz)
                amp = np.sqrt(mw) * np.exp(1j * ph)
                np.add.at(phase_accum, (iy[valid], ix[valid]), amp[valid])
            else:
                np.add.at(power_accum_mw, (iy[valid], ix[valid]), mw[valid])

            path_len += seg_len

    if coherent:
        power_accum_mw = np.abs(phase_accum) ** 2
    return power_accum_mw


def combine_fields(direct_rssi_dbm: np.ndarray, multipath_mw: np.ndarray) -> np.ndarray:
    """Total received power = direct/transmitted field + incoherent reflected
    field, summed in the linear (mW) domain."""
    total_mw = physics.dbm_to_mw(direct_rssi_dbm) + multipath_mw
    return physics.mw_to_dbm(total_mw)
