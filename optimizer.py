"""Automated mesh-node placement optimizer.

Finds AP positions (and the minimal AP count) needed to guarantee a target
RSSI threshold across a target fraction of walkable floor space, using
simulated annealing or a genetic algorithm. Uses only the fast vectorized
direct-path field during search (multipath is skipped for speed - it mainly
fills small NLOS pockets and is not needed to drive placement search).
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

from models import AccessPoint, FloorPlan, SignalGrid
from raytracer import direct_path_field


@dataclass
class OptimizationResult:
    positions: List[Tuple[float, float]]
    coverage_fraction: float
    n_aps: int
    history: List[float] = field(default_factory=list)


def default_walkable_mask(floor_plan: FloorPlan, grid: SignalGrid) -> np.ndarray:
    """All grid cells are walkable by default; pass an explicit mask to
    exclude e.g. mechanical rooms or building exteriors."""
    return np.ones(grid.shape, dtype=bool)


def _coverage_fraction(positions: List[Tuple[float, float]], floor_plan: FloorPlan,
                        grid: SignalGrid, tx_power_dbm: float, band_ghz: float,
                        target_rssi_dbm: float, gamma: float,
                        walkable_mask: np.ndarray) -> float:
    best = np.full(grid.shape, -999.0)
    for (x, y) in positions:
        ap = AccessPoint(id="_opt", x=x, y=y, tx_power_dbm=tx_power_dbm, band_ghz=band_ghz)
        field_map = direct_path_field(ap, floor_plan.walls, grid, gamma=gamma)
        best = np.maximum(best, field_map)
    covered = (best >= target_rssi_dbm) & walkable_mask
    denom = max(int(walkable_mask.sum()), 1)
    return float(covered.sum()) / denom


def simulated_annealing_placement(floor_plan: FloorPlan, grid: SignalGrid, n_aps: int,
                                   tx_power_dbm: float = 20.0, band_ghz: float = 5.0,
                                   target_rssi_dbm: float = -67.0, gamma: float = 3.0,
                                   iterations: int = 300, initial_temp: float = 5.0,
                                   cooling: float = 0.97, seed: Optional[int] = None,
                                   walkable_mask: Optional[np.ndarray] = None) -> OptimizationResult:
    rng = random.Random(seed)
    mask = walkable_mask if walkable_mask is not None else default_walkable_mask(floor_plan, grid)

    positions = [(rng.uniform(0, floor_plan.width_m), rng.uniform(0, floor_plan.height_m))
                 for _ in range(n_aps)]
    best_cov = _coverage_fraction(positions, floor_plan, grid, tx_power_dbm, band_ghz,
                                   target_rssi_dbm, gamma, mask)
    best_positions = list(positions)
    temp = initial_temp
    history = [best_cov]

    for _ in range(iterations):
        idx = rng.randrange(n_aps)
        old = positions[idx]
        step = max(0.5, temp)
        nx = min(max(old[0] + rng.uniform(-step, step), 0.0), floor_plan.width_m)
        ny = min(max(old[1] + rng.uniform(-step, step), 0.0), floor_plan.height_m)
        positions[idx] = (nx, ny)

        cov = _coverage_fraction(positions, floor_plan, grid, tx_power_dbm, band_ghz,
                                  target_rssi_dbm, gamma, mask)
        delta = cov - best_cov
        accept = delta > 0 or rng.random() < math.exp(delta / max(temp, 1e-3))
        if accept:
            if cov > best_cov:
                best_cov = cov
                best_positions = list(positions)
        else:
            positions[idx] = old

        temp *= cooling
        history.append(best_cov)
        if best_cov >= 0.999:
            break

    return OptimizationResult(best_positions, best_cov, n_aps, history)


def genetic_algorithm_placement(floor_plan: FloorPlan, grid: SignalGrid, n_aps: int,
                                 tx_power_dbm: float = 20.0, band_ghz: float = 5.0,
                                 target_rssi_dbm: float = -67.0, gamma: float = 3.0,
                                 population_size: int = 24, generations: int = 40,
                                 mutation_rate: float = 0.2, seed: Optional[int] = None,
                                 walkable_mask: Optional[np.ndarray] = None) -> OptimizationResult:
    rng = random.Random(seed)
    mask = walkable_mask if walkable_mask is not None else default_walkable_mask(floor_plan, grid)

    def rand_individual() -> List[Tuple[float, float]]:
        return [(rng.uniform(0, floor_plan.width_m), rng.uniform(0, floor_plan.height_m))
                for _ in range(n_aps)]

    def fitness(ind: List[Tuple[float, float]]) -> float:
        return _coverage_fraction(ind, floor_plan, grid, tx_power_dbm, band_ghz,
                                   target_rssi_dbm, gamma, mask)

    population = [rand_individual() for _ in range(population_size)]
    history: List[float] = []
    best_ind, best_fit = None, -1.0

    for _ in range(generations):
        scored = sorted(((fitness(ind), ind) for ind in population), key=lambda t: -t[0])
        if scored[0][0] > best_fit:
            best_fit, best_ind = scored[0][0], scored[0][1]
        history.append(best_fit)
        if best_fit >= 0.999:
            break

        survivors = [ind for _, ind in scored[:max(2, population_size // 3)]]
        children = []
        while len(children) < population_size - len(survivors):
            pa, pb = rng.choice(survivors), rng.choice(survivors)
            cut = rng.randrange(1, n_aps) if n_aps > 1 else 1
            child = pa[:cut] + pb[cut:]
            if rng.random() < mutation_rate:
                mi = rng.randrange(n_aps)
                child[mi] = (rng.uniform(0, floor_plan.width_m), rng.uniform(0, floor_plan.height_m))
            children.append(child)
        population = survivors + children

    return OptimizationResult(best_ind or [], best_fit if best_fit >= 0 else 0.0, n_aps, history)


def find_minimal_ap_count(floor_plan: FloorPlan, grid: SignalGrid,
                           target_rssi_dbm: float = -67.0, target_coverage: float = 0.95,
                           tx_power_dbm: float = 20.0, band_ghz: float = 5.0, gamma: float = 3.0,
                           max_aps: int = 8, method: str = "annealing",
                           walkable_mask: Optional[np.ndarray] = None,
                           seed: Optional[int] = None, **search_kwargs) -> OptimizationResult:
    """Sweep n = 1..max_aps and return the first placement that satisfies
    the coverage target (best-effort result at max_aps otherwise).
    Extra ``search_kwargs`` (e.g. iterations, generations) are forwarded to
    the underlying search method.
    """
    fn = simulated_annealing_placement if method == "annealing" else genetic_algorithm_placement
    result = None
    for n in range(1, max_aps + 1):
        result = fn(floor_plan, grid, n, tx_power_dbm=tx_power_dbm, band_ghz=band_ghz,
                    target_rssi_dbm=target_rssi_dbm, gamma=gamma, seed=seed,
                    walkable_mask=walkable_mask, **search_kwargs)
        if result.coverage_fraction >= target_coverage:
            return result
    return result
