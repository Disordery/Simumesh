"""Graph-based mesh backhaul topology analyzer.

Covers point-to-point backhaul link budgets, greedy co-channel allocation,
Dijkstra/MST routing to a gateway, and per-hop airtime-contention throughput
degradation for single-radio vs dedicated dual/tri-radio backhaul nodes.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra, minimum_spanning_tree

import physics
from models import AccessPoint, Wall
from raytracer import point_to_point_path_loss

CHANNELS_24GHZ = [1, 6, 11]
CHANNELS_5GHZ = [36, 40, 44, 48, 149, 153, 157, 161]
CHANNELS_6GHZ = [1, 5, 9, 13, 17, 21, 25, 29, 33, 37]
CHANNEL_SETS = {2.4: CHANNELS_24GHZ, 5.0: CHANNELS_5GHZ, 6.0: CHANNELS_6GHZ}

MIN_VIABLE_SNR_DB = 6.0  # links weaker than this are treated as unusable


@dataclass
class MeshLink:
    node_a: str
    node_b: str
    rssi_dbm: float
    distance_m: float
    band_ghz: float
    phy_rate_mbps: float


def backhaul_link_budget(a: AccessPoint, b: AccessPoint, walls: List[Wall],
                          gamma: float = 2.5, model: str = "log_distance") -> MeshLink:
    """Point-to-point RSSI/PHY-rate between two mesh nodes through interior walls."""
    band = min(a.band_ghz, b.band_ghz)
    pl = point_to_point_path_loss(a.pos, b.pos, walls, band, gamma=gamma, model=model)
    rssi = physics.rssi_dbm(min(a.tx_power_dbm, b.tx_power_dbm), pl)
    snr = rssi - physics.NOISE_FLOOR_DBM
    mcs = physics.mcs_lookup(snr, bandwidth_mhz=80, spatial_streams=2)
    dist = float(np.hypot(*(a.pos - b.pos)))
    return MeshLink(a.id, b.id, float(rssi), dist, band, mcs["phy_rate_mbps"])


# --------------------------------------------------------------------------
# Channel allocation (greedy graph coloring on the interference graph)
# --------------------------------------------------------------------------

def allocate_channels(aps: List[AccessPoint], walls: List[Wall], band_ghz: float,
                       interference_threshold_dbm: float = -75.0) -> Dict[str, int]:
    band_aps = [ap for ap in aps if abs(ap.band_ghz - band_ghz) < 0.01]
    channels = CHANNEL_SETS.get(band_ghz, CHANNELS_5GHZ)
    n = len(band_aps)
    if n == 0:
        return {}
    conflict = np.zeros((n, n), dtype=bool)
    for i in range(n):
        for j in range(i + 1, n):
            link = backhaul_link_budget(band_aps[i], band_aps[j], walls)
            if link.rssi_dbm > interference_threshold_dbm:
                conflict[i, j] = conflict[j, i] = True

    assigned: Dict[str, int] = {}
    order = np.argsort(-conflict.sum(axis=1))  # most-constrained-first heuristic
    for idx in order:
        ap = band_aps[idx]
        neighbor_channels = {
            assigned[band_aps[k].id]
            for k in range(n) if conflict[idx, k] and band_aps[k].id in assigned
        }
        choice = next((c for c in channels if c not in neighbor_channels), channels[0])
        assigned[ap.id] = choice
    return assigned


# --------------------------------------------------------------------------
# Routing graph (Dijkstra shortest path + MST) to the gateway
# --------------------------------------------------------------------------

def build_routing_graph(aps: List[AccessPoint], walls: List[Wall], gateway_id: str,
                         gamma: float = 2.5) -> dict:
    n = len(aps)
    idx = {ap.id: i for i, ap in enumerate(aps)}
    cost = np.zeros((n, n))
    links: Dict[Tuple[str, str], MeshLink] = {}

    for i in range(n):
        for j in range(i + 1, n):
            link = backhaul_link_budget(aps[i], aps[j], walls, gamma=gamma)
            snr = link.rssi_dbm - physics.NOISE_FLOOR_DBM
            if snr > MIN_VIABLE_SNR_DB:
                weight = 1000.0 / max(link.phy_rate_mbps, 0.1)  # latency-proxy cost
                cost[i, j] = cost[j, i] = weight
                links[(aps[i].id, aps[j].id)] = link
                links[(aps[j].id, aps[i].id)] = link

    graph = csr_matrix(cost)
    gw_idx = idx[gateway_id]
    dist, predecessors = dijkstra(graph, directed=False, indices=gw_idx, return_predecessors=True)
    mst = minimum_spanning_tree(graph)

    routes: Dict[str, List[str]] = {}
    for ap in aps:
        if ap.id == gateway_id:
            continue
        path = []
        cur = idx[ap.id]
        visited_guard = 0
        while cur != gw_idx and cur != -9999 and visited_guard <= n:
            path.append(aps[cur].id)
            cur = predecessors[cur]
            visited_guard += 1
        if cur == gw_idx:
            path.append(gateway_id)
            path.reverse()
            routes[ap.id] = path
        else:
            routes[ap.id] = []  # unreachable

    return {"distances": dist, "routes": routes, "mst": mst, "links": links, "idx": idx}


# --------------------------------------------------------------------------
# Airtime contention & hop-penalty throughput
# --------------------------------------------------------------------------

def hop_degraded_throughput(base_phy_mbps: float, hop_count: int, dedicated_backhaul: bool) -> float:
    """Dedicated (dual/tri-radio) backhaul avoids shared-airtime contention;
    single-radio mesh nodes must split airtime between receiving and
    forwarding, roughly halving throughput per additional hop (TDMA-style
    contention)."""
    if hop_count <= 0:
        return base_phy_mbps
    if dedicated_backhaul:
        return base_phy_mbps
    return base_phy_mbps * (0.5 ** (hop_count - 1))


def estimate_mesh_throughput(routing: dict, aps_by_id: Dict[str, AccessPoint]) -> Dict[str, dict]:
    """Bottleneck-link throughput to the gateway for every non-gateway node,
    degraded by per-hop airtime contention (per-node radio configuration)."""
    results: Dict[str, dict] = {}
    for ap_id, path in routing["routes"].items():
        if not path:
            results[ap_id] = {"hops": None, "path": [], "bottleneck_phy_mbps": 0.0,
                               "effective_throughput_mbps": 0.0, "reachable": False}
            continue
        hop_count = len(path) - 1
        rates = []
        for a, b in zip(path[:-1], path[1:]):
            link = routing["links"].get((a, b))
            if link:
                rates.append(link.phy_rate_mbps)
        bottleneck = min(rates) if rates else 0.0
        dedicated = aps_by_id[ap_id].dedicated_backhaul
        effective = hop_degraded_throughput(bottleneck, hop_count, dedicated)
        results[ap_id] = {
            "hops": hop_count, "path": path,
            "bottleneck_phy_mbps": round(bottleneck, 1),
            "effective_throughput_mbps": round(effective, 1),
            "reachable": True,
        }
    return results
