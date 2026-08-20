"""Headless CLI entry point for the Wi-Fi propagation & mesh simulator.

Loads a JSON floor plan, runs ray-traced coverage simulation (or the AP
placement optimizer, or mesh backhaul analysis), and writes raw numpy
arrays + matplotlib heatmap PNGs to --output-dir. No interactive windows.
"""
from __future__ import annotations

import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import mesh
import optimizer
import physics
from models import FloorPlan, SignalGrid
from raytracer import WallIndex, accumulate_multipath_grid, cast_ap_rays, combine_fields, direct_path_field


def build_grid(floor_plan: FloorPlan, resolution_m: float | None) -> SignalGrid:
    res = resolution_m or floor_plan.resolution_m
    return SignalGrid(floor_plan.width_m, floor_plan.height_m, res)


def run_coverage(floor_plan: FloorPlan, grid: SignalGrid, n_rays: int = 720,
                  max_bounces: int = 2, gamma: float = 3.0, multipath: bool = True) -> dict:
    """RSSI(dBm) map per AP, clipped to the physical [-95,-30] dBm range."""
    results = {}
    wall_index = WallIndex(floor_plan.walls, floor_plan.width_m, floor_plan.height_m) if multipath else None
    for ap in floor_plan.access_points:
        direct = direct_path_field(ap, floor_plan.walls, grid, gamma=gamma)
        if multipath:
            paths = cast_ap_rays(ap, floor_plan.walls, wall_index, n_rays=n_rays, max_bounces=max_bounces)
            mp_mw = accumulate_multipath_grid(grid, ap, paths)
            final = combine_fields(direct, mp_mw)
        else:
            final = direct
        results[ap.id] = np.clip(final, physics.NOISE_FLOOR_DBM, physics.MAX_RSSI_DBM)
    return results


def compute_sinr(rssi_maps: dict, aps) -> dict:
    """SINR(dB) per AP, accounting for co-channel interference (CCI) from
    other APs sharing the same band + channel."""
    sinr_maps = {}
    for ap in aps:
        signal = rssi_maps[ap.id]
        interference_mw = np.zeros_like(signal)
        for other in aps:
            if other.id == ap.id:
                continue
            if other.channel == ap.channel and abs(other.band_ghz - ap.band_ghz) < 0.01:
                interference_mw = interference_mw + physics.dbm_to_mw(rssi_maps[other.id])
        sinr_maps[ap.id] = physics.sinr_db(signal, interference_mw)
    return sinr_maps


def best_server_map(rssi_maps: dict):
    ids = list(rssi_maps.keys())
    stacked = np.stack([rssi_maps[i] for i in ids], axis=0)
    best_idx = np.argmax(stacked, axis=0)
    best_rssi = np.max(stacked, axis=0)
    return best_rssi, best_idx, ids


def save_heatmap(array: np.ndarray, floor_plan: FloorPlan, title: str, path: str,
                  vmin: float = -95, vmax: float = -30, cmap: str = "RdYlGn",
                  cbar_label: str = "dBm") -> None:
    fig, ax = plt.subplots(figsize=(10, 10 * floor_plan.height_m / max(floor_plan.width_m, 1e-6)))
    im = ax.imshow(array, extent=[0, floor_plan.width_m, 0, floor_plan.height_m],
                    origin="lower", cmap=cmap, vmin=vmin, vmax=vmax)
    for w in floor_plan.walls:
        ax.plot([w.x1, w.x2], [w.y1, w.y2], color="black", linewidth=max(1.0, w.thickness_m * 10))
    for ap in floor_plan.access_points:
        ax.plot(ap.x, ap.y, marker="^", color="blue", markersize=10, markeredgecolor="white")
        ax.annotate(ap.id, (ap.x, ap.y), textcoords="offset points", xytext=(6, 6), fontsize=8, color="blue")
    ax.set_title(title)
    ax.set_xlabel("meters"); ax.set_ylabel("meters")
    ax.set_aspect("equal")
    fig.colorbar(im, ax=ax, label=cbar_label, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def cmd_optimize(args, floor_plan: FloorPlan, grid: SignalGrid) -> None:
    result = optimizer.find_minimal_ap_count(
        floor_plan, grid, target_rssi_dbm=args.target_rssi, target_coverage=args.target_coverage,
        tx_power_dbm=args.tx_power, band_ghz=args.band, max_aps=args.max_aps, method=args.opt_method,
        seed=args.seed,
    )
    report = {
        "n_aps": result.n_aps,
        "coverage_fraction": result.coverage_fraction,
        "target_rssi_dbm": args.target_rssi,
        "target_coverage": args.target_coverage,
        "positions": result.positions,
        "converged": result.coverage_fraction >= args.target_coverage,
    }
    with open(os.path.join(args.output_dir, "optimization_report.json"), "w") as f:
        json.dump(report, f, indent=2)

    optimized_plan = FloorPlan(floor_plan.width_m, floor_plan.height_m, floor_plan.resolution_m,
                                walls=floor_plan.walls, access_points=[])
    from models import AccessPoint
    for i, (x, y) in enumerate(result.positions):
        optimized_plan.access_points.append(AccessPoint(
            id=f"opt_ap_{i+1}", x=x, y=y, tx_power_dbm=args.tx_power, band_ghz=args.band,
        ))
    optimized_plan.to_json(os.path.join(args.output_dir, "optimized_floorplan.json"))

    best = np.full(grid.shape, physics.NOISE_FLOOR_DBM)
    for ap in optimized_plan.access_points:
        best = np.maximum(best, direct_path_field(ap, floor_plan.walls, grid, gamma=args.gamma))
    save_heatmap(np.clip(best, physics.NOISE_FLOOR_DBM, physics.MAX_RSSI_DBM), floor_plan,
                 f"Optimized Placement ({result.n_aps} APs, {result.coverage_fraction*100:.1f}% >= {args.target_rssi} dBm)",
                 os.path.join(args.output_dir, "heatmap_optimized.png"))

    print(json.dumps(report, indent=2))


def cmd_simulate(args, floor_plan: FloorPlan, grid: SignalGrid) -> None:
    rssi_maps = run_coverage(floor_plan, grid, n_rays=args.n_rays, max_bounces=args.max_bounces,
                              gamma=args.gamma, multipath=not args.no_multipath)
    sinr_maps = compute_sinr(rssi_maps, floor_plan.access_points)
    best_rssi, best_idx, ap_ids = best_server_map(rssi_maps)

    np.save(os.path.join(args.output_dir, "rssi_best.npy"), best_rssi)
    np.save(os.path.join(args.output_dir, "best_server_index.npy"), best_idx)
    with open(os.path.join(args.output_dir, "ap_index.json"), "w") as f:
        json.dump(ap_ids, f, indent=2)

    for ap_id, arr in rssi_maps.items():
        np.save(os.path.join(args.output_dir, f"rssi_{ap_id}.npy"), arr)
        save_heatmap(arr, floor_plan, f"RSSI - {ap_id} ({floor_plan.access_points[0].band_ghz}GHz)",
                     os.path.join(args.output_dir, f"heatmap_rssi_{ap_id}.png"))
    for ap_id, arr in sinr_maps.items():
        np.save(os.path.join(args.output_dir, f"sinr_{ap_id}.npy"), arr)
        save_heatmap(arr, floor_plan, f"SINR - {ap_id}",
                     os.path.join(args.output_dir, f"heatmap_sinr_{ap_id}.png"),
                     vmin=-10, vmax=40, cmap="viridis", cbar_label="dB")

    save_heatmap(best_rssi, floor_plan, "Best-Server RSSI Coverage",
                 os.path.join(args.output_dir, "heatmap_best_rssi.png"))

    coverage_pct = float((best_rssi >= args.target_rssi).mean() * 100)
    coverage_report = {
        "target_rssi_dbm": args.target_rssi,
        "coverage_percent": coverage_pct,
        "mean_rssi_dbm": float(best_rssi.mean()),
        "aps": ap_ids,
        "grid_shape": list(best_rssi.shape),
        "resolution_m": grid.resolution_m,
    }
    with open(os.path.join(args.output_dir, "coverage_report.json"), "w") as f:
        json.dump(coverage_report, f, indent=2)
    print(json.dumps(coverage_report, indent=2))

    if args.mesh:
        cmd_mesh(args, floor_plan)


def cmd_mesh(args, floor_plan: FloorPlan) -> None:
    aps = floor_plan.access_points
    gateway_id = args.gateway_id or next((a.id for a in aps if a.is_gateway), None) or aps[0].id
    routing = mesh.build_routing_graph(aps, floor_plan.walls, gateway_id, gamma=args.gamma)
    aps_by_id = {a.id: a for a in aps}
    throughput = mesh.estimate_mesh_throughput(routing, aps_by_id)
    channels_24 = mesh.allocate_channels(aps, floor_plan.walls, 2.4)
    channels_5 = mesh.allocate_channels(aps, floor_plan.walls, 5.0)
    channels_6 = mesh.allocate_channels(aps, floor_plan.walls, 6.0)

    mesh_report = {
        "gateway": gateway_id,
        "routes": routing["routes"],
        "throughput": throughput,
        "channel_allocation": {"2.4ghz": channels_24, "5ghz": channels_5, "6ghz": channels_6},
    }
    with open(os.path.join(args.output_dir, "mesh_report.json"), "w") as f:
        json.dump(mesh_report, f, indent=2, default=str)
    print(json.dumps(mesh_report, indent=2, default=str))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Headless Wi-Fi propagation & mesh simulator")
    parser.add_argument("--floorplan", required=True, help="Path to floor plan JSON")
    parser.add_argument("--output-dir", default="./output")
    parser.add_argument("--resolution", type=float, default=None, help="Override grid resolution (m/cell)")
    parser.add_argument("--n-rays", type=int, default=720, help="Angular rays per AP (360-1440)")
    parser.add_argument("--max-bounces", type=int, default=2)
    parser.add_argument("--no-multipath", action="store_true", help="Skip reflection ray tracing (direct field only)")
    parser.add_argument("--gamma", type=float, default=None, help="Log-distance path-loss exponent")
    parser.add_argument("--environment", choices=list(physics.GAMMA_BY_ENVIRONMENT), default="office")
    parser.add_argument("--target-rssi", type=float, default=-67.0)
    parser.add_argument("--mesh", action="store_true", help="Also run mesh backhaul/routing analysis")
    parser.add_argument("--gateway-id", default=None)
    parser.add_argument("--optimize", action="store_true", help="Run AP placement optimizer instead of simulating")
    parser.add_argument("--max-aps", type=int, default=6)
    parser.add_argument("--target-coverage", type=float, default=0.95)
    parser.add_argument("--opt-method", choices=["annealing", "genetic"], default="annealing")
    parser.add_argument("--tx-power", type=float, default=20.0, help="TX power (dBm) for optimizer-placed APs")
    parser.add_argument("--band", type=float, default=5.0, help="Band (GHz) for optimizer-placed APs")
    parser.add_argument("--seed", type=int, default=None)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.gamma is None:
        args.gamma = physics.GAMMA_BY_ENVIRONMENT[args.environment]

    os.makedirs(args.output_dir, exist_ok=True)
    floor_plan = FloorPlan.from_json(args.floorplan)
    grid = build_grid(floor_plan, args.resolution)

    if args.optimize:
        cmd_optimize(args, floor_plan, grid)
    else:
        cmd_simulate(args, floor_plan, grid)


if __name__ == "__main__":
    main()
