# Wi-Fi Propagation & Mesh Simulator

A physics-grounded 2D Wi-Fi signal attenuation and mesh backhaul simulator.
The compute engine (`models.py`, `physics.py`, `raytracer.py`, `mesh.py`,
`optimizer.py`, `main.py`) is entirely headless and runs from the CLI. A
standalone Tkinter GUI (`cad_editor.py`) is provided separately for drawing
floor plans, and shares only the JSON data schema with the engine.

## Install

```bash
pip install -r requirements.txt
```

`numba` is optional (JIT-accelerates the ray tracer); everything works
without it, just slower on very dense wall layouts. `tkinter` is required
only for `cad_editor.py` - see `requirements.txt` for install notes on
Debian/Ubuntu.

## Architecture

| File | Responsibility |
|---|---|
| `models.py` | `Material`, `Wall`, `AccessPoint`, `SignalGrid`, `FloorPlan` dataclasses + JSON I/O |
| `physics.py` | FSPL, log-distance shadowing, COST231 multi-wall loss, SINR, Shannon capacity, MCS lookup |
| `raytracer.py` | Vectorized direct-path field, bucket-grid spatial index, Numba-JIT ray/wall intersection, multipath reflection tracer |
| `mesh.py` | Backhaul link budgets, channel allocation, Dijkstra/MST routing, hop-penalty throughput |
| `optimizer.py` | Simulated annealing / genetic AP placement search, minimal-AP-count sweep |
| `cad_editor.py` | Standalone Tkinter floor plan editor (walls, APs, JSON export/import) |
| `main.py` | Headless CLI: simulate coverage, run mesh analysis, or run the optimizer |

## Quick start

Draw a floor plan visually:

```bash
python3 cad_editor.py
```

...or hand-author one (see `sample_floorplan.json` and the schema below),
then simulate:

```bash
python3 main.py --floorplan sample_floorplan.json --output-dir output \
    --n-rays 720 --max-bounces 2 --environment office --mesh
```

This writes to `output/`:

* `rssi_<ap_id>.npy`, `sinr_<ap_id>.npy` - raw per-AP arrays (dBm / dB)
* `rssi_best.npy`, `best_server_index.npy` - best-server coverage + which AP wins each cell
* `heatmap_*.png` - matplotlib coverage/SINR renders with walls and APs overlaid
* `coverage_report.json` - summary stats (% of area above `--target-rssi`)
* `mesh_report.json` (with `--mesh`) - routing tree, per-node throughput, channel plan

Find the minimal AP count for 95% coverage at -67 dBm:

```bash
python3 main.py --floorplan sample_floorplan.json --output-dir output \
    --optimize --max-aps 6 --target-rssi -67 --target-coverage 0.95 --opt-method annealing
```

Writes `optimization_report.json`, `optimized_floorplan.json` (ready to
re-simulate or reload in the CAD editor), and `heatmap_optimized.png`.

### CLI reference

```
--floorplan PATH        Input floor plan JSON (required)
--output-dir DIR        Output directory (default ./output)
--resolution M           Override grid resolution, meters/cell
--n-rays N               Angular rays per AP for multipath, 360-1440 (default 720)
--max-bounces N          Max specular reflections per ray (default 2)
--no-multipath           Skip reflection tracing; direct/transmitted field only
--gamma G                Log-distance path-loss exponent (overrides --environment)
--environment {free_space,open_office,office,residential,dense_obstruction}
--target-rssi DBM        Coverage threshold (default -67.0)
--mesh                   Also run mesh backhaul/routing analysis
--gateway-id ID          Gateway AP id (default: first AP with is_gateway=true, else first AP)
--optimize               Run the placement optimizer instead of simulating
--max-aps N              Max AP count to search (default 6)
--target-coverage F       Fraction of area required above --target-rssi (default 0.95)
--opt-method {annealing,genetic}
--tx-power DBM, --band GHZ   TX power / band for optimizer-placed APs
--seed N                 RNG seed for reproducible optimizer runs
```

## Floor plan JSON schema

```json
{
  "dimensions": {"width_m": 20.0, "height_m": 15.0, "resolution_m": 0.05},
  "walls": [
    {"x1": 0.0, "y1": 0.0, "x2": 20.0, "y2": 0.0, "material": "concrete", "thickness_m": 0.2}
  ],
  "access_points": [
    {"id": "router_1", "x": 10.0, "y": 7.5, "tx_power_dbm": 20, "band_ghz": 5.0, "channel": 36,
     "antenna_azimuth_deg": 0, "antenna_beamwidth_deg": 360, "antenna_gain_dbi": 2.0,
     "is_gateway": true, "dedicated_backhaul": true}
  ]
}
```

Material keys: `drywall`, `brick`, `concrete`, `glass`, `low_e_glass`, `wood`, `metal`
(see `MATERIAL_LIBRARY` in `models.py` for the per-band dB values).

## Physics model

* **FSPL**: `20*log10(d_km) + 20*log10(f_MHz) + 32.44`
* **Log-distance shadowing**: `PL(d) = PL(1m) + 10*gamma*log10(d)`, gamma set per
  `--environment` (or `--gamma` directly).
* **COST231 Multi-Wall Model**: total path loss = base propagation loss above
  + sum of angle-adjusted wall penetration losses for every wall genuinely
  crossed between transmitter and receive point.
* **Angle-dependent wall loss**: `L_eff = L_material / cos(theta)`, incidence
  angle capped at 80° to avoid the 1/cos singularity at grazing incidence.
* **Multipath**: `raytracer.cast_ap_rays` fires 360-1440 angular rays per AP,
  each undergoing specular reflection off wall surfaces (loss = material
  reflection coefficient) for up to `--max-bounces` bounces. Reflected power
  is deposited into the grid and summed incoherently with the direct field
  by default; `accumulate_multipath_grid(..., coherent=True)` sums complex
  amplitudes using carrier phase (`physics.path_phase`) instead.
* **SINR**: computed from co-channel interference (APs sharing band+channel)
  summed in the linear domain against a -95 dBm noise floor.
* **PHY rate**: Shannon capacity (`physics.shannon_capacity_mbps`) and an
  MCS index lookup table (`physics.mcs_lookup`) scaled by bandwidth/streams.

### Known simplifications

* No explicit edge-diffraction (UTD/GTD) model - only direct transmission +
  specular reflection are traced, so some finely discretized shadow
  boundaries near wall edges can show sampling-noise speckle at low ray
  counts. Increase `--n-rays` to reduce this.
* Reflection is single-bounce-per-surface specular; diffuse scattering is
  not modeled.
* Coherent multipath summation is available but off by default, since
  sub-wavelength position sensitivity is not meaningful at typical grid
  resolutions.

## Mesh backhaul analysis (`mesh.py`)

* `backhaul_link_budget` - point-to-point RSSI/PHY-rate between two nodes
  through interior walls.
* `allocate_channels` - greedy graph-coloring channel assignment per band
  from an interference graph (RSSI threshold between AP pairs).
* `build_routing_graph` - Dijkstra shortest path + minimum spanning tree
  (via `scipy.sparse.csgraph`) to a designated gateway, edge weight =
  inverse PHY rate (latency proxy).
* `estimate_mesh_throughput` - bottleneck-link throughput per node, with
  `hop_degraded_throughput` applying a 50%-per-hop airtime-contention
  penalty for single-radio (shared) backhaul nodes, versus no penalty for
  `dedicated_backhaul=true` (dual/tri-radio) nodes.

## Optimizer (`optimizer.py`)

`simulated_annealing_placement` and `genetic_algorithm_placement` search AP
coordinates to maximize the fraction of walkable area above a target RSSI,
using the fast vectorized direct-path field (multipath is skipped during
search for speed). `find_minimal_ap_count` sweeps AP count upward until the
coverage target is met or `--max-aps` is reached.

Optimization runs on the grid resolution you pass in - use a coarser
`--resolution` (e.g. 0.4-0.5m) for faster search on large floor plans, then
re-simulate the winning layout at full resolution.

## CAD editor (`cad_editor.py`)

```bash
python3 cad_editor.py
```

* **Wall tool**: click-drag to draw a wall snapped to the grid resolution;
  pick material + thickness from the toolbar before drawing.
* **Access Point tool**: click to place, opens a properties dialog (TX
  power, band, channel, antenna gain/azimuth/beamwidth, gateway/dedicated
  backhaul flags).
* **Select / Delete**: click a wall or AP to select it, `Delete`/`Backspace`
  to remove. Double-click an AP to edit its properties.
* **File > New/Open/Save/Save As** - reads and writes the same JSON schema
  `main.py` consumes.

## Performance notes

* `direct_path_field` is fully vectorized (chunked AP-to-every-grid-cell
  intersection against all walls) - a few hundred ms even for ~50k grid
  points against ~80 walls.
* The multipath ray tracer uses a uniform-bucket spatial index
  (`WallIndex`) for candidate wall lookup per ray, and JIT-compiles the
  innermost intersection kernel with Numba when available.
* 1440 rays x 4 bounces against ~80 walls typically completes in well
  under a second per AP after JIT warm-up.
