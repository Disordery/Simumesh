"""Data structures for the Wi-Fi propagation & mesh simulator.

Shared by every other module (physics, raytracer, mesh, optimizer, main, and
the standalone cad_editor GUI) so that floor-plan JSON has a single schema.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional

import numpy as np

BANDS_GHZ = (2.4, 5.0, 6.0)


# --------------------------------------------------------------------------
# Materials
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class Material:
    """Per-band penetration loss (normal incidence) and reflection coefficient."""
    name: str
    attenuation_db: Dict[float, float]
    reflection_coeff: float = 0.3  # fraction of incident power reflected [0,1]

    def loss_db(self, band_ghz: float) -> float:
        nearest = min(self.attenuation_db, key=lambda b: abs(b - band_ghz))
        return self.attenuation_db[nearest]


MATERIAL_LIBRARY: Dict[str, Material] = {
    "drywall": Material("Drywall / Plasterboard", {2.4: 3.0, 5.0: 4.0, 6.0: 5.0}, 0.15),
    "brick": Material("Standard Brick", {2.4: 6.0, 5.0: 10.0, 6.0: 12.0}, 0.35),
    "concrete": Material("Reinforced Concrete", {2.4: 18.0, 5.0: 28.0, 6.0: 35.0}, 0.55),
    "glass": Material("Clear Glass", {2.4: 2.0, 5.0: 3.0, 6.0: 4.0}, 0.20),
    "low_e_glass": Material("Low-E / Tinted Glass", {2.4: 10.0, 5.0: 18.0, 6.0: 22.0}, 0.45),
    "wood": Material("Solid Wood", {2.4: 4.0, 5.0: 7.0, 6.0: 9.0}, 0.20),
    "metal": Material("Metal / Elevator Shaft", {2.4: 35.0, 5.0: 45.0, 6.0: 50.0}, 0.85),
}


def get_material(key: str) -> Material:
    return MATERIAL_LIBRARY.get(key, MATERIAL_LIBRARY["drywall"])


# --------------------------------------------------------------------------
# Walls
# --------------------------------------------------------------------------

@dataclass
class Wall:
    x1: float
    y1: float
    x2: float
    y2: float
    material_key: str = "drywall"
    thickness_m: float = 0.1

    @property
    def material(self) -> Material:
        return get_material(self.material_key)

    @property
    def p1(self) -> np.ndarray:
        return np.array([self.x1, self.y1], dtype=float)

    @property
    def p2(self) -> np.ndarray:
        return np.array([self.x2, self.y2], dtype=float)

    @property
    def vector(self) -> np.ndarray:
        return self.p2 - self.p1

    @property
    def length(self) -> float:
        return float(np.hypot(*self.vector))

    @property
    def normal(self) -> np.ndarray:
        dx, dy = self.vector
        n = np.array([-dy, dx])
        norm = np.linalg.norm(n)
        return n / norm if norm > 1e-12 else np.array([0.0, 0.0])


# --------------------------------------------------------------------------
# Access points / mesh radios
# --------------------------------------------------------------------------

@dataclass
class Radio:
    """A single radio inside a dual/tri-radio mesh node."""
    role: str = "access"        # "access" or "backhaul"
    band_ghz: float = 5.0
    channel: int = 36
    tx_power_dbm: float = 20.0


@dataclass
class AccessPoint:
    id: str
    x: float
    y: float
    tx_power_dbm: float = 20.0
    band_ghz: float = 5.0
    channel: int = 36
    antenna_azimuth_deg: float = 0.0
    antenna_beamwidth_deg: float = 360.0   # 360 = omnidirectional
    antenna_gain_dbi: float = 2.0
    is_gateway: bool = False
    dedicated_backhaul: bool = False       # True => dual/tri-radio node
    radios: List[Radio] = field(default_factory=list)

    @property
    def pos(self) -> np.ndarray:
        return np.array([self.x, self.y], dtype=float)

    def antenna_gain(self, azimuth_to_target_deg: float) -> float:
        """Scalar directional-antenna gain toward a single bearing."""
        if self.antenna_beamwidth_deg >= 359.9:
            return self.antenna_gain_dbi
        diff = abs(((azimuth_to_target_deg - self.antenna_azimuth_deg) + 180) % 360 - 180)
        half = self.antenna_beamwidth_deg / 2.0
        if diff <= half:
            return self.antenna_gain_dbi
        return self.antenna_gain_dbi - 0.3 * (diff - half)  # side-lobe rolloff

    def antenna_gain_vectorized(self, azimuth_to_target_deg: np.ndarray) -> np.ndarray:
        """Vectorized directional-antenna gain pattern for a grid of bearings."""
        az = np.asarray(azimuth_to_target_deg, dtype=float)
        if self.antenna_beamwidth_deg >= 359.9:
            return np.full_like(az, self.antenna_gain_dbi, dtype=float)
        diff = np.abs(((az - self.antenna_azimuth_deg) + 180) % 360 - 180)
        half = self.antenna_beamwidth_deg / 2.0
        return np.where(diff <= half, self.antenna_gain_dbi, self.antenna_gain_dbi - 0.3 * (diff - half))


# --------------------------------------------------------------------------
# Signal grid
# --------------------------------------------------------------------------

@dataclass
class SignalGrid:
    """Regular sample grid over the floor-plan rectangle."""
    width_m: float
    height_m: float
    resolution_m: float

    def __post_init__(self):
        self.nx = max(2, int(round(self.width_m / self.resolution_m)) + 1)
        self.ny = max(2, int(round(self.height_m / self.resolution_m)) + 1)
        xs = np.linspace(0.0, self.width_m, self.nx)
        ys = np.linspace(0.0, self.height_m, self.ny)
        self.xx, self.yy = np.meshgrid(xs, ys)          # shape (ny, nx)
        self.points = np.stack([self.xx.ravel(), self.yy.ravel()], axis=1)  # (N,2)

    @property
    def shape(self):
        return (self.ny, self.nx)


# --------------------------------------------------------------------------
# Ray tracer output structures
# --------------------------------------------------------------------------

@dataclass
class RaySegment:
    p1: np.ndarray
    p2: np.ndarray
    cumulative_loss_db: float
    bounce_index: int


# --------------------------------------------------------------------------
# Floor plan container + JSON I/O
# --------------------------------------------------------------------------

@dataclass
class FloorPlan:
    width_m: float
    height_m: float
    resolution_m: float = 0.1
    walls: List[Wall] = field(default_factory=list)
    access_points: List[AccessPoint] = field(default_factory=list)

    @classmethod
    def from_dict(cls, d: dict) -> "FloorPlan":
        dims = d["dimensions"]
        walls = [
            Wall(
                float(w["x1"]), float(w["y1"]), float(w["x2"]), float(w["y2"]),
                w.get("material", "drywall"), float(w.get("thickness_m", 0.1)),
            )
            for w in d.get("walls", [])
        ]
        aps = []
        for a in d.get("access_points", []):
            radios = [Radio(**r) for r in a.get("radios", [])]
            aps.append(AccessPoint(
                id=a["id"], x=float(a["x"]), y=float(a["y"]),
                tx_power_dbm=float(a.get("tx_power_dbm", 20.0)),
                band_ghz=float(a.get("band_ghz", 5.0)),
                channel=int(a.get("channel", 36)),
                antenna_azimuth_deg=float(a.get("antenna_azimuth_deg", 0.0)),
                antenna_beamwidth_deg=float(a.get("antenna_beamwidth_deg", 360.0)),
                antenna_gain_dbi=float(a.get("antenna_gain_dbi", 2.0)),
                is_gateway=bool(a.get("is_gateway", False)),
                dedicated_backhaul=bool(a.get("dedicated_backhaul", False)),
                radios=radios,
            ))
        return cls(
            width_m=float(dims["width_m"]),
            height_m=float(dims["height_m"]),
            resolution_m=float(dims.get("resolution_m", 0.1)),
            walls=walls,
            access_points=aps,
        )

    def to_dict(self) -> dict:
        return {
            "dimensions": {
                "width_m": self.width_m, "height_m": self.height_m,
                "resolution_m": self.resolution_m,
            },
            "walls": [
                {"x1": w.x1, "y1": w.y1, "x2": w.x2, "y2": w.y2,
                 "material": w.material_key, "thickness_m": w.thickness_m}
                for w in self.walls
            ],
            "access_points": [
                {
                    "id": a.id, "x": a.x, "y": a.y,
                    "tx_power_dbm": a.tx_power_dbm, "band_ghz": a.band_ghz,
                    "channel": a.channel,
                    "antenna_azimuth_deg": a.antenna_azimuth_deg,
                    "antenna_beamwidth_deg": a.antenna_beamwidth_deg,
                    "antenna_gain_dbi": a.antenna_gain_dbi,
                    "is_gateway": a.is_gateway,
                    "dedicated_backhaul": a.dedicated_backhaul,
                    "radios": [asdict(r) for r in a.radios],
                }
                for a in self.access_points
            ],
        }

    @classmethod
    def from_json(cls, path: str) -> "FloorPlan":
        with open(path, "r") as f:
            return cls.from_dict(json.load(f))

    def to_json(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
