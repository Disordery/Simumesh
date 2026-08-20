"""Standalone visual CAD layout editor.

A lightweight Tkinter GUI for drawing floor plans (walls + access points)
and exporting/importing them as JSON compatible with main.py. This file has
no dependency on physics.py, raytracer.py, mesh.py, or optimizer.py - it
only shares the data schema (models.py) with the compute engine.

Run directly:  python3 cad_editor.py
"""
from __future__ import annotations

import json
import os
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from models import MATERIAL_LIBRARY, AccessPoint, FloorPlan, Wall

MATERIAL_COLORS = {
    "drywall": "#c9c2b4",
    "brick": "#a5533f",
    "concrete": "#6e6e6e",
    "glass": "#8fd3e8",
    "low_e_glass": "#3f8fa0",
    "wood": "#8b5a2b",
    "metal": "#2b2b2b",
}

BAND_CHOICES = ["2.4", "5.0", "6.0"]
DEFAULT_PPM = 30.0          # pixels per meter
GRID_LINE_COLOR = "#e4e4e4"
GRID_MAJOR_COLOR = "#c9c9c9"
AP_RADIUS_PX = 8
HANDLE_RADIUS_PX = 5
SNAP_TOLERANCE_M = 0.05


class ApPropertiesDialog(tk.Toplevel):
    """Modal dialog for creating/editing an AccessPoint's properties."""

    def __init__(self, parent, ap: AccessPoint | None, existing_ids: set[str]):
        super().__init__(parent)
        self.title("Access Point Properties")
        self.resizable(False, False)
        self.result: AccessPoint | None = None
        self._existing_ids = existing_ids - ({ap.id} if ap else set())
        self.grab_set()

        pad = {"padx": 8, "pady": 4}
        r = 0

        def row(label, widget):
            nonlocal r
            ttk.Label(self, text=label).grid(row=r, column=0, sticky="w", **pad)
            widget.grid(row=r, column=1, sticky="ew", **pad)
            r += 1

        self.id_var = tk.StringVar(value=ap.id if ap else self._suggest_id())
        row("ID", ttk.Entry(self, textvariable=self.id_var))

        self.power_var = tk.DoubleVar(value=ap.tx_power_dbm if ap else 20.0)
        row("TX Power (dBm)", ttk.Spinbox(self, from_=0, to=30, increment=1, textvariable=self.power_var))

        self.band_var = tk.StringVar(value=str(ap.band_ghz) if ap else "5.0")
        row("Band (GHz)", ttk.Combobox(self, textvariable=self.band_var, values=BAND_CHOICES, state="readonly"))

        self.channel_var = tk.IntVar(value=ap.channel if ap else 36)
        row("Channel", ttk.Entry(self, textvariable=self.channel_var))

        self.gain_var = tk.DoubleVar(value=ap.antenna_gain_dbi if ap else 2.0)
        row("Antenna Gain (dBi)", ttk.Spinbox(self, from_=0, to=20, increment=0.5, textvariable=self.gain_var))

        self.beamwidth_var = tk.DoubleVar(value=ap.antenna_beamwidth_deg if ap else 360.0)
        row("Beamwidth (deg, 360=omni)", ttk.Spinbox(self, from_=10, to=360, increment=5, textvariable=self.beamwidth_var))

        self.azimuth_var = tk.DoubleVar(value=ap.antenna_azimuth_deg if ap else 0.0)
        row("Azimuth (deg)", ttk.Spinbox(self, from_=0, to=360, increment=5, textvariable=self.azimuth_var))

        self.gateway_var = tk.BooleanVar(value=ap.is_gateway if ap else False)
        row("Gateway node", ttk.Checkbutton(self, variable=self.gateway_var))

        self.dedicated_var = tk.BooleanVar(value=ap.dedicated_backhaul if ap else False)
        row("Dedicated backhaul radio", ttk.Checkbutton(self, variable=self.dedicated_var))

        btns = ttk.Frame(self)
        btns.grid(row=r, column=0, columnspan=2, pady=(10, 8))
        ttk.Button(btns, text="Cancel", command=self.destroy).pack(side="left", padx=6)
        ttk.Button(btns, text="Save", command=self._on_save).pack(side="left", padx=6)

        self.columnconfigure(1, weight=1)
        self._orig_pos = (ap.x, ap.y) if ap else (0.0, 0.0)

    def _suggest_id(self) -> str:
        n = 1
        while f"ap_{n}" in self._existing_ids:
            n += 1
        return f"ap_{n}"

    def _on_save(self):
        new_id = self.id_var.get().strip()
        if not new_id:
            messagebox.showerror("Invalid ID", "AP ID cannot be empty.")
            return
        if new_id in self._existing_ids:
            messagebox.showerror("Duplicate ID", f"An AP with ID '{new_id}' already exists.")
            return
        try:
            self.result = AccessPoint(
                id=new_id, x=self._orig_pos[0], y=self._orig_pos[1],
                tx_power_dbm=float(self.power_var.get()),
                band_ghz=float(self.band_var.get()),
                channel=int(self.channel_var.get()),
                antenna_azimuth_deg=float(self.azimuth_var.get()),
                antenna_beamwidth_deg=float(self.beamwidth_var.get()),
                antenna_gain_dbi=float(self.gain_var.get()),
                is_gateway=bool(self.gateway_var.get()),
                dedicated_backhaul=bool(self.dedicated_var.get()),
            )
        except (ValueError, tk.TclError) as exc:
            messagebox.showerror("Invalid value", str(exc))
            return
        self.destroy()


class NewPlanDialog(tk.Toplevel):
    """Modal dialog for setting up a new floor plan's dimensions."""

    def __init__(self, parent, width_m=20.0, height_m=15.0, resolution_m=0.1):
        super().__init__(parent)
        self.title("New Floor Plan")
        self.resizable(False, False)
        self.result = None
        self.grab_set()

        pad = {"padx": 8, "pady": 4}
        self.w_var = tk.DoubleVar(value=width_m)
        self.h_var = tk.DoubleVar(value=height_m)
        self.res_var = tk.DoubleVar(value=resolution_m)

        ttk.Label(self, text="Width (m)").grid(row=0, column=0, sticky="w", **pad)
        ttk.Entry(self, textvariable=self.w_var).grid(row=0, column=1, **pad)
        ttk.Label(self, text="Height (m)").grid(row=1, column=0, sticky="w", **pad)
        ttk.Entry(self, textvariable=self.h_var).grid(row=1, column=1, **pad)
        ttk.Label(self, text="Grid resolution (m)").grid(row=2, column=0, sticky="w", **pad)
        ttk.Entry(self, textvariable=self.res_var).grid(row=2, column=1, **pad)

        btns = ttk.Frame(self)
        btns.grid(row=3, column=0, columnspan=2, pady=(10, 8))
        ttk.Button(btns, text="Cancel", command=self.destroy).pack(side="left", padx=6)
        ttk.Button(btns, text="Create", command=self._on_create).pack(side="left", padx=6)

    def _on_create(self):
        try:
            self.result = (float(self.w_var.get()), float(self.h_var.get()), float(self.res_var.get()))
        except (ValueError, tk.TclError):
            messagebox.showerror("Invalid value", "Dimensions must be numeric.")
            return
        self.destroy()


class CADEditor:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Wi-Fi Floor Plan CAD Editor")
        self.floor_plan = FloorPlan(width_m=20.0, height_m=15.0, resolution_m=0.1)
        self.current_path: str | None = None

        self.ppm = DEFAULT_PPM
        self.mode = tk.StringVar(value="wall")
        self.material_var = tk.StringVar(value="drywall")
        self.thickness_var = tk.DoubleVar(value=0.1)
        self.status_var = tk.StringVar(value="Ready")
        self.show_grid_var = tk.BooleanVar(value=True)

        self._drag_start_m = None
        self._drag_preview_id = None
        self.selected = None  # ("wall"|"ap", index_or_id)

        self._build_menu()
        self._build_toolbar()
        self._build_canvas()
        self._build_statusbar()
        self.redraw()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_menu(self):
        menubar = tk.Menu(self.root)
        filemenu = tk.Menu(menubar, tearoff=0)
        filemenu.add_command(label="New Plan...", command=self.new_plan_dialog)
        filemenu.add_command(label="Open JSON...", command=self.load_json)
        filemenu.add_command(label="Save", command=self.save_json)
        filemenu.add_command(label="Save As...", command=self.save_json_as)
        filemenu.add_separator()
        filemenu.add_command(label="Quit", command=self.root.quit)
        menubar.add_cascade(label="File", menu=filemenu)
        self.root.config(menu=menubar)

    def _build_toolbar(self):
        bar = ttk.Frame(self.root, padding=6)
        bar.pack(side="top", fill="x")

        ttk.Radiobutton(bar, text="Wall", variable=self.mode, value="wall").pack(side="left", padx=4)
        ttk.Radiobutton(bar, text="Access Point", variable=self.mode, value="ap").pack(side="left", padx=4)
        ttk.Radiobutton(bar, text="Select / Delete", variable=self.mode, value="select").pack(side="left", padx=12)

        ttk.Separator(bar, orient="vertical").pack(side="left", fill="y", padx=10)

        ttk.Label(bar, text="Material:").pack(side="left")
        mat_combo = ttk.Combobox(bar, textvariable=self.material_var, state="readonly",
                                  values=list(MATERIAL_LIBRARY.keys()), width=12)
        mat_combo.pack(side="left", padx=4)

        ttk.Label(bar, text="Thickness (m):").pack(side="left", padx=(8, 0))
        ttk.Spinbox(bar, from_=0.02, to=0.5, increment=0.01, textvariable=self.thickness_var, width=6).pack(side="left")

        ttk.Separator(bar, orient="vertical").pack(side="left", fill="y", padx=10)

        ttk.Button(bar, text="Zoom -", command=lambda: self._zoom(0.8)).pack(side="left", padx=2)
        ttk.Button(bar, text="Zoom +", command=lambda: self._zoom(1.25)).pack(side="left", padx=2)
        ttk.Checkbutton(bar, text="Grid", variable=self.show_grid_var, command=self.redraw).pack(side="left", padx=8)

        ttk.Separator(bar, orient="vertical").pack(side="left", fill="y", padx=10)
        ttk.Button(bar, text="Open JSON", command=self.load_json).pack(side="left", padx=2)
        ttk.Button(bar, text="Save JSON", command=self.save_json).pack(side="left", padx=2)

    def _build_canvas(self):
        frame = ttk.Frame(self.root)
        frame.pack(side="top", fill="both", expand=True)
        self.canvas = tk.Canvas(frame, bg="white", width=800, height=600, cursor="crosshair")
        hbar = ttk.Scrollbar(frame, orient="horizontal", command=self.canvas.xview)
        vbar = ttk.Scrollbar(frame, orient="vertical", command=self.canvas.yview)
        self.canvas.configure(xscrollcommand=hbar.set, yscrollcommand=vbar.set)
        self.canvas.grid(row=0, column=0, sticky="nsew")
        vbar.grid(row=0, column=1, sticky="ns")
        hbar.grid(row=1, column=0, sticky="ew")
        frame.rowconfigure(0, weight=1)
        frame.columnconfigure(0, weight=1)

        self.canvas.bind("<ButtonPress-1>", self.on_canvas_press)
        self.canvas.bind("<B1-Motion>", self.on_canvas_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_canvas_release)
        self.canvas.bind("<Motion>", self.on_canvas_hover)
        self.canvas.bind("<Double-Button-1>", self.on_canvas_double_click)
        self.root.bind("<Delete>", self.on_delete_key)
        self.root.bind("<BackSpace>", self.on_delete_key)

    def _build_statusbar(self):
        bar = ttk.Frame(self.root, padding=(6, 2))
        bar.pack(side="bottom", fill="x")
        ttk.Label(bar, textvariable=self.status_var).pack(side="left")

    # ------------------------------------------------------------------
    # Coordinate transforms
    # ------------------------------------------------------------------

    def m_to_px(self, x_m: float, y_m: float) -> tuple[float, float]:
        return x_m * self.ppm, y_m * self.ppm

    def px_to_m(self, x_px: float, y_px: float) -> tuple[float, float]:
        return x_px / self.ppm, y_px / self.ppm

    def _snap(self, x_m: float, y_m: float) -> tuple[float, float]:
        res = max(self.floor_plan.resolution_m, 0.01)
        return round(x_m / res) * res, round(y_m / res) * res

    def _zoom(self, factor: float):
        self.ppm = max(4.0, min(200.0, self.ppm * factor))
        self.redraw()

    # ------------------------------------------------------------------
    # Drawing
    # ------------------------------------------------------------------

    def redraw(self):
        self.canvas.delete("all")
        w_px, h_px = self.m_to_px(self.floor_plan.width_m, self.floor_plan.height_m)
        self.canvas.configure(scrollregion=(0, 0, w_px + 40, h_px + 40))

        if self.show_grid_var.get():
            self._draw_grid(w_px, h_px)

        self.canvas.create_rectangle(0, 0, w_px, h_px, outline="#999999", width=2)

        for i, wall in enumerate(self.floor_plan.walls):
            self._draw_wall(i, wall)
        for ap in self.floor_plan.access_points:
            self._draw_ap(ap)

    def _draw_grid(self, w_px, h_px):
        step_m = self.floor_plan.resolution_m
        while step_m * self.ppm < 12:   # avoid overdraw at low zoom
            step_m *= 5
        n_x = int(self.floor_plan.width_m / step_m) + 1
        n_y = int(self.floor_plan.height_m / step_m) + 1
        for i in range(n_x + 1):
            x_px, _ = self.m_to_px(i * step_m, 0)
            major = (i % 5 == 0)
            self.canvas.create_line(x_px, 0, x_px, h_px,
                                     fill=GRID_MAJOR_COLOR if major else GRID_LINE_COLOR)
        for j in range(n_y + 1):
            _, y_px = self.m_to_px(0, j * step_m)
            major = (j % 5 == 0)
            self.canvas.create_line(0, y_px, w_px, y_px,
                                     fill=GRID_MAJOR_COLOR if major else GRID_LINE_COLOR)

    def _draw_wall(self, index: int, wall: Wall):
        x1, y1 = self.m_to_px(wall.x1, wall.y1)
        x2, y2 = self.m_to_px(wall.x2, wall.y2)
        color = MATERIAL_COLORS.get(wall.material_key, "#000000")
        width_px = max(2, wall.thickness_m * self.ppm * 2)
        is_selected = self.selected == ("wall", index)
        self.canvas.create_line(x1, y1, x2, y2, fill=color, width=width_px,
                                 capstyle="round", tags=(f"wall_{index}",))
        if is_selected:
            self.canvas.create_line(x1, y1, x2, y2, fill="#2b7de9", width=2, dash=(4, 2))

    def _draw_ap(self, ap: AccessPoint):
        x, y = self.m_to_px(ap.x, ap.y)
        is_selected = self.selected == ("ap", ap.id)
        color = "#d94f4f" if ap.is_gateway else "#2b7de9"
        self.canvas.create_oval(x - AP_RADIUS_PX, y - AP_RADIUS_PX, x + AP_RADIUS_PX, y + AP_RADIUS_PX,
                                 fill=color, outline="white", width=2, tags=(f"ap_{ap.id}",))
        if ap.antenna_beamwidth_deg < 359.9:
            self._draw_beam_cone(x, y, ap)
        label = f"{ap.id} ({ap.band_ghz}GHz/ch{ap.channel})"
        self.canvas.create_text(x + AP_RADIUS_PX + 4, y, text=label, anchor="w", font=("TkDefaultFont", 8))
        if is_selected:
            self.canvas.create_oval(x - AP_RADIUS_PX - 3, y - AP_RADIUS_PX - 3,
                                     x + AP_RADIUS_PX + 3, y + AP_RADIUS_PX + 3,
                                     outline="#2b7de9", width=2, dash=(3, 2))

    def _draw_beam_cone(self, x, y, ap: AccessPoint):
        import math
        r = 3.0 * self.ppm
        half = ap.antenna_beamwidth_deg / 2.0
        a0 = math.radians(ap.antenna_azimuth_deg - half)
        a1 = math.radians(ap.antenna_azimuth_deg + half)
        p1 = (x + r * math.cos(a0), y + r * math.sin(a0))
        p2 = (x + r * math.cos(a1), y + r * math.sin(a1))
        self.canvas.create_line(x, y, *p1, fill="#2b7de9", width=1, dash=(2, 2))
        self.canvas.create_line(x, y, *p2, fill="#2b7de9", width=1, dash=(2, 2))

    # ------------------------------------------------------------------
    # Mouse interaction
    # ------------------------------------------------------------------

    def _event_to_m(self, event) -> tuple[float, float]:
        x_px = self.canvas.canvasx(event.x)
        y_px = self.canvas.canvasy(event.y)
        return self.px_to_m(x_px, y_px)

    def on_canvas_hover(self, event):
        x_m, y_m = self._event_to_m(event)
        self.status_var.set(f"x={x_m:.2f} m, y={y_m:.2f} m   |   mode={self.mode.get()}")

    def on_canvas_press(self, event):
        x_m, y_m = self._event_to_m(event)
        x_m, y_m = self._snap(x_m, y_m)
        mode = self.mode.get()

        if mode == "wall":
            self._drag_start_m = (x_m, y_m)
        elif mode == "ap":
            existing_ids = {a.id for a in self.floor_plan.access_points}
            dlg = ApPropertiesDialog(self.root, None, existing_ids)
            dlg._orig_pos = (x_m, y_m)
            self.root.wait_window(dlg)
            if dlg.result is not None:
                dlg.result.x, dlg.result.y = x_m, y_m
                self.floor_plan.access_points.append(dlg.result)
                self.redraw()
        elif mode == "select":
            self._select_at(x_m, y_m)
            self.redraw()

    def on_canvas_drag(self, event):
        if self.mode.get() != "wall" or self._drag_start_m is None:
            return
        x_m, y_m = self._event_to_m(event)
        x_m, y_m = self._snap(x_m, y_m)
        x0, y0 = self.m_to_px(*self._drag_start_m)
        x1, y1 = self.m_to_px(x_m, y_m)
        if self._drag_preview_id is not None:
            self.canvas.delete(self._drag_preview_id)
        color = MATERIAL_COLORS.get(self.material_var.get(), "#000000")
        self._drag_preview_id = self.canvas.create_line(x0, y0, x1, y1, fill=color,
                                                          width=max(2, self.thickness_var.get() * self.ppm * 2),
                                                          dash=(6, 3))

    def on_canvas_release(self, event):
        if self.mode.get() != "wall" or self._drag_start_m is None:
            return
        x_m, y_m = self._event_to_m(event)
        x_m, y_m = self._snap(x_m, y_m)
        if self._drag_preview_id is not None:
            self.canvas.delete(self._drag_preview_id)
            self._drag_preview_id = None
        x0, y0 = self._drag_start_m
        self._drag_start_m = None
        if abs(x_m - x0) < SNAP_TOLERANCE_M and abs(y_m - y0) < SNAP_TOLERANCE_M:
            return  # zero-length wall, ignore
        wall = Wall(x0, y0, x_m, y_m, self.material_var.get(), float(self.thickness_var.get()))
        self.floor_plan.walls.append(wall)
        self.redraw()

    def on_canvas_double_click(self, event):
        if self.mode.get() not in ("select", "ap"):
            return
        x_m, y_m = self._event_to_m(event)
        hit = self._hit_test_ap(x_m, y_m)
        if hit is not None:
            existing_ids = {a.id for a in self.floor_plan.access_points}
            dlg = ApPropertiesDialog(self.root, hit, existing_ids)
            self.root.wait_window(dlg)
            if dlg.result is not None:
                dlg.result.x, dlg.result.y = hit.x, hit.y
                idx = self.floor_plan.access_points.index(hit)
                self.floor_plan.access_points[idx] = dlg.result
                self.redraw()

    def on_delete_key(self, _event):
        if self.selected is None:
            return
        kind, ref = self.selected
        if kind == "wall":
            del self.floor_plan.walls[ref]
        elif kind == "ap":
            self.floor_plan.access_points = [a for a in self.floor_plan.access_points if a.id != ref]
        self.selected = None
        self.redraw()

    # ------------------------------------------------------------------
    # Hit testing / selection
    # ------------------------------------------------------------------

    def _hit_test_ap(self, x_m, y_m) -> AccessPoint | None:
        tol = AP_RADIUS_PX / self.ppm
        for ap in self.floor_plan.access_points:
            if (ap.x - x_m) ** 2 + (ap.y - y_m) ** 2 <= tol ** 2 * 4:
                return ap
        return None

    def _hit_test_wall(self, x_m, y_m) -> int | None:
        tol = 8.0 / self.ppm
        for i, w in enumerate(self.floor_plan.walls):
            if self._point_segment_dist(x_m, y_m, w.x1, w.y1, w.x2, w.y2) <= tol:
                return i
        return None

    @staticmethod
    def _point_segment_dist(px, py, x1, y1, x2, y2) -> float:
        dx, dy = x2 - x1, y2 - y1
        length_sq = dx * dx + dy * dy
        if length_sq < 1e-12:
            return ((px - x1) ** 2 + (py - y1) ** 2) ** 0.5
        t = max(0.0, min(1.0, ((px - x1) * dx + (py - y1) * dy) / length_sq))
        cx, cy = x1 + t * dx, y1 + t * dy
        return ((px - cx) ** 2 + (py - cy) ** 2) ** 0.5

    def _select_at(self, x_m, y_m):
        ap = self._hit_test_ap(x_m, y_m)
        if ap is not None:
            self.selected = ("ap", ap.id)
            self.status_var.set(f"Selected AP '{ap.id}' - press Delete to remove")
            return
        wi = self._hit_test_wall(x_m, y_m)
        if wi is not None:
            self.selected = ("wall", wi)
            self.status_var.set(f"Selected wall #{wi} - press Delete to remove")
            return
        self.selected = None

    # ------------------------------------------------------------------
    # File I/O
    # ------------------------------------------------------------------

    def new_plan_dialog(self):
        dlg = NewPlanDialog(self.root, self.floor_plan.width_m, self.floor_plan.height_m,
                             self.floor_plan.resolution_m)
        self.root.wait_window(dlg)
        if dlg.result is not None:
            w, h, res = dlg.result
            self.floor_plan = FloorPlan(width_m=w, height_m=h, resolution_m=res)
            self.current_path = None
            self.selected = None
            self.redraw()

    def load_json(self):
        path = filedialog.askopenfilename(filetypes=[("JSON floor plan", "*.json")])
        if not path:
            return
        try:
            self.floor_plan = FloorPlan.from_json(path)
            self.current_path = path
            self.selected = None
            self.redraw()
            self.status_var.set(f"Loaded {os.path.basename(path)}")
        except Exception as exc:
            messagebox.showerror("Load failed", str(exc))

    def save_json(self):
        if self.current_path is None:
            self.save_json_as()
            return
        self.floor_plan.to_json(self.current_path)
        self.status_var.set(f"Saved {os.path.basename(self.current_path)}")

    def save_json_as(self):
        path = filedialog.asksaveasfilename(defaultextension=".json",
                                             filetypes=[("JSON floor plan", "*.json")])
        if not path:
            return
        self.floor_plan.to_json(path)
        self.current_path = path
        self.status_var.set(f"Saved {os.path.basename(path)}")


def main():
    root = tk.Tk()
    CADEditor(root)
    root.mainloop()


if __name__ == "__main__":
    main()
