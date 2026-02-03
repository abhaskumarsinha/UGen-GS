from __future__ import annotations

from typing import List, Tuple, Dict, Any

from dataclasses import dataclass
from typing import Tuple
import struct



@dataclass
class PlyImporterConfig:
    ply_path: str

    sh_max_degree: int = 3
    default_scale: float = 1.0
    default_alpha: float = 1.0
    default_quat: Tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)

from UGen.io.base import ImporterBase

class PlyImporter(ImporterBase):
    """
    PLY -> Gaussian importer

    Cameras are not stored in PLY, so load() returns:
        gaussians, []
    """

    def __init__(self, cfg: PlyImporterConfig):
        super().__init__(cfg.ply_path)
        self.cfg = cfg

    def load(self) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        ply_path = self.cfg.ply_path
        sh_max_degree = self.cfg.sh_max_degree

        if sh_max_degree < 0 or sh_max_degree > 6:
            raise ValueError("sh_max_degree must be between 0 and 6")

        K_supported = (sh_max_degree + 1) ** 2
        MAX_COEFFS = 3 * K_supported
        Y00 = 0.282095

        with open(ply_path, "rb") as f:
            header_lines: List[str] = []

            # --- read header ---
            while True:
                raw = f.readline()
                if not raw:
                    raise ValueError("Unexpected EOF while reading PLY header")
                line = raw.decode("ascii").rstrip("\r\n")
                header_lines.append(line)
                if line.strip() == "end_header":
                    break

            fmt_line = next((l for l in header_lines if l.startswith("format ")), None)
            if fmt_line is None or "binary_little_endian" not in fmt_line:
                raise ValueError("PLY must be binary_little_endian")

            vertex_count = None
            in_vertex = False
            prop_names: List[str] = []

            for ln in header_lines:
                parts = ln.split()
                if len(parts) >= 3 and parts[0] == "element" and parts[1] == "vertex":
                    vertex_count = int(parts[2])
                    in_vertex = True
                    continue
                if ln.startswith("element") and "vertex" not in ln:
                    in_vertex = False
                if in_vertex and parts and parts[0] == "property" and len(parts) >= 3:
                    prop_names.append(parts[-1])

            if vertex_count is None:
                raise ValueError("PLY header missing vertex element")

            n_props = len(prop_names)
            fmt = "<" + "f" * n_props
            size = struct.calcsize(fmt)

            prop_index = {name: i for i, name in enumerate(prop_names)}

            dc_keys = [f"f_dc_{i}" for i in range(3)]
            rest_keys = []
            j = 0
            while f"f_rest_{j}" in prop_index:
                rest_keys.append(f"f_rest_{j}")
                j += 1

            total_coeffs = 0
            if all(k in prop_index for k in dc_keys):
                total_coeffs += 3
            total_coeffs += len(rest_keys)

            total_coeffs = min(total_coeffs, MAX_COEFFS)
            K = total_coeffs // 3
            if K == 0:
                raise ValueError("No SH coefficients found in PLY")

            gaussians: List[Dict[str, Any]] = []

            for _ in range(vertex_count):
                data = f.read(size)
                if len(data) < size:
                    raise EOFError("Unexpected EOF while reading vertices")
                vals = struct.unpack(fmt, data)

                def p(name: str, default=0.0):
                    idx = prop_index.get(name)
                    return float(vals[idx]) if idx is not None else float(default)

                x, y, z = p("x"), p("y"), p("z")

                normal = None
                if all(k in prop_index for k in ("nx", "ny", "nz")):
                    normal = [p("nx"), p("ny"), p("nz")]

                coeffs: List[float] = []
                for k in range(3):
                    if len(coeffs) < MAX_COEFFS:
                        coeffs.append(p(f"f_dc_{k}", 0.0))
                for key in rest_keys:
                    if len(coeffs) >= MAX_COEFFS:
                        break
                    coeffs.append(p(key))

                if len(coeffs) < 3 * K:
                    coeffs.extend([0.0] * (3 * K - len(coeffs)))
                coeffs = coeffs[:3 * K]

                # DC heuristic: raw RGB vs SH-DC
                dc_guess = coeffs[:3]
                if all(0.0 <= v <= 1.0 for v in dc_guess):
                    coeffs[0] /= Y00
                    coeffs[1] /= Y00
                    coeffs[2] /= Y00

                if all(k in prop_index for k in ("scale_0", "scale_1", "scale_2")):
                    scale = [p("scale_0"), p("scale_1"), p("scale_2")]
                else:
                    scale = [self.cfg.default_scale] * 3

                alpha = p("opacity", self.cfg.default_alpha)

                if all(k in prop_index for k in ("rot_0", "rot_1", "rot_2", "rot_3")):
                    quat = [p("rot_0"), p("rot_1"), p("rot_2"), p("rot_3")]
                else:
                    quat = list(self.cfg.default_quat)

                gaussians.append({
                    "position": [x, y, z],
                    "normal": normal,
                    "sh": coeffs,
                    "scale": scale,
                    "alpha": float(alpha),
                    "quat": quat,
                })

        cameras: List[Dict[str, Any]] = []
        return gaussians, cameras
