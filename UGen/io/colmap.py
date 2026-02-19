from __future__ import annotations

import os
import warnings
from typing import List, Dict, Any, Tuple

import numpy as np
from PIL import Image

from UGen.io.base import ImporterBase

from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass
class ColmapImporterConfig:
    scene_dir: str

    # camera / image options
    resize: Optional[Tuple[int, int] | float | int] = None
    to_rgb_range: Tuple[float, float] = (0.0, 1.0)

    # gaussian options
    sh_degree: int = 0
    default_scale: float = 0.001
    default_alpha: float = 1.0
    default_quat: Tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)


class ColmapImporter:
    """
    COLMAP -> JSON-like importer

    load() returns:
        gaussians: List[Dict[str, Any]]
        cameras:   List[Dict[str, Any]]
    """

    def __init__(self, cfg: ColmapImporterConfig):
        self.cfg = cfg
        self.scene_dir = cfg.scene_dir
        self.sparse0 = os.path.join(self.scene_dir, "sparse", "0")
        self.images_dir = os.path.join(self.scene_dir, "images")

    @staticmethod
    def _read_txt_rows(path: str):
        with open(path, "r") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                yield line

    @staticmethod
    def _parse_cameras_txt(path: str) -> Dict[int, Dict[str, Any]]:
        cams = {}
        for line in ColmapImporter._read_txt_rows(path):
            parts = line.split()
            cam_id = int(parts[0])
            cams[cam_id] = {
                "model": parts[1],
                "width": int(parts[2]),
                "height": int(parts[3]),
                "params": [float(x) for x in parts[4:]],
            }
        return cams

    @staticmethod
    def _parse_images_txt(path: str) -> Dict[int, Dict[str, Any]]:
        images = {}
        with open(path, "r") as f:
            lines = [
                l.strip()
                for l in f.readlines()
                if l.strip() and not l.startswith("#")
            ]

        i = 0
        while i < len(lines):
            parts = lines[i].split()
            image_id = int(parts[0])
            images[image_id] = {
                "q": list(map(float, parts[1:5])),
                "t": list(map(float, parts[5:8])),
                "camera_id": int(parts[8]),
                "name": parts[9],
            }
            i += 2
        return images


    @staticmethod
    def _intrinsics_to_pinhole(model: str, params: List[float]):
        model = model.upper()

        if model == "PINHOLE":
            fx, fy, cx, cy = params[:4]
        elif model in ("SIMPLE_PINHOLE", "SIMPLE_RADIAL", "RADIAL"):
            f, cx, cy = params[:3]
            fx = fy = f
        else:
            if len(params) >= 4:
                fx, fy, cx, cy = params[:4]
                warnings.warn(
                    f"Unknown camera model '{model}', assuming pinhole"
                )
            else:
                raise ValueError(f"Unsupported camera model: {model}")

        return float(fx), float(fy), float(cx), float(cy)

    @staticmethod
    def _resize_image(img: Image.Image, resize):
        if resize is None:
            return img
        if isinstance(resize, tuple):
            return img.resize(resize, Image.LANCZOS)
        if isinstance(resize, float):
            w, h = img.size
            return img.resize((int(w * resize), int(h * resize)), Image.LANCZOS)
        if isinstance(resize, int):
            w, h = img.size
            if max(w, h) <= resize:
                return img
            scale = resize / float(max(w, h))
            return img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
        raise ValueError("Invalid resize argument")


    def _import_cameras(self) -> List[Dict[str, Any]]:
        cams_txt = os.path.join(self.sparse0, "cameras.txt")
        imgs_txt = os.path.join(self.sparse0, "images.txt")

        cams_raw = self._parse_cameras_txt(cams_txt)
        imgs_raw = self._parse_images_txt(imgs_txt)

        cameras_out = []

        for image_id in sorted(imgs_raw.keys()):
            info = imgs_raw[image_id]
            cam_meta = cams_raw[info["camera_id"]]

            fx, fy, cx, cy = self._intrinsics_to_pinhole(
                cam_meta["model"], cam_meta["params"]
            )

            img_path = os.path.join(self.images_dir, os.path.basename(info["name"]))
            if not os.path.exists(img_path):
                raise FileNotFoundError(img_path)

            img = Image.open(img_path).convert("RGB")
            img = self._resize_image(img, self.cfg.resize)

            arr = np.asarray(img).astype(np.float32) / 255.0
            rmin, rmax = self.cfg.to_rgb_range
            if (rmin, rmax) != (0.0, 1.0):
                arr = arr * (rmax - rmin) + rmin

            cameras_out.append({
                "width": cam_meta["width"],
                "height": cam_meta["height"],
                "fx": fx,
                "fy": fy,
                "cx": cx,
                "cy": cy,
                "rotation": info["q"],   # qw,qx,qy,qz
                "translation": info["t"],
                "image": arr,            # H x W x 3 float32
            })

        return cameras_out

    def _import_gaussians(self) -> List[Dict[str, Any]]:
        points_txt = os.path.join(self.sparse0, "points3D.txt")

        K = (self.cfg.sh_degree + 1) ** 2
        Y00 = 0.282095

        gaussians = []

        for line in self._read_txt_rows(points_txt):
            parts = line.split()
            x, y, z = map(float, parts[1:4])
            r, g, b = map(int, parts[4:7])

            rgb = np.array([r, g, b], dtype=np.float32) / 255.0
            sh = np.zeros((K, 3), dtype=np.float32)
            sh[0] = rgb / Y00

            gaussians.append({
                "position": [x, y, z],
                "sh": sh.reshape(-1).tolist(),
                "scale": [self.cfg.default_scale] * 3,
                "alpha": self.cfg.default_alpha,
                "quat": list(self.cfg.default_quat),
            })

        return gaussians

    def load(self) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Returns:
            gaussians, cameras
        """
        cameras = self._import_cameras()
        gaussians = self._import_gaussians()
        return gaussians, cameras



