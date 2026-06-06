from __future__ import annotations

import importlib
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from graspnt_rm.camera_realsense import CameraIntrinsics


GRASPNET_API_SOURCE_ROOT = Path(__file__).resolve().parents[1] / "graspnetAPI"


@dataclass(frozen=True)
class GraspCandidate:
    translation: np.ndarray
    rotation_matrix: np.ndarray
    score: float
    width: float


@dataclass(frozen=True)
class InferenceDebugData:
    workspace_mask: np.ndarray
    cloud_points: np.ndarray
    cloud_colors: np.ndarray


def filter_grasp_candidates(
    candidates: list[GraspCandidate],
    config: dict[str, Any],
) -> list[GraspCandidate]:
    min_score = float(config.get("min_score", 0.0))
    score_filtered = [candidate for candidate in candidates if candidate.score >= min_score]
    if not score_filtered:
        return []

    if "top_down_angle_deg" not in config:
        return score_filtered

    axis_index = int(config.get("approach_axis", 0))
    vertical = np.array([0.0, 0.0, 1.0])
    threshold = np.deg2rad(float(config["top_down_angle_deg"]))
    top_down = []
    for candidate in score_filtered:
        approach = np.asarray(candidate.rotation_matrix, dtype=float)[:, axis_index]
        norm = np.linalg.norm(approach)
        if norm == 0:
            continue
        cos_angle = np.clip(np.dot(approach / norm, vertical), -1.0, 1.0)
        if np.arccos(cos_angle) <= threshold:
            top_down.append(candidate)
    return top_down if top_down else score_filtered


def build_workspace_mask(depth: np.ndarray, workspace_config: dict[str, Any]) -> np.ndarray:
    mask = np.asarray(depth) > 0
    if workspace_config.get("mode", "center") == "center":
        height, width = mask.shape
        x0 = int(width * float(workspace_config["x_min_ratio"]))
        x1 = int(width * float(workspace_config["x_max_ratio"]))
        y0 = int(height * float(workspace_config["y_min_ratio"]))
        y1 = int(height * float(workspace_config["y_max_ratio"]))

        center_mask = np.zeros_like(mask, dtype=bool)
        center_mask[y0:y1, x0:x1] = True
        mask &= center_mask
    return mask


def _import_grasp_group() -> type:
    try:
        graspnet_api = importlib.import_module("graspnetAPI")
    except ImportError as exc:
        raise RuntimeError(
            "missing GraspNet runtime dependency: graspnetAPI. "
            "Install graspnetAPI before creating GraspNetRunner."
        ) from exc

    grasp_group = getattr(graspnet_api, "GraspGroup", None)
    if grasp_group is not None:
        return grasp_group

    source_root = GRASPNET_API_SOURCE_ROOT
    if source_root.exists():
        source_root_text = str(source_root)
        if source_root_text in sys.path:
            sys.path.remove(source_root_text)
        sys.path.insert(0, source_root_text)
        importlib.invalidate_caches()

        module = sys.modules.get("graspnetAPI")
        if module is not None and getattr(module, "__spec__", None) is not None:
            try:
                graspnet_api = importlib.reload(module)
            except ImportError:
                sys.modules.pop("graspnetAPI", None)
                try:
                    graspnet_api = importlib.import_module("graspnetAPI")
                except ImportError as exc:
                    raise RuntimeError(
                        "missing GraspNet runtime dependency: "
                        "graspnetAPI.GraspGroup. Ensure graspnetAPI is installed "
                        f"or that the source checkout path {source_root} is importable."
                    ) from exc
        else:
            sys.modules.pop("graspnetAPI", None)
            try:
                graspnet_api = importlib.import_module("graspnetAPI")
            except ImportError as exc:
                raise RuntimeError(
                    "missing GraspNet runtime dependency: "
                    "graspnetAPI.GraspGroup. Ensure graspnetAPI is installed "
                    f"or that the source checkout path {source_root} is importable."
                ) from exc

        grasp_group = getattr(graspnet_api, "GraspGroup", None)
        if grasp_group is not None:
            return grasp_group

    raise RuntimeError(
        "missing GraspNet runtime dependency: graspnetAPI.GraspGroup. "
        "Ensure graspnetAPI is installed or that the source checkout path "
        f"{source_root} is importable."
    )


class GraspNetRunner:
    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.last_debug: InferenceDebugData | None = None
        root = Path(config["root"])
        for relative_path in ("models", "dataset", "utils"):
            runtime_path = str(root / relative_path)
            if runtime_path not in sys.path:
                sys.path.append(runtime_path)

        try:
            torch = importlib.import_module("torch")
            o3d = importlib.import_module("open3d")
            GraspGroup = _import_grasp_group()
            collision_detector = importlib.import_module("collision_detector")
            data_utils = importlib.import_module("data_utils")
            graspnet = importlib.import_module("graspnet")
        except ImportError as exc:
            missing = exc.name or str(exc)
            raise RuntimeError(
                "missing GraspNet runtime dependency: "
                f"{missing}. Install torch, open3d, graspnetAPI, and the "
                "graspnet-baseline runtime modules before creating GraspNetRunner."
            ) from exc

        self._torch = torch
        self._o3d = o3d
        self.GraspGroup = GraspGroup
        self.CameraInfo = data_utils.CameraInfo
        self.ModelFreeCollisionDetector = collision_detector.ModelFreeCollisionDetector
        self.create_point_cloud_from_depth_image = (
            data_utils.create_point_cloud_from_depth_image
        )
        self.pred_decode = graspnet.pred_decode

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.net = graspnet.GraspNet(
            input_feature_dim=0,
            num_view=int(config["num_view"]),
            num_angle=12,
            num_depth=4,
            cylinder_radius=0.05,
            hmin=-0.02,
            hmax_list=[0.01, 0.02, 0.03, 0.04],
            is_training=False,
        ).to(self.device)

        checkpoint = torch.load(config["checkpoint"], map_location=self.device)
        self.net.load_state_dict(checkpoint["model_state_dict"])
        self.net.eval()

    def build_workspace_mask(
        self,
        depth: np.ndarray,
        workspace_config: dict[str, Any],
    ) -> np.ndarray:
        return build_workspace_mask(depth, workspace_config)

    def infer(
        self,
        color: np.ndarray,
        depth: np.ndarray,
        intrinsics: CameraIntrinsics,
        workspace_config: dict[str, Any],
    ) -> tuple[list[GraspCandidate], dict[str, int]]:
        camera = self.CameraInfo(
            intrinsics.width,
            intrinsics.height,
            intrinsics.fx,
            intrinsics.fy,
            intrinsics.cx,
            intrinsics.cy,
            intrinsics.scale,
        )
        cloud = self.create_point_cloud_from_depth_image(depth, camera, organized=True)
        color_float = np.asarray(color, dtype=np.float32) / 255.0
        workspace_mask = self.build_workspace_mask(depth, workspace_config)
        cloud_masked = cloud[workspace_mask]
        color_masked = color_float[workspace_mask]
        if len(cloud_masked) == 0:
            self.last_debug = None
            raise RuntimeError("workspace mask produced zero valid depth points")
        self.last_debug = InferenceDebugData(
            workspace_mask=np.asarray(workspace_mask, dtype=bool),
            cloud_points=np.asarray(cloud_masked, dtype=np.float32),
            cloud_colors=np.asarray(color_masked, dtype=np.float32),
        )

        torch = self._torch
        o3d = self._o3d

        num_point = int(self.config["num_point"])
        if len(cloud_masked) >= num_point:
            idxs = np.random.choice(len(cloud_masked), num_point, replace=False)
        else:
            idxs_keep = np.arange(len(cloud_masked))
            idxs_extra = np.random.choice(
                len(cloud_masked),
                num_point - len(cloud_masked),
                replace=True,
            )
            idxs = np.concatenate([idxs_keep, idxs_extra], axis=0)

        cloud_sampled = torch.from_numpy(
            cloud_masked[idxs][np.newaxis].astype(np.float32)
        ).to(self.device)
        end_points = {
            "point_clouds": cloud_sampled,
            "cloud_colors": color_masked[idxs],
        }

        with torch.no_grad():
            end_points = self.net(end_points)
            grasp_preds = self.pred_decode(end_points)

        gg = self.GraspGroup(grasp_preds[0].detach().cpu().numpy())

        cloud_o3d = o3d.geometry.PointCloud()
        cloud_o3d.points = o3d.utility.Vector3dVector(cloud_masked.astype(np.float32))
        cloud_o3d.colors = o3d.utility.Vector3dVector(color_masked.astype(np.float32))

        collision_thresh = float(self.config.get("collision_thresh", 0.0))
        if collision_thresh > 0:
            detector = self.ModelFreeCollisionDetector(
                np.asarray(cloud_o3d.points),
                voxel_size=float(self.config.get("voxel_size", 0.01)),
            )
            collision_mask = detector.detect(
                gg,
                approach_dist=0.05,
                collision_thresh=collision_thresh,
            )
            gg = gg[~collision_mask]

        gg = gg.nms()
        gg = gg.sort_by_score()

        candidates = [
            GraspCandidate(
                translation=np.asarray(grasp.translation, dtype=float),
                rotation_matrix=np.asarray(grasp.rotation_matrix, dtype=float),
                score=float(grasp.score),
                width=float(grasp.width),
            )
            for grasp in gg
        ]
        candidates = filter_grasp_candidates(candidates, self.config)
        report = {
            "valid_workspace_points": int(len(cloud_masked)),
            "candidate_count": int(len(candidates)),
        }
        return candidates, report
