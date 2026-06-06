import sys
import types

import numpy as np
import pytest


def test_module_imports_without_graspnet_runtime_dependencies():
    from graspnt_rm.graspnet_infer import GraspCandidate

    candidate = GraspCandidate(
        translation=np.array([0.1, 0.2, 0.3]),
        rotation_matrix=np.eye(3),
        score=0.75,
        width=0.04,
    )

    assert candidate.score == 0.75
    with pytest.raises(Exception):
        candidate.score = 0.2


def test_build_workspace_mask_center_mode_applies_ratio_bounds():
    from graspnt_rm.graspnet_infer import build_workspace_mask

    depth = np.array(
        [
            [1, 1, 1, 1],
            [1, 0, 2, 1],
            [1, 3, 4, 1],
            [1, 1, 1, 1],
        ],
        dtype=np.uint16,
    )

    mask = build_workspace_mask(
        depth,
        {
            "mode": "center",
            "x_min_ratio": 0.25,
            "x_max_ratio": 0.75,
            "y_min_ratio": 0.25,
            "y_max_ratio": 0.75,
        },
    )

    expected = np.array(
        [
            [False, False, False, False],
            [False, False, True, False],
            [False, True, True, False],
            [False, False, False, False],
        ]
    )
    np.testing.assert_array_equal(mask, expected)


def test_build_workspace_mask_does_not_raise_for_zero_valid_points():
    from graspnt_rm.graspnet_infer import build_workspace_mask

    depth = np.zeros((2, 3), dtype=np.uint16)

    mask = build_workspace_mask(depth, {"mode": "all"})

    assert mask.shape == depth.shape
    assert mask.dtype == np.bool_
    assert not mask.any()


def test_filter_grasp_candidates_rejects_low_score_and_prefers_top_down():
    from graspnt_rm.graspnet_infer import GraspCandidate, filter_grasp_candidates

    side_grasp = GraspCandidate(
        translation=np.array([0.0, 0.0, 0.2]),
        rotation_matrix=np.array(
            [
                [0.0, 0.0, 1.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
            ],
            dtype=float,
        ),
        score=0.9,
        width=0.04,
    )
    top_down = GraspCandidate(
        translation=np.array([0.0, 0.0, 0.2]),
        rotation_matrix=np.array(
            [
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
                [1.0, 0.0, 0.0],
            ],
            dtype=float,
        ),
        score=0.7,
        width=0.04,
    )
    low_score = GraspCandidate(
        translation=np.array([0.0, 0.0, 0.2]),
        rotation_matrix=np.eye(3),
        score=0.01,
        width=0.04,
    )

    filtered = filter_grasp_candidates(
        [side_grasp, top_down, low_score],
        {"min_score": 0.05, "top_down_angle_deg": 30, "approach_axis": 0},
    )

    assert filtered == [top_down]


def test_filter_grasp_candidates_returns_empty_when_all_scores_too_low():
    from graspnt_rm.graspnet_infer import GraspCandidate, filter_grasp_candidates

    filtered = filter_grasp_candidates(
        [
            GraspCandidate(
                translation=np.array([0.0, 0.0, 0.2]),
                rotation_matrix=np.eye(3),
                score=0.01,
                width=0.04,
            )
        ],
        {"min_score": 0.05},
    )

    assert filtered == []


def test_infer_raises_when_workspace_has_zero_valid_points():
    from graspnt_rm.camera_realsense import CameraIntrinsics
    from graspnt_rm.graspnet_infer import GraspNetRunner

    runner = object.__new__(GraspNetRunner)
    runner.config = {"num_point": 4, "collision_thresh": 0.0, "voxel_size": 0.01}
    runner.CameraInfo = lambda *args: args
    runner.create_point_cloud_from_depth_image = lambda depth, camera, organized: np.zeros(
        (*depth.shape, 3),
        dtype=np.float32,
    )

    with pytest.raises(RuntimeError, match="zero valid depth points"):
        runner.infer(
            color=np.zeros((2, 2, 3), dtype=np.uint8),
            depth=np.zeros((2, 2), dtype=np.uint16),
            intrinsics=CameraIntrinsics(
                width=2,
                height=2,
                fx=1.0,
                fy=1.0,
                cx=1.0,
                cy=1.0,
                scale=1000.0,
            ),
            workspace_config={"mode": "all"},
        )


def test_init_missing_dependency_raises_clear_runtime_error(monkeypatch):
    from graspnt_rm.graspnet_infer import GraspNetRunner

    monkeypatch.setitem(sys.modules, "torch", None)

    with pytest.raises(RuntimeError, match="missing GraspNet runtime dependency.*torch"):
        GraspNetRunner(
            {
                "root": "/tmp/does-not-matter",
                "checkpoint": "/tmp/checkpoint.tar",
                "num_view": 300,
                "num_point": 4,
            }
        )


def test_import_grasp_group_falls_back_to_nested_source_checkout(
    monkeypatch,
    tmp_path,
):
    from graspnt_rm import graspnet_infer

    outer_module = types.ModuleType("graspnetAPI")
    monkeypatch.setitem(sys.modules, "graspnetAPI", outer_module)
    source_root = tmp_path / "graspnetAPI"
    package_dir = source_root / "graspnetAPI"
    package_dir.mkdir(parents=True)
    (package_dir / "__init__.py").write_text(
        "class GraspGroup:\n"
        "    pass\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(graspnet_infer, "GRASPNET_API_SOURCE_ROOT", source_root)

    grasp_group = graspnet_infer._import_grasp_group()

    assert grasp_group.__name__ == "GraspGroup"
    assert str(source_root) in sys.path


def test_import_grasp_group_missing_after_fallback_raises_clear_runtime_error(
    monkeypatch,
    tmp_path,
):
    from graspnt_rm import graspnet_infer

    outer_module = types.ModuleType("graspnetAPI")
    monkeypatch.setitem(sys.modules, "graspnetAPI", outer_module)
    source_root = tmp_path / "graspnetAPI"
    package_dir = source_root / "graspnetAPI"
    package_dir.mkdir(parents=True)
    (package_dir / "__init__.py").write_text("VALUE = 1\n", encoding="utf-8")
    monkeypatch.setattr(graspnet_infer, "GRASPNET_API_SOURCE_ROOT", source_root)

    with pytest.raises(RuntimeError, match="graspnetAPI.*GraspGroup"):
        graspnet_infer._import_grasp_group()
