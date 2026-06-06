import pytest

from graspnt_rm.safety import (
    validate_grasp_pose,
    validate_motion_plan,
)


def test_validate_grasp_pose_rejects_low_z():
    with pytest.raises(ValueError, match="below min_grasp_z"):
        validate_grasp_pose([0.2, 0.0, 0.03, 3.14, 0.0, 0.0], min_grasp_z=0.05)


def test_validate_grasp_pose_accepts_safe_z():
    validate_grasp_pose([0.2, 0.0, 0.06, 3.14, 0.0, 0.0], min_grasp_z=0.05)


def test_validate_grasp_pose_rejects_pose_outside_workspace_bounds():
    with pytest.raises(ValueError, match="outside workspace bounds"):
        validate_grasp_pose(
            [0.9, 0.0, 0.2, 3.14, 0.0, 0.0],
            min_grasp_z=0.05,
            workspace_bounds={"x": [-0.5, 0.5], "y": [-0.5, 0.5], "z": [0.05, 0.8]},
        )


def test_validate_motion_plan_checks_all_motion_poses():
    plan = {
        "pre_grasp_pose": [0.2, 0.0, 0.2, 0.0, 0.0, 0.0],
        "grasp_pose": [0.2, 0.0, 0.2, 0.0, 0.0, 0.0],
        "lift_pose": [0.2, 0.0, 0.9, 0.0, 0.0, 0.0],
    }

    with pytest.raises(ValueError, match="lift_pose"):
        validate_motion_plan(
            plan,
            {
                "min_grasp_z": 0.05,
                "workspace_bounds": {
                    "x": [-0.5, 0.5],
                    "y": [-0.5, 0.5],
                    "z": [0.05, 0.8],
                },
            },
        )


def test_validate_grasp_pose_rejects_string_values():
    with pytest.raises(ValueError, match="real numeric values"):
        validate_grasp_pose([0.2, 0.0, "0.06", 3.14, 0.0, 0.0], min_grasp_z=0.05)

