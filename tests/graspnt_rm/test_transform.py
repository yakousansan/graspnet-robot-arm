import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from graspnt_rm.transform import (
    DEFAULT_GRASPNET_TO_GRIPPER,
    build_transform,
    camera_grasp_to_base_pose,
    graspnet_to_camera_transform,
    hand_eye_transform,
    offset_pose_along_base_z,
    offset_pose_along_local_z,
    pose_to_transform,
    transform_to_pose,
)


def test_pose_round_trip_with_identity_rotation():
    pose = [0.12, -0.03, 0.45, 0.0, 0.0, 0.0]

    transform = pose_to_transform(pose)
    round_trip = transform_to_pose(transform)

    np.testing.assert_allclose(transform[:3, :3], np.eye(3))
    assert isinstance(round_trip, list)
    np.testing.assert_allclose(round_trip, pose, atol=1e-12)


def test_default_graspnet_to_gripper_matches_plan_rotation():
    expected = np.array(
        [
            [0.0, 0.0, 1.0],
            [0.0, 1.0, 0.0],
            [-1.0, 0.0, 0.0],
        ],
        dtype=float,
    )

    np.testing.assert_allclose(DEFAULT_GRASPNET_TO_GRIPPER, expected)


def test_hand_eye_transform_uses_camera_to_end_without_inverse():
    rotation = Rotation.from_euler("xyz", [0.0, 0.0, np.pi / 2]).as_matrix()
    translation = [0.1, -0.2, 0.3]

    transform = hand_eye_transform(rotation, translation)

    camera_point = np.array([0.25, 0.0, 0.0, 1.0])
    expected_end_point = np.r_[rotation @ camera_point[:3] + translation, 1.0]
    inverse_end_point = np.linalg.inv(build_transform(rotation, translation)) @ camera_point

    np.testing.assert_allclose(transform @ camera_point, expected_end_point, atol=1e-12)
    assert not np.allclose(transform @ camera_point, inverse_end_point)


def test_hand_eye_transform_rejects_invalid_rotation_length_with_build_transform_error():
    with pytest.raises(ValueError, match="rotation must be"):
        hand_eye_transform([1.0, 0.0, 0.0, 1.0], [0.1, -0.2, 0.3])


def test_graspnet_to_camera_transform_applies_gripper_length_offset():
    translation = [0.3, -0.2, 0.5]
    rotation = Rotation.from_euler("xyz", [0.2, -0.1, 0.4]).as_matrix()
    gripper_length = 0.08

    camera_from_gripper = graspnet_to_camera_transform(
        translation,
        rotation,
        gripper_length=gripper_length,
    )
    expected_offset = DEFAULT_GRASPNET_TO_GRIPPER @ np.array(
        [0.0, 0.0, -gripper_length],
        dtype=float,
    )
    expected = build_transform(rotation, translation) @ build_transform(
        DEFAULT_GRASPNET_TO_GRIPPER,
        expected_offset,
    )

    np.testing.assert_allclose(camera_from_gripper, expected, atol=1e-12)


def test_gripper_length_offset_follows_gripper_axis_not_grasp_axis():
    translation = [0.0, 0.0, 0.0]
    rotation = np.eye(3)
    gripper_length = 0.08

    camera_from_gripper = graspnet_to_camera_transform(
        translation,
        rotation,
        gripper_length=gripper_length,
    )

    np.testing.assert_allclose(
        camera_from_gripper[:3, 3],
        DEFAULT_GRASPNET_TO_GRIPPER @ np.array([0.0, 0.0, -gripper_length]),
        atol=1e-12,
    )
    assert not np.allclose(camera_from_gripper[:3, 3], [0.0, 0.0, -gripper_length])


def test_camera_grasp_to_base_pose_matches_raw_plan_api_chain():
    grasp_translation = [0.03, -0.04, 0.5]
    grasp_rotation = Rotation.from_euler("xyz", [-0.4, 0.1, 0.2]).as_matrix()
    current_end_pose = [0.4, -0.1, 0.2, 0.1, -0.2, 0.3]
    hand_eye_rotation = Rotation.from_euler("xyz", [0.2, 0.3, -0.1]).as_matrix()
    hand_eye_translation = [0.05, 0.02, 0.1]
    gripper_length = 0.08

    base_pose = camera_grasp_to_base_pose(
        grasp_translation,
        grasp_rotation,
        current_end_pose,
        hand_eye_rotation,
        hand_eye_translation,
        gripper_length=gripper_length,
    )
    base_transform = pose_to_transform(base_pose)
    expected_transform = (
        pose_to_transform(current_end_pose)
        @ hand_eye_transform(hand_eye_rotation, hand_eye_translation)
        @ graspnet_to_camera_transform(
            grasp_translation,
            grasp_rotation,
            gripper_length=gripper_length,
        )
    )
    origin = np.array([0.0, 0.0, 0.0, 1.0])

    assert isinstance(base_pose, list)
    np.testing.assert_allclose(base_transform, expected_transform, atol=1e-12)
    np.testing.assert_allclose(
        base_transform @ origin,
        expected_transform @ origin,
        atol=1e-12,
    )


def test_offset_pose_along_local_z_returns_list_and_uses_pose_orientation():
    pose = [0.1, -0.2, 0.3, 0.0, 0.5, 0.0]

    offset_pose = offset_pose_along_local_z(pose, 0.07)
    expected_z_axis = pose_to_transform(pose)[:3, 2]

    assert isinstance(offset_pose, list)
    np.testing.assert_allclose(
        offset_pose[:3],
        np.asarray(pose[:3]) + expected_z_axis * 0.07,
        atol=1e-12,
    )
    np.testing.assert_allclose(offset_pose[3:], pose[3:], atol=1e-12)


def test_offset_pose_along_base_z_only_changes_world_z():
    pose = [0.1, -0.2, 0.3, 0.0, 0.5, 0.0]

    offset_pose = offset_pose_along_base_z(pose, 0.07)

    assert isinstance(offset_pose, list)
    np.testing.assert_allclose(offset_pose[:3], [0.1, -0.2, 0.37], atol=1e-12)
    np.testing.assert_allclose(offset_pose[3:], pose[3:], atol=1e-12)
