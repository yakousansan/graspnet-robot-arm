from pathlib import Path
from types import SimpleNamespace

import numpy as np


class FakeCv2:
    COLORMAP_JET = 0

    def imwrite(self, path, image):
        with open(path, "wb") as file:
            file.write(b"png")
        return True

    def applyColorMap(self, image, color_map):
        return np.repeat(image[:, :, None], 3, axis=2)


class FakeOpen3D:
    class geometry:
        class PointCloud:
            pass

        class LineSet:
            pass

    class utility:
        @staticmethod
        def Vector3dVector(value):
            return np.asarray(value, dtype=float)

        @staticmethod
        def Vector2iVector(value):
            return np.asarray(value, dtype=np.int32)

    class visualization:
        last_visualizer = None

        class Visualizer:
            def __init__(self):
                self.geometries = []
                self.window = None
                FakeOpen3D.visualization.last_visualizer = self

            def create_window(self, **kwargs):
                self.window = kwargs
                return True

            def add_geometry(self, geometry):
                self.geometries.append(geometry)

            def poll_events(self):
                pass

            def update_renderer(self):
                pass

            def capture_screen_image(self, path, do_render=True):
                with open(path, "wb") as file:
                    file.write(b"3d-grasps")

            def destroy_window(self):
                pass


def test_project_point_to_pixel_uses_camera_intrinsics():
    from graspnt_rm.visualization import project_point_to_pixel

    intrinsics = SimpleNamespace(fx=500.0, fy=500.0, cx=320.0, cy=240.0)

    assert project_point_to_pixel([0.10, 0.05, 0.50], intrinsics) == (420, 290)
    assert project_point_to_pixel([0.10, 0.05, 0.00], intrinsics) is None


def test_build_workspace_overlay_tints_pixels_outside_mask():
    from graspnt_rm.visualization import build_workspace_overlay

    color = np.zeros((2, 2, 3), dtype=np.uint8)
    mask = np.array([[True, False], [False, True]])

    overlay = build_workspace_overlay(color, mask, outside_tint=(255, 0, 0), alpha=0.5)

    np.testing.assert_array_equal(overlay[0, 0], [0, 0, 0])
    np.testing.assert_array_equal(overlay[1, 1], [0, 0, 0])
    np.testing.assert_array_equal(overlay[0, 1], [127, 0, 0])
    np.testing.assert_array_equal(overlay[1, 0], [127, 0, 0])


def test_candidate_to_visualization_dict_converts_numpy_arrays():
    from graspnt_rm.visualization import candidate_to_visualization_dict

    candidate = SimpleNamespace(
        translation=np.array([0.1, 0.2, 0.3]),
        rotation_matrix=np.eye(3),
        score=0.82,
        width=0.04,
    )

    result = candidate_to_visualization_dict(candidate)

    assert result == {
        "translation": [0.1, 0.2, 0.3],
        "rotation_matrix": np.eye(3).tolist(),
        "score": 0.82,
        "width": 0.04,
    }


def test_preview_workspace_returns_confirmed_frame(monkeypatch):
    from graspnt_rm import visualization

    frame = SimpleNamespace(
        color=np.zeros((2, 2, 3), dtype=np.uint8),
        depth=np.ones((2, 2), dtype=np.uint16),
        intrinsics=SimpleNamespace(width=2, height=2, fx=1, fy=1, cx=0, cy=0),
    )

    class FakeCamera:
        def __init__(self):
            self.captures = 0

        def capture(self, warmup_frames=0):
            self.captures += 1
            return frame

    class FakeCv2:
        COLORMAP_JET = 0

        def __init__(self):
            self.images = []

        def applyColorMap(self, image, color_map):
            return np.repeat(image[:, :, None], 3, axis=2)

        def imshow(self, window_name, image):
            self.images.append((window_name, image.copy()))

        def waitKey(self, delay):
            return ord(" ")

        def destroyAllWindows(self):
            pass

    fake_cv2 = FakeCv2()
    monkeypatch.setattr(visualization, "_import_cv2", lambda: fake_cv2)

    camera = FakeCamera()
    result = visualization.preview_workspace(
        camera,
        {"mode": "center", "x_min_ratio": 0.5, "x_max_ratio": 1.0, "y_min_ratio": 0.0, "y_max_ratio": 1.0},
        {"enabled": True, "show_depth": True},
    )

    assert result is frame
    assert camera.captures == 1
    assert fake_cv2.images[0][0] == "GRASPNT Workspace Preview"


def test_preview_workspace_returns_none_when_cancelled(monkeypatch):
    from graspnt_rm import visualization

    frame = SimpleNamespace(
        color=np.zeros((2, 2, 3), dtype=np.uint8),
        depth=np.ones((2, 2), dtype=np.uint16),
        intrinsics=SimpleNamespace(width=2, height=2, fx=1, fy=1, cx=0, cy=0),
    )

    class FakeCamera:
        def capture(self, warmup_frames=0):
            return frame

    class FakeCv2:
        COLORMAP_JET = 0

        def applyColorMap(self, image, color_map):
            return np.repeat(image[:, :, None], 3, axis=2)

        def imshow(self, window_name, image):
            pass

        def waitKey(self, delay):
            return ord("q")

        def destroyAllWindows(self):
            pass

    monkeypatch.setattr(visualization, "_import_cv2", lambda: FakeCv2())

    result = visualization.preview_workspace(
        FakeCamera(),
        {"mode": "all"},
        {"enabled": True},
    )

    assert result is None


def test_build_3d_grasp_geometries_marks_best_candidate_green():
    from graspnt_rm.visualization import _build_3d_grasp_geometries

    candidates = [
        SimpleNamespace(
            translation=np.array([0.1, 0.0, 0.2]),
            rotation_matrix=np.eye(3),
            score=0.9,
            width=0.04,
        ),
        SimpleNamespace(
            translation=np.array([0.2, 0.0, 0.2]),
            rotation_matrix=np.eye(3),
            score=0.5,
            width=0.05,
        ),
    ]
    debug_data = SimpleNamespace(
        cloud_points=np.array([[0.1, 0.0, 0.2]]),
        cloud_colors=np.array([[255, 0, 0]], dtype=np.uint8),
    )

    geometries = _build_3d_grasp_geometries(
        FakeOpen3D,
        candidates,
        debug_data,
        {"top_n": 20, "color_order": "bgr"},
    )

    assert len(geometries) == 3
    np.testing.assert_allclose(geometries[1].colors[0], [0.0, 1.0, 0.0])
    np.testing.assert_allclose(geometries[2].colors[0], [0.65, 0.65, 0.65])


def test_save_debug_artifacts_writes_3d_grasp_candidate_image(monkeypatch, tmp_path):
    from graspnt_rm import visualization

    monkeypatch.setattr(visualization, "_import_cv2", lambda: FakeCv2())
    monkeypatch.setattr(visualization, "_import_open3d", lambda: FakeOpen3D)

    frame = SimpleNamespace(
        color=np.zeros((2, 2, 3), dtype=np.uint8),
        depth=np.ones((2, 2), dtype=np.uint16),
        intrinsics=SimpleNamespace(width=2, height=2, fx=1, fy=1, cx=0, cy=0),
    )
    candidates = [
        SimpleNamespace(
            translation=np.array([0.1, 0.0, 0.2]),
            rotation_matrix=np.eye(3),
            score=0.9,
            width=0.04,
        )
    ]
    debug_data = SimpleNamespace(
        cloud_points=np.array([[0.1, 0.0, 0.2]]),
        cloud_colors=np.array([[255, 0, 0]], dtype=np.uint8),
        workspace_mask=np.ones((2, 2), dtype=bool),
    )

    saved_files = visualization.save_debug_artifacts(
        frame,
        candidates,
        debug_data,
        {"grasp_pose": [0.1, 0.2, 0.3, 0.0, 0.0, 0.0]},
        {
            "debug_dir": str(tmp_path),
            "save_point_cloud": False,
            "save_3d_grasp_image": True,
            "top_n": 20,
        },
    )

    grasp_images = [path for path in saved_files if Path(path).name == "3d_grasps.png"]
    assert len(grasp_images) == 1
    assert (tmp_path / Path(grasp_images[0]).name).read_bytes() == b"3d-grasps"


def test_save_debug_artifacts_keeps_existing_3d_grasp_images(monkeypatch, tmp_path):
    from graspnt_rm import visualization

    monkeypatch.setattr(visualization, "_import_cv2", lambda: FakeCv2())
    monkeypatch.setattr(visualization, "_import_open3d", lambda: FakeOpen3D)

    existing_path = tmp_path / "3d_grasps.png"
    existing_path.write_bytes(b"previous")

    frame = SimpleNamespace(
        color=np.zeros((2, 2, 3), dtype=np.uint8),
        depth=np.ones((2, 2), dtype=np.uint16),
        intrinsics=SimpleNamespace(width=2, height=2, fx=1, fy=1, cx=0, cy=0),
    )
    candidates = [
        SimpleNamespace(
            translation=np.array([0.1, 0.0, 0.2]),
            rotation_matrix=np.eye(3),
            score=0.9,
            width=0.04,
        )
    ]
    debug_data = SimpleNamespace(
        cloud_points=np.array([[0.1, 0.0, 0.2]]),
        cloud_colors=np.array([[255, 0, 0]], dtype=np.uint8),
        workspace_mask=np.ones((2, 2), dtype=bool),
    )

    saved_files = visualization.save_debug_artifacts(
        frame,
        candidates,
        debug_data,
        {"grasp_pose": [0.1, 0.2, 0.3, 0.0, 0.0, 0.0]},
        {
            "debug_dir": str(tmp_path),
            "save_point_cloud": False,
            "save_3d_grasp_image": True,
            "top_n": 20,
        },
    )

    grasp_images = [path for path in saved_files if Path(path).name == "3d_grasps_001.png"]
    assert len(grasp_images) == 1
    assert existing_path.read_bytes() == b"previous"
    assert (tmp_path / "3d_grasps_001.png").read_bytes() == b"3d-grasps"
