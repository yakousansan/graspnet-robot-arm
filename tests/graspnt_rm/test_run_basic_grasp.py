from types import SimpleNamespace
import importlib
import sys

import numpy as np
import pytest

from graspnt_rm.transform import offset_pose_along_base_z, offset_pose_along_local_z


def import_orchestrator(monkeypatch):
    sys.modules.pop("graspnt_rm.run_basic_grasp", None)
    return importlib.import_module("graspnt_rm.run_basic_grasp")


def base_config(**overrides):
    config = {
        "graspnet": {
            "root": "/tmp/graspnet",
            "checkpoint": "/tmp/checkpoint.tar",
            "num_point": 20000,
            "num_view": 300,
        },
        "camera": {"width": 640, "height": 480, "fps": 30},
        "hand_eye": {
            "rotation": np.eye(3).reshape(-1).tolist(),
            "translation": [0.0, 0.0, 0.0],
        },
        "workspace": {"mode": "all"},
        "safety": {
            "gripper_length": 0.08,
            "min_grasp_z": 0.05,
            "pre_grasp_offset": 0.08,
            "lift_offset": 0.10,
        },
    }
    for key, value in overrides.items():
        config[key].update(value)
    return config


def test_build_plan_computes_six_value_poses(monkeypatch):
    run_basic_grasp = import_orchestrator(monkeypatch)
    candidate = SimpleNamespace(
        translation=np.array([0.10, 0.20, 0.30]),
        rotation_matrix=np.eye(3),
        score=0.91,
        width=0.045,
    )

    plan = run_basic_grasp.build_plan(
        base_config(safety={"gripper_length": 0.02}),
        candidate,
        current_end_pose=[0.40, -0.10, 0.20, 0.0, 0.0, 0.0],
    )

    assert plan["score"] == pytest.approx(0.91)
    assert plan["width"] == pytest.approx(0.045)
    assert len(plan["grasp_pose"]) == 6
    assert len(plan["pre_grasp_pose"]) == 6
    assert len(plan["lift_pose"]) == 6
    np.testing.assert_allclose(
        plan["pre_grasp_pose"],
        offset_pose_along_local_z(plan["grasp_pose"], -0.08),
        atol=1e-12,
    )
    np.testing.assert_allclose(
        plan["lift_pose"],
        offset_pose_along_base_z(plan["grasp_pose"], 0.10),
        atol=1e-12,
    )


def test_build_plan_accepts_dict_candidate(monkeypatch):
    run_basic_grasp = import_orchestrator(monkeypatch)
    candidate = {
        "translation": [0.10, 0.20, 0.30],
        "rotation_matrix": np.eye(3).tolist(),
        "score": 0.72,
        "width": 0.035,
    }

    plan = run_basic_grasp.build_plan(
        base_config(),
        candidate,
        current_end_pose=[0.40, -0.10, 0.20, 0.0, 0.0, 0.0],
    )

    assert plan["score"] == pytest.approx(0.72)
    assert plan["width"] == pytest.approx(0.035)
    assert len(plan["grasp_pose"]) == 6


def test_build_plan_rejects_unsafe_base_z_lift_pose(monkeypatch):
    run_basic_grasp = import_orchestrator(monkeypatch)
    candidate = {
        "translation": [0.10, 0.20, 0.30],
        "rotation_matrix": np.eye(3).tolist(),
        "score": 0.72,
        "width": 0.035,
    }

    with pytest.raises(ValueError, match="lift_pose"):
        run_basic_grasp.build_plan(
            base_config(
                safety={
                        "lift_offset": 0.25,
                        "workspace_bounds": {
                            "x": [-1.0, 1.0],
                            "y": [-1.0, 1.0],
                            "z": [0.05, 0.6],
                        }
                    }
                ),
            candidate,
            current_end_pose=[0.40, -0.10, 0.20, 0.0, 0.0, 0.0],
        )


def test_run_calls_visualization_when_enabled(monkeypatch, tmp_path):
    run_basic_grasp = import_orchestrator(monkeypatch)
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
graspnet:
  root: /tmp/graspnet
  checkpoint: /tmp/checkpoint.tar
  num_point: 4
  num_view: 300
camera:
  width: 320
  height: 240
  fps: 15
hand_eye:
  rotation: [1, 0, 0, 0, 1, 0, 0, 0, 1]
  translation: [0, 0, 0]
workspace:
  mode: all
safety:
  gripper_length: 0.08
  min_grasp_z: 0.05
  pre_grasp_offset: 0.08
  lift_offset: 0.10
visualization:
  enabled: true
  mode: save_only
  top_n: 5
  save_debug: false
""",
        encoding="utf-8",
    )
    calls = []

    class FakeCamera:
        def __init__(self, width, height, fps):
            pass

        def start(self):
            pass

        def capture(self):
            return SimpleNamespace(
                color=np.zeros((2, 2, 3), dtype=np.uint8),
                depth=np.ones((2, 2), dtype=np.uint16),
                intrinsics=SimpleNamespace(width=2, height=2, fx=1, fy=1, cx=0, cy=0),
            )

        def stop(self):
            pass

    class FakeRunner:
        def __init__(self, config):
            self.last_debug = SimpleNamespace(
                workspace_mask=np.ones((2, 2), dtype=bool),
                cloud_points=np.zeros((4, 3), dtype=np.float32),
                cloud_colors=np.zeros((4, 3), dtype=np.float32),
            )

        def infer(self, color, depth, intrinsics, workspace_config):
            return [
                SimpleNamespace(
                    translation=np.array([0.1, 0.2, 0.3]),
                    rotation_matrix=np.eye(3),
                    score=0.8,
                    width=0.04,
                )
            ], {"candidate_count": 1, "valid_workspace_points": 4}

    class FakeUdpClient:
        def __init__(self, **kwargs):
            pass

        def request_pose(self):
            return {"type": "pose_response", "status": "ok", "end_pose": [0.4, -0.1, 0.2, 0, 0, 0]}

        def execute_grasp(self, plan):
            return {"type": "result", "status": "success"}

        def close(self):
            pass

    def fake_visualize_debug(frame, candidates, debug_data, plan, config):
        calls.append(
            (
                "visualize",
                len(candidates),
                debug_data.workspace_mask.shape,
                plan["score"],
                config["top_n"],
            )
        )
        return {"enabled": True}

    monkeypatch.setattr(run_basic_grasp, "RealSenseCamera", FakeCamera)
    monkeypatch.setattr(run_basic_grasp, "GraspNetRunner", FakeRunner)
    monkeypatch.setattr(run_basic_grasp, "UdpRobotClient", FakeUdpClient)
    monkeypatch.setattr(run_basic_grasp, "visualize_debug", fake_visualize_debug)

    result = run_basic_grasp.run(config_path)

    assert calls == [("visualize", 1, (2, 2), 0.8, 5)]
    assert result["visualization"] == {"enabled": True}


def test_run_visualizes_before_rejecting_unsafe_plan(monkeypatch, tmp_path):
    run_basic_grasp = import_orchestrator(monkeypatch)
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
graspnet:
  root: /tmp/graspnet
  checkpoint: /tmp/checkpoint.tar
  num_point: 4
  num_view: 300
camera:
  width: 320
  height: 240
  fps: 15
hand_eye:
  rotation: [1, 0, 0, 0, 1, 0, 0, 0, 1]
  translation: [0, 0, 0]
workspace:
  mode: all
safety:
  gripper_length: 0.08
  min_grasp_z: 0.05
  pre_grasp_offset: 0.08
  lift_offset: 0.10
visualization:
  enabled: true
  mode: save_only
  top_n: 5
  save_debug: false
""",
        encoding="utf-8",
    )
    calls = []

    class FakeCamera:
        def __init__(self, width, height, fps):
            pass

        def start(self):
            pass

        def capture(self):
            return SimpleNamespace(
                color=np.zeros((2, 2, 3), dtype=np.uint8),
                depth=np.ones((2, 2), dtype=np.uint16),
                intrinsics=SimpleNamespace(width=2, height=2, fx=1, fy=1, cx=0, cy=0),
            )

        def stop(self):
            calls.append(("camera_stop",))

    class FakeRunner:
        def __init__(self, config):
            self.last_debug = None

        def infer(self, color, depth, intrinsics, workspace_config):
            return [
                SimpleNamespace(
                    translation=np.array([0.1, 0.2, -0.4]),
                    rotation_matrix=np.eye(3),
                    score=0.8,
                    width=0.04,
                )
            ], {"candidate_count": 1, "valid_workspace_points": 4}

    class FakeUdpClient:
        def __init__(self, **kwargs):
            pass

        def request_pose(self):
            return {"type": "pose_response", "status": "ok", "end_pose": [0.4, -0.1, 0.2, 0, 0, 0]}

        def close(self):
            calls.append(("udp_close",))

    def fake_visualize_debug(frame, candidates, debug_data, plan, config):
        calls.append(("visualize", plan["grasp_pose"][2]))
        return {"enabled": True}

    monkeypatch.setattr(run_basic_grasp, "RealSenseCamera", FakeCamera)
    monkeypatch.setattr(run_basic_grasp, "GraspNetRunner", FakeRunner)
    monkeypatch.setattr(run_basic_grasp, "UdpRobotClient", FakeUdpClient)
    monkeypatch.setattr(run_basic_grasp, "visualize_debug", fake_visualize_debug)

    with pytest.raises(ValueError, match="grasp_pose"):
        run_basic_grasp.run(config_path)

    assert calls[0][0] == "visualize"
    assert calls[-2:] == [("camera_stop",), ("udp_close",)]


def test_run_udp_cpp_sends_plan_without_python_robot_sdk(
    monkeypatch,
    tmp_path,
):
    run_basic_grasp = import_orchestrator(monkeypatch)
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
graspnet:
  root: /tmp/graspnet
  checkpoint: /tmp/checkpoint.tar
  num_point: 4
  num_view: 300
camera:
  width: 320
  height: 240
  fps: 15
hand_eye:
  rotation: [1, 0, 0, 0, 1, 0, 0, 0, 1]
  translation: [0, 0, 0]
workspace:
  mode: all
safety:
  gripper_length: 0.08
  min_grasp_z: 0.05
  pre_grasp_offset: 0.08
  lift_offset: 0.10
execution:
  backend: udp_cpp
  udp_host: 127.0.0.1
  udp_port: 6556
visualization:
  enabled: false
""",
        encoding="utf-8",
    )
    calls = []

    class FakeCamera:
        def __init__(self, width, height, fps):
            pass

        def start(self):
            pass

        def capture(self):
            calls.append(("capture",))
            return SimpleNamespace(
                color=np.zeros((2, 2, 3), dtype=np.uint8),
                depth=np.ones((2, 2), dtype=np.uint16),
                intrinsics=SimpleNamespace(width=2, height=2, fx=1, fy=1, cx=0, cy=0),
            )

        def stop(self):
            pass

    class FakeRunner:
        def __init__(self, config):
            self.last_debug = None

        def infer(self, color, depth, intrinsics, workspace_config):
            return [
                SimpleNamespace(
                    translation=np.array([0.1, 0.2, 0.3]),
                    rotation_matrix=np.eye(3),
                    score=0.8,
                    width=0.04,
                )
            ], {"candidate_count": 1, "valid_workspace_points": 4}

    class FakeUdpClient:
        def __init__(self, **kwargs):
            calls.append(("udp_init", kwargs["host"], kwargs["port"]))

        def request_pose(self):
            calls.append(("pose_request",))
            return {"type": "pose_response", "status": "ok", "end_pose": [0.4, -0.1, 0.2, 0, 0, 0]}

        def execute_grasp(self, *args, **kwargs):
            calls.append(("execute_grasp", args, kwargs))
            return {"type": "result", "status": "success"}

        def close(self):
            calls.append(("udp_close",))

    monkeypatch.setattr(run_basic_grasp, "RealSenseCamera", FakeCamera)
    monkeypatch.setattr(run_basic_grasp, "GraspNetRunner", FakeRunner)
    monkeypatch.setattr(run_basic_grasp, "UdpRobotClient", FakeUdpClient)

    result = run_basic_grasp.run(config_path)

    assert result["robot_state"]["end_pose"] == [0.4, -0.1, 0.2, 0, 0, 0]
    assert ("execute_grasp",) in [call[:1] for call in calls]
    assert result["execution_result"] == {"type": "result", "status": "success"}
    assert calls[:3] == [
        ("capture",),
        ("udp_init", "127.0.0.1", 6556),
        ("pose_request",),
    ]
    assert calls[-1] == ("udp_close",)


def test_run_previews_workspace_before_requesting_executor_pose(monkeypatch, tmp_path):
    run_basic_grasp = import_orchestrator(monkeypatch)
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
graspnet:
  root: /tmp/graspnet
  checkpoint: /tmp/checkpoint.tar
  num_point: 4
  num_view: 300
camera:
  width: 320
  height: 240
  fps: 15
hand_eye:
  rotation: [1, 0, 0, 0, 1, 0, 0, 0, 1]
  translation: [0, 0, 0]
workspace:
  mode: all
camera_preview:
  enabled: true
  show_depth: true
safety:
  gripper_length: 0.08
  min_grasp_z: 0.05
  pre_grasp_offset: 0.08
  lift_offset: 0.10
execution:
  backend: udp_cpp
  udp_host: 127.0.0.1
  udp_port: 6556
visualization:
  enabled: false
""",
        encoding="utf-8",
    )
    calls = []
    frame = SimpleNamespace(
        color=np.zeros((2, 2, 3), dtype=np.uint8),
        depth=np.ones((2, 2), dtype=np.uint16),
        intrinsics=SimpleNamespace(width=2, height=2, fx=1, fy=1, cx=0, cy=0),
    )

    class FakeCamera:
        def __init__(self, width, height, fps):
            pass

        def start(self):
            calls.append(("camera_start",))

        def capture(self):
            calls.append(("capture",))
            return frame

        def stop(self):
            calls.append(("camera_stop",))

    class FakeRunner:
        def __init__(self, config):
            calls.append(("runner_init",))
            self.last_debug = None

        def infer(self, color, depth, intrinsics, workspace_config):
            calls.append(("infer",))
            return [
                SimpleNamespace(
                    translation=np.array([0.1, 0.2, 0.3]),
                    rotation_matrix=np.eye(3),
                    score=0.8,
                    width=0.04,
                )
            ], {"candidate_count": 1, "valid_workspace_points": 4}

    class FakeUdpClient:
        def __init__(self, **kwargs):
            calls.append(("udp_init",))

        def request_pose(self):
            calls.append(("pose_request",))
            return {"type": "pose_response", "status": "ok", "end_pose": [0.4, -0.1, 0.2, 0, 0, 0]}

        def execute_grasp(self, plan):
            calls.append(("execute_grasp",))
            return {"type": "result", "status": "success"}

        def close(self):
            calls.append(("udp_close",))

    def fake_preview_workspace(camera, workspace_config, preview_config):
        calls.append(("preview", preview_config["show_depth"]))
        return frame

    monkeypatch.setattr(run_basic_grasp, "RealSenseCamera", FakeCamera)
    monkeypatch.setattr(run_basic_grasp, "GraspNetRunner", FakeRunner)
    monkeypatch.setattr(run_basic_grasp, "UdpRobotClient", FakeUdpClient)
    monkeypatch.setattr(run_basic_grasp, "preview_workspace", fake_preview_workspace)

    run_basic_grasp.run(config_path)

    assert calls.index(("preview", True)) < calls.index(("pose_request",))
    assert calls.index(("pose_request",)) < calls.index(("infer",))


def test_run_stops_when_workspace_preview_is_cancelled(monkeypatch, tmp_path):
    run_basic_grasp = import_orchestrator(monkeypatch)
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
graspnet:
  root: /tmp/graspnet
  checkpoint: /tmp/checkpoint.tar
  num_point: 4
  num_view: 300
camera:
  width: 320
  height: 240
  fps: 15
hand_eye:
  rotation: [1, 0, 0, 0, 1, 0, 0, 0, 1]
  translation: [0, 0, 0]
workspace:
  mode: all
camera_preview:
  enabled: true
safety:
  gripper_length: 0.08
  min_grasp_z: 0.05
  pre_grasp_offset: 0.08
  lift_offset: 0.10
execution:
  backend: udp_cpp
  udp_host: 127.0.0.1
  udp_port: 6556
visualization:
  enabled: false
""",
        encoding="utf-8",
    )
    calls = []

    class FakeCamera:
        def __init__(self, width, height, fps):
            pass

        def start(self):
            calls.append(("camera_start",))

        def stop(self):
            calls.append(("camera_stop",))

    class FakeRunner:
        def __init__(self, config):
            calls.append(("runner_init",))

    class FakeUdpClient:
        def __init__(self, **kwargs):
            calls.append(("udp_init",))

    def fake_preview_workspace(camera, workspace_config, preview_config):
        calls.append(("preview",))
        return None

    monkeypatch.setattr(run_basic_grasp, "RealSenseCamera", FakeCamera)
    monkeypatch.setattr(run_basic_grasp, "GraspNetRunner", FakeRunner)
    monkeypatch.setattr(run_basic_grasp, "UdpRobotClient", FakeUdpClient)
    monkeypatch.setattr(run_basic_grasp, "preview_workspace", fake_preview_workspace)

    with pytest.raises(RuntimeError, match="preview cancelled"):
        run_basic_grasp.run(config_path)

    assert calls == [("camera_start",), ("preview",), ("camera_stop",)]


def test_run_udp_cpp_sends_plan_without_python_confirmation(monkeypatch, tmp_path):
    run_basic_grasp = import_orchestrator(monkeypatch)
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
graspnet:
  root: /tmp/graspnet
  checkpoint: /tmp/checkpoint.tar
  num_point: 4
  num_view: 300
camera:
  width: 320
  height: 240
  fps: 15
hand_eye:
  rotation: [1, 0, 0, 0, 1, 0, 0, 0, 1]
  translation: [0, 0, 0]
workspace:
  mode: all
safety:
  gripper_length: 0.08
  min_grasp_z: 0.05
  pre_grasp_offset: 0.08
  lift_offset: 0.10
execution:
  backend: udp_cpp
  udp_host: 127.0.0.1
  udp_port: 6556
visualization:
  enabled: false
""",
        encoding="utf-8",
    )
    calls = []

    class FakeCamera:
        def __init__(self, width, height, fps):
            pass

        def start(self):
            pass

        def capture(self):
            return SimpleNamespace(
                color=np.zeros((2, 2, 3), dtype=np.uint8),
                depth=np.ones((2, 2), dtype=np.uint16),
                intrinsics=SimpleNamespace(width=2, height=2, fx=1, fy=1, cx=0, cy=0),
            )

        def stop(self):
            pass

    class FakeRunner:
        def __init__(self, config):
            self.last_debug = None

        def infer(self, color, depth, intrinsics, workspace_config):
            return [
                SimpleNamespace(
                    translation=np.array([0.1, 0.2, 0.3]),
                    rotation_matrix=np.eye(3),
                    score=0.8,
                    width=0.04,
                )
            ], {"candidate_count": 1, "valid_workspace_points": 4}

    class FakeUdpClient:
        def __init__(self, **kwargs):
            pass

        def request_pose(self):
            return {"type": "pose_response", "status": "ok", "end_pose": [0.4, -0.1, 0.2, 0, 0, 0]}

        def execute_grasp(self, plan):
            calls.append(("execute_grasp", plan["score"]))
            return {"type": "result", "status": "success"}

        def close(self):
            pass

    monkeypatch.setattr(run_basic_grasp, "RealSenseCamera", FakeCamera)
    monkeypatch.setattr(run_basic_grasp, "GraspNetRunner", FakeRunner)
    monkeypatch.setattr(run_basic_grasp, "UdpRobotClient", FakeUdpClient)

    result = run_basic_grasp.run(config_path)

    assert calls == [("execute_grasp", 0.8)]
    assert result["execution_result"] == {"type": "result", "status": "success"}


def test_run_records_video_around_udp_execution(monkeypatch, tmp_path):
    run_basic_grasp = import_orchestrator(monkeypatch)
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
graspnet:
  root: /tmp/graspnet
  checkpoint: /tmp/checkpoint.tar
  num_point: 4
  num_view: 300
camera:
  width: 320
  height: 240
  fps: 15
hand_eye:
  rotation: [1, 0, 0, 0, 1, 0, 0, 0, 1]
  translation: [0, 0, 0]
workspace:
  mode: all
safety:
  gripper_length: 0.08
  min_grasp_z: 0.05
  pre_grasp_offset: 0.08
  lift_offset: 0.10
execution:
  backend: udp_cpp
recording:
  enabled: true
  output_dir: videos
  filename_stem: grasp_video
  extension: .avi
  codec: MJPG
  fps: 15
visualization:
  enabled: false
""",
        encoding="utf-8",
    )
    calls = []

    class FakeCamera:
        def __init__(self, width, height, fps):
            pass

        def start(self):
            calls.append(("camera_start",))

        def capture(self):
            calls.append(("capture",))
            return SimpleNamespace(
                color=np.zeros((2, 2, 3), dtype=np.uint8),
                depth=np.ones((2, 2), dtype=np.uint16),
                intrinsics=SimpleNamespace(width=2, height=2, fx=1, fy=1, cx=0, cy=0),
            )

        def stop(self):
            calls.append(("camera_stop",))

    class FakeRunner:
        def __init__(self, config):
            self.last_debug = None

        def infer(self, color, depth, intrinsics, workspace_config):
            return [
                SimpleNamespace(
                    translation=np.array([0.1, 0.2, 0.3]),
                    rotation_matrix=np.eye(3),
                    score=0.8,
                    width=0.04,
                )
            ], {"candidate_count": 1, "valid_workspace_points": 4}

    class FakeUdpClient:
        def __init__(self, **kwargs):
            pass

        def request_pose(self):
            return {"type": "pose_response", "status": "ok", "end_pose": [0.4, -0.1, 0.2, 0, 0, 0]}

        def execute_grasp(self, plan):
            calls.append(("execute_grasp",))
            return {"type": "result", "status": "success"}

        def close(self):
            calls.append(("udp_close",))

    class FakeVideoRecorder:
        def __init__(self, camera, config):
            calls.append(("recorder_init", config["output_dir"], config["codec"]))

        def start(self):
            calls.append(("record_start",))

        def stop(self):
            calls.append(("record_stop",))
            return "videos/grasp_video.avi"

    monkeypatch.setattr(run_basic_grasp, "RealSenseCamera", FakeCamera)
    monkeypatch.setattr(run_basic_grasp, "GraspNetRunner", FakeRunner)
    monkeypatch.setattr(run_basic_grasp, "UdpRobotClient", FakeUdpClient)
    monkeypatch.setattr(run_basic_grasp, "VideoRecorder", FakeVideoRecorder, raising=False)

    result = run_basic_grasp.run(config_path)

    assert calls.index(("record_start",)) < calls.index(("execute_grasp",))
    assert calls.index(("execute_grasp",)) < calls.index(("record_stop",))
    assert result["recording"] == {"enabled": True, "video_path": "videos/grasp_video.avi"}


def test_run_stops_video_recording_when_udp_execution_fails(monkeypatch, tmp_path):
    run_basic_grasp = import_orchestrator(monkeypatch)
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
graspnet:
  root: /tmp/graspnet
  checkpoint: /tmp/checkpoint.tar
  num_point: 4
  num_view: 300
camera:
  width: 320
  height: 240
  fps: 15
hand_eye:
  rotation: [1, 0, 0, 0, 1, 0, 0, 0, 1]
  translation: [0, 0, 0]
workspace:
  mode: all
safety:
  gripper_length: 0.08
  min_grasp_z: 0.05
  pre_grasp_offset: 0.08
  lift_offset: 0.10
execution:
  backend: udp_cpp
recording:
  enabled: true
visualization:
  enabled: false
""",
        encoding="utf-8",
    )
    calls = []

    class FakeCamera:
        def __init__(self, width, height, fps):
            pass

        def start(self):
            pass

        def capture(self):
            return SimpleNamespace(
                color=np.zeros((2, 2, 3), dtype=np.uint8),
                depth=np.ones((2, 2), dtype=np.uint16),
                intrinsics=SimpleNamespace(width=2, height=2, fx=1, fy=1, cx=0, cy=0),
            )

        def stop(self):
            calls.append(("camera_stop",))

    class FakeRunner:
        def __init__(self, config):
            self.last_debug = None

        def infer(self, color, depth, intrinsics, workspace_config):
            return [
                SimpleNamespace(
                    translation=np.array([0.1, 0.2, 0.3]),
                    rotation_matrix=np.eye(3),
                    score=0.8,
                    width=0.04,
                )
            ], {"candidate_count": 1, "valid_workspace_points": 4}

    class FakeUdpClient:
        def __init__(self, **kwargs):
            pass

        def request_pose(self):
            return {"type": "pose_response", "status": "ok", "end_pose": [0.4, -0.1, 0.2, 0, 0, 0]}

        def execute_grasp(self, plan):
            calls.append(("execute_grasp",))
            raise RuntimeError("robot failed")

        def close(self):
            calls.append(("udp_close",))

    class FakeVideoRecorder:
        def __init__(self, camera, config):
            pass

        def start(self):
            calls.append(("record_start",))

        def stop(self):
            calls.append(("record_stop",))
            return "debug_outputs/videos/grasp_video.avi"

    monkeypatch.setattr(run_basic_grasp, "RealSenseCamera", FakeCamera)
    monkeypatch.setattr(run_basic_grasp, "GraspNetRunner", FakeRunner)
    monkeypatch.setattr(run_basic_grasp, "UdpRobotClient", FakeUdpClient)
    monkeypatch.setattr(run_basic_grasp, "VideoRecorder", FakeVideoRecorder, raising=False)

    with pytest.raises(RuntimeError, match="robot failed"):
        run_basic_grasp.run(config_path)

    assert calls[:3] == [("record_start",), ("execute_grasp",), ("record_stop",)]
