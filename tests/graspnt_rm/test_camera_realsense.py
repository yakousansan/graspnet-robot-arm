import builtins
import importlib
import sys
import types

import numpy as np
import pytest


def import_camera_module(monkeypatch, rs_module):
    sys.modules.pop("graspnt_rm.camera_realsense", None)
    if rs_module is None:
        monkeypatch.delitem(sys.modules, "pyrealsense2", raising=False)
        original_import = builtins.__import__

        def fake_import(name, *args, **kwargs):
            if name == "pyrealsense2":
                raise ModuleNotFoundError("No module named 'pyrealsense2'")
            return original_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", fake_import)
    else:
        monkeypatch.setitem(sys.modules, "pyrealsense2", rs_module)
    return importlib.import_module("graspnt_rm.camera_realsense")


def test_import_succeeds_without_pyrealsense2_but_start_explains_requirement(monkeypatch):
    camera_module = import_camera_module(monkeypatch, None)

    camera = camera_module.RealSenseCamera()

    with pytest.raises(RuntimeError, match="pyrealsense2 is required"):
        camera.start()


def test_capture_aligns_depth_to_color_and_converts_depth_scale(monkeypatch):
    color = np.zeros((2, 3, 3), dtype=np.uint8)
    depth = np.array([[100, 200, 300], [400, 500, 600]], dtype=np.uint16)
    rs = FakeRealSenseModule(color=color, depth=depth, depth_scale=0.001)
    camera_module = import_camera_module(monkeypatch, rs)

    camera = camera_module.RealSenseCamera(width=3, height=2, fps=15)
    camera.start()
    frame = camera.capture(warmup_frames=2)

    assert rs.config.enabled_streams == [
        (rs.stream.color, 3, 2, rs.format.bgr8, 15),
        (rs.stream.depth, 3, 2, rs.format.z16, 15),
    ]
    assert rs.align.stream == rs.stream.color
    assert rs.align.process_calls == 3
    assert rs.pipeline.wait_calls == 3
    np.testing.assert_array_equal(frame.color, color)
    np.testing.assert_array_equal(frame.depth, depth)
    assert frame.depth_scale == pytest.approx(0.001)
    assert frame.intrinsics == camera_module.CameraIntrinsics(
        width=3,
        height=2,
        fx=610.5,
        fy=611.5,
        cx=1.5,
        cy=1.0,
        scale=1000.0,
    )

    camera.stop()
    assert rs.pipeline.stop_calls == 1


def test_start_rejects_non_positive_depth_scale(monkeypatch):
    rs = FakeRealSenseModule(
        color=np.zeros((2, 3, 3), dtype=np.uint8),
        depth=np.zeros((2, 3), dtype=np.uint16),
        depth_scale=0.0,
    )
    camera_module = import_camera_module(monkeypatch, rs)

    with pytest.raises(RuntimeError, match="Invalid RealSense depth scale"):
        camera_module.RealSenseCamera().start()


class FakeRealSenseModule:
    def __init__(self, color, depth, depth_scale):
        self.stream = types.SimpleNamespace(color="color", depth="depth")
        self.format = types.SimpleNamespace(bgr8="bgr8", z16="z16")
        self.config = FakeConfigFactory()
        self.pipeline = FakePipelineFactory(color, depth, depth_scale)
        self.align = FakeAlignFactory()


class FakeConfigFactory:
    def __call__(self):
        self.instance = FakeConfig()
        return self.instance

    @property
    def enabled_streams(self):
        return self.instance.enabled_streams


class FakeConfig:
    def __init__(self):
        self.enabled_streams = []

    def enable_stream(self, stream, width, height, image_format, fps):
        self.enabled_streams.append((stream, width, height, image_format, fps))


class FakePipelineFactory:
    def __init__(self, color, depth, depth_scale):
        self._color = color
        self._depth = depth
        self._depth_scale = depth_scale

    def __call__(self):
        self.instance = FakePipeline(self._color, self._depth, self._depth_scale)
        return self.instance

    @property
    def wait_calls(self):
        return self.instance.wait_calls

    @property
    def stop_calls(self):
        return self.instance.stop_calls


class FakePipeline:
    def __init__(self, color, depth, depth_scale):
        self._color = color
        self._depth = depth
        self._depth_scale = depth_scale
        self.started_with = None
        self.wait_calls = 0
        self.stop_calls = 0

    def start(self, config):
        self.started_with = config
        return FakeProfile(self._depth_scale)

    def wait_for_frames(self):
        self.wait_calls += 1
        return FakeFrameset(self._color, self._depth)

    def stop(self):
        self.stop_calls += 1


class FakeAlignFactory:
    def __call__(self, stream):
        align = FakeAlign(stream)
        self._align = align
        return align

    @property
    def stream(self):
        return self._align.stream

    @property
    def process_calls(self):
        return self._align.process_calls


class FakeAlign:
    def __init__(self, stream):
        self.stream = stream
        self.process_calls = 0

    def process(self, frameset):
        self.process_calls += 1
        return frameset


class FakeProfile:
    def __init__(self, depth_scale):
        self._depth_scale = depth_scale

    def get_device(self):
        return FakeDevice(self._depth_scale)


class FakeDevice:
    def __init__(self, depth_scale):
        self._depth_scale = depth_scale

    def first_depth_sensor(self):
        return FakeDepthSensor(self._depth_scale)


class FakeDepthSensor:
    def __init__(self, depth_scale):
        self._depth_scale = depth_scale

    def get_depth_scale(self):
        return self._depth_scale


class FakeFrameset:
    def __init__(self, color, depth):
        self._color = color
        self._depth = depth

    def get_color_frame(self):
        return FakeColorFrame(self._color)

    def get_depth_frame(self):
        return FakeDepthFrame(self._depth)


class FakeColorFrame:
    def __init__(self, color):
        self._color = color

    def get_data(self):
        return self._color


class FakeDepthFrame:
    def __init__(self, depth):
        self._depth = depth
        self.profile = FakeDepthFrameProfile()

    def get_data(self):
        return self._depth


class FakeDepthFrameProfile:
    def as_video_stream_profile(self):
        return FakeVideoStreamProfile()


class FakeVideoStreamProfile:
    @property
    def intrinsics(self):
        return types.SimpleNamespace(
            width=3,
            height=2,
            fx=610.5,
            fy=611.5,
            ppx=1.5,
            ppy=1.0,
        )
