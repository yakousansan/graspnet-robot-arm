from types import SimpleNamespace

import numpy as np


def test_video_recorder_uses_incremented_name_and_releases_writer(monkeypatch, tmp_path):
    from graspnt_rm import video_recorder

    writers = []

    class FakeWriter:
        def __init__(self, path, fourcc, fps, size):
            self.path = path
            self.fourcc = fourcc
            self.fps = fps
            self.size = size
            self.frames = []
            self.released = False
            writers.append(self)

        def isOpened(self):
            return True

        def write(self, frame):
            self.frames.append(frame.copy())

        def release(self):
            self.released = True

    class FakeCv2:
        @staticmethod
        def VideoWriter_fourcc(*codec):
            return "".join(codec)

        VideoWriter = FakeWriter

    class FakeCamera:
        def capture(self, warmup_frames=0):
            return SimpleNamespace(
                color=np.zeros((3, 4, 3), dtype=np.uint8),
            )

    monkeypatch.setattr(video_recorder, "_import_cv2", lambda: FakeCv2)
    existing_path = tmp_path / "grasp_video.avi"
    existing_path.write_bytes(b"previous")

    recorder = video_recorder.VideoRecorder(
        FakeCamera(),
        {
            "output_dir": str(tmp_path),
            "filename_stem": "grasp_video",
            "extension": ".avi",
            "codec": "MJPG",
            "fps": 30,
        },
    )

    path = recorder.start()
    stopped_path = recorder.stop()

    assert path == str(tmp_path / "grasp_video_001.avi")
    assert stopped_path == path
    assert existing_path.read_bytes() == b"previous"
    assert writers[0].fourcc == "MJPG"
    assert writers[0].size == (4, 3)
    assert writers[0].frames
    assert writers[0].released is True
