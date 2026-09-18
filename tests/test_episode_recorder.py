import json
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import cv2
import numpy as np

from spatial_interface.episode_recorder import ActMode, EpisodeRecorder, _FFmpegWriter


class VideoEncoderTests(unittest.TestCase):
    @unittest.skipUnless(shutil.which("ffmpeg") and shutil.which("ffprobe"), "FFmpeg tools required")
    def test_unavailable_opencv_encoder_falls_back_to_decodable_h264(self):
        with tempfile.TemporaryDirectory() as folder:
            failed = mock.Mock()
            failed.isOpened.return_value = False
            with mock.patch("spatial_interface.episode_recorder.cv2.VideoWriter", return_value=failed):
                recorder = EpisodeRecorder(folder, "cameras")
                for index in range(12):
                    frame = np.full((32, 32, 3), (index * 15, 80, 160), dtype=np.uint8)
                    obs = {f"camera{k}_image": frame for k in range(4)}
                    mode = ActMode.Waypoint if index == 0 else ActMode.Interpolate
                    recorder.record(mode, obs, np.zeros(7))
                recorder.end_episode(save=True)
            failed.release.assert_called_once()
            video = Path(folder) / "cameras.mp4"
            result = subprocess.run(
                ["ffprobe", "-v", "error", "-select_streams", "v:0",
                 "-show_entries", "stream=codec_name,pix_fmt,nb_frames,width,height",
                 "-of", "json", str(video)],
                text=True, capture_output=True, check=True, timeout=15,
            )
            stream = json.loads(result.stdout)["streams"][0]
            self.assertEqual(stream["codec_name"], "h264")
            self.assertEqual(stream["pix_fmt"], "yuv420p")
            self.assertEqual(int(stream["nb_frames"]), 12)
            cap = cv2.VideoCapture(str(video))
            try:
                ok, decoded = cap.read()
                self.assertTrue(ok)
                self.assertEqual(decoded.shape, (480, 640, 3))
            finally:
                cap.release()
            segments = json.loads((Path(folder) / "cameras_segments.json").read_text())
            self.assertEqual(segments["total_frames"], 12)
            self.assertFalse((Path(folder) / "cameras.npz").exists())

    def test_missing_ffmpeg_raises_instead_of_claiming_video_saved(self):
        with mock.patch("spatial_interface.episode_recorder.shutil.which", return_value=None):
            with self.assertRaisesRegex(RuntimeError, "ffmpeg is unavailable"):
                _FFmpegWriter("unused.mp4", 30, (32, 32))

    def test_encoder_nonzero_exit_is_reported(self):
        with tempfile.TemporaryDirectory() as folder:
            executable = Path(folder) / "failing_encoder"
            executable.write_text("#!/bin/sh\necho 'test encoder failure' >&2\nexit 9\n")
            executable.chmod(0o700)
            with mock.patch("spatial_interface.episode_recorder.shutil.which", return_value=str(executable)):
                writer = _FFmpegWriter(str(Path(folder) / "cameras.mp4"), 30, (32, 32))
                with self.assertRaisesRegex(RuntimeError, "exit 9.*test encoder failure"):
                    writer.release()


if __name__ == "__main__":
    unittest.main()
