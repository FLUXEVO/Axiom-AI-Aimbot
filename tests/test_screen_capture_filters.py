from unittest.mock import patch

import numpy as np


def _make_config():
    with patch("core.config._get_screen_size", return_value=(1920, 1080)):
        from core.config import Config
        return Config()


def test_apply_configured_video_filters_noop_for_non_uvc_ndi():
    from core.screen_capture import apply_configured_video_filters

    config = _make_config()
    config.video_filters = [{"name": "Sharpen", "enabled": True, "options": {"strength": 1.0}}]
    frame = np.full((16, 16, 4), 128, dtype=np.uint8)
    output = apply_configured_video_filters(frame.copy(), config, "mss")
    assert np.array_equal(output, frame)


def test_apply_configured_video_filters_changes_frame_for_uvc():
    from core.screen_capture import apply_configured_video_filters

    config = _make_config()
    config.video_filters = [
        {"name": "Brightness/Contrast", "enabled": True, "options": {"brightness": 20, "contrast": 1.4}},
        {"name": "Sharpen", "enabled": True, "options": {"strength": 0.8}},
    ]

    x = np.linspace(0, 255, 64, dtype=np.uint8)
    y = np.linspace(255, 0, 64, dtype=np.uint8)
    grid = np.outer(x, np.ones_like(y)).astype(np.uint8)
    bgr = np.stack([grid, np.flipud(grid), grid], axis=2)
    alpha = np.full((64, 64, 1), 255, dtype=np.uint8)
    frame = np.concatenate([bgr, alpha], axis=2)

    output = apply_configured_video_filters(frame.copy(), config, "uvc")

    assert output.shape == frame.shape
    assert not np.array_equal(output[:, :, :3], frame[:, :, :3])
    assert np.array_equal(output[:, :, 3], frame[:, :, 3])
