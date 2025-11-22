# -*- coding: utf-8 -*-
"""
Pytest configuration and fixtures for webcam_capture_test_opus41 tests.
"""

import pytest
import numpy as np
import cv2
import sys
import os
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))


# ========================== Sample Image Fixtures ==========================

@pytest.fixture
def sample_bgr_frame():
    """Create a sample BGR frame (720p) for testing."""
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    # Add some color regions for testing
    frame[0:360, 0:640] = [255, 0, 0]      # Blue (top-left)
    frame[0:360, 640:1280] = [0, 255, 0]   # Green (top-right)
    frame[360:720, 0:640] = [0, 0, 255]    # Red (bottom-left)
    frame[360:720, 640:1280] = [255, 255, 255]  # White (bottom-right)
    return frame


@pytest.fixture
def sample_small_frame():
    """Create a small sample frame (100x100) for faster testing."""
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    frame[:50, :50] = [255, 0, 0]      # Blue
    frame[:50, 50:] = [0, 255, 0]      # Green
    frame[50:, :50] = [0, 0, 255]      # Red
    frame[50:, 50:] = [255, 255, 255]  # White
    return frame


@pytest.fixture
def sample_gray_frame():
    """Create a grayscale frame for testing."""
    return np.random.randint(0, 256, (480, 640), dtype=np.uint8)


@pytest.fixture
def frame_with_objects():
    """Create a frame with clear objects (white rectangles on black)."""
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    # Add white rectangles as detectable objects
    cv2.rectangle(frame, (50, 50), (150, 150), (255, 255, 255), -1)
    cv2.rectangle(frame, (200, 100), (350, 250), (255, 255, 255), -1)
    cv2.rectangle(frame, (400, 200), (550, 400), (255, 255, 255), -1)
    return frame


@pytest.fixture
def frame_with_edges():
    """Create a frame with clear edges for edge detection testing."""
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    # Draw some lines and shapes
    cv2.rectangle(frame, (100, 100), (300, 300), (255, 255, 255), 3)
    cv2.circle(frame, (450, 240), 80, (255, 255, 255), 3)
    cv2.line(frame, (0, 400), (640, 400), (255, 255, 255), 2)
    return frame


@pytest.fixture
def noisy_frame():
    """Create a frame with noise for testing noise reduction."""
    frame = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
    return frame


# ========================== Mock Fixtures ==========================

@pytest.fixture
def mock_video_capture():
    """Create a mock VideoCapture object."""
    mock_cap = MagicMock(spec=cv2.VideoCapture)
    mock_cap.isOpened.return_value = True
    mock_cap.get.side_effect = lambda prop: {
        cv2.CAP_PROP_FRAME_WIDTH: 1280,
        cv2.CAP_PROP_FRAME_HEIGHT: 720,
        cv2.CAP_PROP_FPS: 30,
    }.get(prop, 0)

    # Create a sample frame for read()
    sample_frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    mock_cap.read.return_value = (True, sample_frame)

    return mock_cap


@pytest.fixture
def mock_video_capture_fail():
    """Create a mock VideoCapture that fails to open."""
    mock_cap = MagicMock(spec=cv2.VideoCapture)
    mock_cap.isOpened.return_value = False
    mock_cap.read.return_value = (False, None)
    return mock_cap


@pytest.fixture
def mock_video_writer():
    """Create a mock VideoWriter object."""
    mock_writer = MagicMock(spec=cv2.VideoWriter)
    mock_writer.isOpened.return_value = True
    return mock_writer


@pytest.fixture
def mock_video_writer_fail():
    """Create a mock VideoWriter that fails to open."""
    mock_writer = MagicMock(spec=cv2.VideoWriter)
    mock_writer.isOpened.return_value = False
    return mock_writer


# ========================== Directory Fixtures ==========================

@pytest.fixture
def temp_output_dir(tmp_path):
    """Create a temporary output directory structure."""
    output_dir = tmp_path / "output"
    capture_dir = output_dir / "captures"
    video_dir = output_dir / "videos"

    output_dir.mkdir(parents=True, exist_ok=True)
    capture_dir.mkdir(parents=True, exist_ok=True)
    video_dir.mkdir(parents=True, exist_ok=True)

    return {
        'output': output_dir,
        'captures': capture_dir,
        'videos': video_dir,
        'root': tmp_path
    }


@pytest.fixture
def temp_working_dir(tmp_path, monkeypatch):
    """Change to a temporary working directory for tests."""
    monkeypatch.chdir(tmp_path)

    # Create output directories
    output_dir = tmp_path / "output"
    capture_dir = output_dir / "captures"
    video_dir = output_dir / "videos"

    output_dir.mkdir(parents=True, exist_ok=True)
    capture_dir.mkdir(parents=True, exist_ok=True)
    video_dir.mkdir(parents=True, exist_ok=True)

    return tmp_path


# ========================== WebcamCapture Fixtures ==========================

@pytest.fixture
def webcam_capture_instance():
    """Create a WebcamCapture instance without initializing camera."""
    from webcam_capture_test_opus41 import WebcamCapture

    with patch.object(cv2, 'VideoCapture'):
        capture = WebcamCapture(camera_index=0, resolution=(1280, 720))

    return capture


@pytest.fixture
def webcam_capture_with_mock_camera(mock_video_capture, sample_bgr_frame, temp_working_dir):
    """Create a WebcamCapture instance with mocked camera."""
    from webcam_capture_test_opus41 import WebcamCapture

    mock_video_capture.read.return_value = (True, sample_bgr_frame.copy())

    with patch.object(cv2, 'VideoCapture', return_value=mock_video_capture):
        capture = WebcamCapture(camera_index=0, resolution=(1280, 720))
        capture.cap = mock_video_capture
        capture.last_frame = sample_bgr_frame.copy()

    return capture


@pytest.fixture
def webcam_capture_initialized(mock_video_capture, sample_bgr_frame, temp_working_dir):
    """Create a fully initialized WebcamCapture instance."""
    from webcam_capture_test_opus41 import WebcamCapture

    mock_video_capture.read.return_value = (True, sample_bgr_frame.copy())

    with patch.object(cv2, 'VideoCapture', return_value=mock_video_capture):
        capture = WebcamCapture(camera_index=0, resolution=(1280, 720))
        capture.cap = mock_video_capture
        capture.last_frame = sample_bgr_frame.copy()
        capture.is_running = True
        capture.frame_count = 100
        capture.fps = 30.0

    return capture


# ========================== Logging Fixtures ==========================

@pytest.fixture
def capture_logs(caplog):
    """Capture log messages during tests."""
    import logging
    caplog.set_level(logging.DEBUG)
    return caplog


@pytest.fixture
def suppress_logging():
    """Suppress logging output during tests."""
    import logging
    logging.disable(logging.CRITICAL)
    yield
    logging.disable(logging.NOTSET)


# ========================== Platform Fixtures ==========================

@pytest.fixture
def mock_windows_platform(monkeypatch):
    """Mock Windows platform."""
    monkeypatch.setattr(sys, 'platform', 'win32')


@pytest.fixture
def mock_linux_platform(monkeypatch):
    """Mock Linux platform."""
    monkeypatch.setattr(sys, 'platform', 'linux')


@pytest.fixture
def mock_macos_platform(monkeypatch):
    """Mock macOS platform."""
    monkeypatch.setattr(sys, 'platform', 'darwin')


# ========================== OpenCV Mock Fixtures ==========================

@pytest.fixture
def mock_cv2_imwrite(mocker):
    """Mock cv2.imwrite to avoid actual file operations."""
    return mocker.patch.object(cv2, 'imwrite', return_value=True)


@pytest.fixture
def mock_cv2_imshow(mocker):
    """Mock cv2.imshow to avoid GUI operations."""
    return mocker.patch.object(cv2, 'imshow')


@pytest.fixture
def mock_cv2_waitkey(mocker):
    """Mock cv2.waitKey to control test flow."""
    return mocker.patch.object(cv2, 'waitKey', return_value=ord('q'))


@pytest.fixture
def mock_cv2_destroyallwindows(mocker):
    """Mock cv2.destroyAllWindows to avoid GUI cleanup."""
    return mocker.patch.object(cv2, 'destroyAllWindows')


@pytest.fixture
def mock_cv2_namedwindow(mocker):
    """Mock cv2.namedWindow to avoid GUI window creation."""
    return mocker.patch.object(cv2, 'namedWindow')


@pytest.fixture
def mock_cv2_resizewindow(mocker):
    """Mock cv2.resizeWindow to avoid GUI operations."""
    return mocker.patch.object(cv2, 'resizeWindow')


@pytest.fixture
def mock_all_cv2_gui(mocker):
    """Mock all OpenCV GUI operations."""
    return {
        'imshow': mocker.patch.object(cv2, 'imshow'),
        'waitKey': mocker.patch.object(cv2, 'waitKey', return_value=ord('q')),
        'destroyAllWindows': mocker.patch.object(cv2, 'destroyAllWindows'),
        'namedWindow': mocker.patch.object(cv2, 'namedWindow'),
        'resizeWindow': mocker.patch.object(cv2, 'resizeWindow'),
    }


# ========================== CUDA Mock Fixtures ==========================

@pytest.fixture
def mock_cuda_available(mocker):
    """Mock CUDA as available."""
    mock_cuda = MagicMock()
    mock_cuda.getCudaEnabledDeviceCount.return_value = 1
    mock_cuda.setDevice = MagicMock()
    mock_cuda.getDevice.return_value = 0
    mocker.patch.object(cv2, 'cuda', mock_cuda)
    return mock_cuda


@pytest.fixture
def mock_cuda_unavailable(mocker):
    """Mock CUDA as unavailable."""
    mock_cuda = MagicMock()
    mock_cuda.getCudaEnabledDeviceCount.return_value = 0
    mocker.patch.object(cv2, 'cuda', mock_cuda)
    return mock_cuda


# ========================== Helper Functions ==========================

def create_contour_frame(width=640, height=480, num_objects=3):
    """Create a frame with multiple objects for contour detection testing."""
    frame = np.zeros((height, width, 3), dtype=np.uint8)

    for i in range(num_objects):
        x = (i + 1) * width // (num_objects + 1) - 40
        y = height // 2 - 40
        cv2.rectangle(frame, (x, y), (x + 80, y + 80), (255, 255, 255), -1)

    return frame


def create_test_video_frame(frame_number, width=1280, height=720):
    """Create a unique test frame for video testing."""
    frame = np.zeros((height, width, 3), dtype=np.uint8)

    # Add frame number as text
    cv2.putText(frame, f"Frame {frame_number}", (50, height // 2),
                cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)

    # Add varying color based on frame number
    color = (frame_number % 256, (frame_number * 2) % 256, (frame_number * 3) % 256)
    cv2.rectangle(frame, (100, 100), (200, 200), color, -1)

    return frame
