# -*- coding: utf-8 -*-
"""
Unit tests for WebcamCapture class.
Tests cover initialization, camera operations, image processing, and recording.
"""

import pytest
import numpy as np
import cv2
import sys
import time
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch, PropertyMock

sys.path.insert(0, str(Path(__file__).parent.parent))


class TestWebcamCaptureInit:
    """Tests for WebcamCapture.__init__"""

    @pytest.mark.unit
    def test_init_default_parameters(self):
        """Test initialization with default parameters."""
        from webcam_capture_test_opus41 import WebcamCapture

        capture = WebcamCapture()

        assert capture.camera_index == 0
        assert capture.resolution == (1280, 720)
        assert capture.cap is None
        assert capture.is_running is False
        assert capture.frame_count == 0
        assert capture.fps == 0
        assert capture.last_frame is None
        assert capture.recording is False
        assert capture.video_writer is None

    @pytest.mark.unit
    def test_init_custom_parameters(self):
        """Test initialization with custom parameters."""
        from webcam_capture_test_opus41 import WebcamCapture

        capture = WebcamCapture(camera_index=2, resolution=(1920, 1080))

        assert capture.camera_index == 2
        assert capture.resolution == (1920, 1080)

    @pytest.mark.unit
    def test_init_various_resolutions(self):
        """Test initialization with various resolution settings."""
        from webcam_capture_test_opus41 import WebcamCapture

        resolutions = [
            (640, 480),   # VGA
            (1280, 720),  # 720p
            (1920, 1080), # 1080p
            (3840, 2160), # 4K
        ]

        for res in resolutions:
            capture = WebcamCapture(camera_index=0, resolution=res)
            assert capture.resolution == res

    @pytest.mark.unit
    def test_init_fps_tracking_variables(self):
        """Test that FPS tracking variables are initialized."""
        from webcam_capture_test_opus41 import WebcamCapture

        capture = WebcamCapture()

        assert hasattr(capture, 'fps_start_time')
        assert hasattr(capture, 'fps_frame_count')
        assert capture.fps_frame_count == 0


class TestWebcamCaptureInitializeCamera:
    """Tests for WebcamCapture.initialize_camera"""

    @pytest.mark.unit
    def test_initialize_camera_success(self, mock_video_capture, sample_bgr_frame, temp_working_dir):
        """Test successful camera initialization."""
        from webcam_capture_test_opus41 import WebcamCapture

        mock_video_capture.read.return_value = (True, sample_bgr_frame)

        with patch.object(cv2, 'VideoCapture', return_value=mock_video_capture):
            capture = WebcamCapture()
            result = capture.initialize_camera()

        assert result is True
        assert capture.cap is not None
        mock_video_capture.isOpened.assert_called()

    @pytest.mark.unit
    def test_initialize_camera_failure_not_opened(self, mock_video_capture_fail, temp_working_dir):
        """Test camera initialization when camera fails to open."""
        from webcam_capture_test_opus41 import WebcamCapture

        with patch.object(cv2, 'VideoCapture', return_value=mock_video_capture_fail):
            capture = WebcamCapture()
            result = capture.initialize_camera()

        assert result is False

    @pytest.mark.unit
    def test_initialize_camera_failure_no_frame(self, mock_video_capture, temp_working_dir):
        """Test camera initialization when no frame can be read."""
        from webcam_capture_test_opus41 import WebcamCapture

        mock_video_capture.read.return_value = (False, None)

        with patch.object(cv2, 'VideoCapture', return_value=mock_video_capture):
            capture = WebcamCapture()
            result = capture.initialize_camera()

        assert result is False

    @pytest.mark.unit
    def test_initialize_camera_windows_platform(
        self, mock_video_capture, sample_bgr_frame, temp_working_dir, mock_windows_platform
    ):
        """Test camera initialization uses DirectShow on Windows."""
        from webcam_capture_test_opus41 import WebcamCapture

        mock_video_capture.read.return_value = (True, sample_bgr_frame)

        with patch.object(cv2, 'VideoCapture', return_value=mock_video_capture) as mock_vc:
            capture = WebcamCapture()
            capture.initialize_camera()

            # Check that DirectShow was used
            mock_vc.assert_called_once_with(0, cv2.CAP_DSHOW)

    @pytest.mark.unit
    def test_initialize_camera_linux_platform(
        self, mock_video_capture, sample_bgr_frame, temp_working_dir, mock_linux_platform
    ):
        """Test camera initialization uses default API on Linux."""
        from webcam_capture_test_opus41 import WebcamCapture

        mock_video_capture.read.return_value = (True, sample_bgr_frame)

        with patch.object(cv2, 'VideoCapture', return_value=mock_video_capture) as mock_vc:
            capture = WebcamCapture()
            capture.initialize_camera()

            # Check that default API was used (no CAP_DSHOW)
            mock_vc.assert_called_once_with(0)

    @pytest.mark.unit
    def test_initialize_camera_sets_properties(self, mock_video_capture, sample_bgr_frame, temp_working_dir):
        """Test that camera properties are set correctly."""
        from webcam_capture_test_opus41 import WebcamCapture

        mock_video_capture.read.return_value = (True, sample_bgr_frame)

        with patch.object(cv2, 'VideoCapture', return_value=mock_video_capture):
            capture = WebcamCapture(resolution=(1920, 1080))
            capture.initialize_camera()

        # Verify set was called with correct parameters
        calls = mock_video_capture.set.call_args_list
        # Check that set was called with frame width and height properties
        call_args = [call[0] for call in calls]  # Get positional args
        assert any(cv2.CAP_PROP_FRAME_WIDTH == args[0] for args in call_args)
        assert any(cv2.CAP_PROP_FRAME_HEIGHT == args[0] for args in call_args)

    @pytest.mark.unit
    def test_initialize_camera_exception_handling(self, temp_working_dir):
        """Test exception handling during camera initialization."""
        from webcam_capture_test_opus41 import WebcamCapture

        with patch.object(cv2, 'VideoCapture', side_effect=Exception("Camera error")):
            capture = WebcamCapture()
            result = capture.initialize_camera()

        assert result is False


class TestWebcamCaptureCalculateFPS:
    """Tests for WebcamCapture.calculate_fps"""

    @pytest.mark.unit
    def test_calculate_fps_increments_counter(self):
        """Test that FPS calculation increments frame counter."""
        from webcam_capture_test_opus41 import WebcamCapture

        capture = WebcamCapture()
        initial_count = capture.fps_frame_count

        capture.calculate_fps()

        assert capture.fps_frame_count == initial_count + 1

    @pytest.mark.unit
    def test_calculate_fps_resets_after_30_frames(self):
        """Test that FPS is calculated and counter resets after 30 frames."""
        from webcam_capture_test_opus41 import WebcamCapture

        capture = WebcamCapture()
        capture.fps_frame_count = 29  # One less than threshold
        capture.fps_start_time = time.time() - 1.0  # 1 second ago

        capture.calculate_fps()

        # After 30 frames, counter should reset to 0
        assert capture.fps_frame_count == 0
        assert capture.fps > 0  # FPS should be calculated

    @pytest.mark.unit
    def test_calculate_fps_accuracy(self):
        """Test FPS calculation accuracy."""
        from webcam_capture_test_opus41 import WebcamCapture

        capture = WebcamCapture()
        capture.fps_frame_count = 29
        capture.fps_start_time = time.time() - 1.0  # Exactly 1 second ago

        capture.calculate_fps()

        # 30 frames in 1 second = ~30 FPS
        assert 25 < capture.fps < 35  # Allow some tolerance

    @pytest.mark.unit
    def test_calculate_fps_handles_zero_elapsed_time(self):
        """Test FPS calculation handles zero elapsed time."""
        from webcam_capture_test_opus41 import WebcamCapture

        capture = WebcamCapture()
        capture.fps_frame_count = 29
        capture.fps_start_time = time.time()  # Current time (0 elapsed)

        # Should not raise exception
        capture.calculate_fps()


class TestWebcamCapturePreprocessFrame:
    """Tests for WebcamCapture.preprocess_frame"""

    @pytest.mark.unit
    def test_preprocess_frame_returns_dict(self, sample_bgr_frame):
        """Test that preprocess_frame returns a dictionary."""
        from webcam_capture_test_opus41 import WebcamCapture

        capture = WebcamCapture()
        result = capture.preprocess_frame(sample_bgr_frame)

        assert isinstance(result, dict)

    @pytest.mark.unit
    def test_preprocess_frame_contains_gray(self, sample_bgr_frame):
        """Test grayscale conversion."""
        from webcam_capture_test_opus41 import WebcamCapture

        capture = WebcamCapture()
        result = capture.preprocess_frame(sample_bgr_frame)

        assert 'gray' in result
        assert result['gray'].ndim == 2  # Should be 2D (grayscale)
        assert result['gray'].shape[:2] == sample_bgr_frame.shape[:2]

    @pytest.mark.unit
    def test_preprocess_frame_contains_blur(self, sample_bgr_frame):
        """Test Gaussian blur."""
        from webcam_capture_test_opus41 import WebcamCapture

        capture = WebcamCapture()
        result = capture.preprocess_frame(sample_bgr_frame)

        assert 'blur' in result
        assert result['blur'].shape == sample_bgr_frame.shape

    @pytest.mark.unit
    def test_preprocess_frame_contains_edges(self, sample_bgr_frame):
        """Test Canny edge detection."""
        from webcam_capture_test_opus41 import WebcamCapture

        capture = WebcamCapture()
        result = capture.preprocess_frame(sample_bgr_frame)

        assert 'edges' in result
        assert result['edges'].ndim == 2  # Should be 2D

    @pytest.mark.unit
    def test_preprocess_frame_contains_binary(self, sample_bgr_frame):
        """Test binary thresholding."""
        from webcam_capture_test_opus41 import WebcamCapture

        capture = WebcamCapture()
        result = capture.preprocess_frame(sample_bgr_frame)

        assert 'binary' in result
        # Binary image should only have 0 and 255
        unique_vals = np.unique(result['binary'])
        assert all(v in [0, 255] for v in unique_vals)

    @pytest.mark.unit
    def test_preprocess_frame_contains_adaptive(self, sample_bgr_frame):
        """Test adaptive threshold."""
        from webcam_capture_test_opus41 import WebcamCapture

        capture = WebcamCapture()
        result = capture.preprocess_frame(sample_bgr_frame)

        assert 'adaptive' in result
        assert result['adaptive'].ndim == 2

    @pytest.mark.unit
    def test_preprocess_frame_contains_morphology(self, sample_bgr_frame):
        """Test morphological operations."""
        from webcam_capture_test_opus41 import WebcamCapture

        capture = WebcamCapture()
        result = capture.preprocess_frame(sample_bgr_frame)

        assert 'morphology' in result
        assert result['morphology'].ndim == 2

    @pytest.mark.unit
    def test_preprocess_frame_with_noisy_image(self, noisy_frame):
        """Test preprocessing with noisy image."""
        from webcam_capture_test_opus41 import WebcamCapture

        capture = WebcamCapture()
        result = capture.preprocess_frame(noisy_frame)

        # Should still return all expected keys
        expected_keys = ['gray', 'blur', 'edges', 'binary', 'adaptive', 'morphology']
        for key in expected_keys:
            assert key in result

    @pytest.mark.unit
    def test_preprocess_frame_exception_handling(self):
        """Test exception handling with invalid input."""
        from webcam_capture_test_opus41 import WebcamCapture

        capture = WebcamCapture()

        # Invalid frame (None) - function handles exception internally
        # and returns empty or partial dict
        result = capture.preprocess_frame(None)
        # Should return dict (possibly empty due to exception handling)
        assert isinstance(result, dict)


class TestWebcamCaptureDetectObjects:
    """Tests for WebcamCapture.detect_objects"""

    @pytest.mark.unit
    def test_detect_objects_returns_tuple(self, frame_with_objects):
        """Test that detect_objects returns a tuple."""
        from webcam_capture_test_opus41 import WebcamCapture

        capture = WebcamCapture()
        result = capture.detect_objects(frame_with_objects)

        assert isinstance(result, tuple)
        assert len(result) == 2

    @pytest.mark.unit
    def test_detect_objects_returns_frame_and_list(self, frame_with_objects):
        """Test return types of detect_objects."""
        from webcam_capture_test_opus41 import WebcamCapture

        capture = WebcamCapture()
        frame_with_detections, objects = capture.detect_objects(frame_with_objects)

        assert isinstance(frame_with_detections, np.ndarray)
        assert isinstance(objects, list)

    @pytest.mark.unit
    def test_detect_objects_finds_objects(self, frame_with_objects):
        """Test that objects are detected in frame."""
        from webcam_capture_test_opus41 import WebcamCapture

        capture = WebcamCapture()
        _, objects = capture.detect_objects(frame_with_objects)

        # Should detect at least one object
        assert len(objects) >= 1

    @pytest.mark.unit
    def test_detect_objects_object_structure(self, frame_with_objects):
        """Test structure of detected object info."""
        from webcam_capture_test_opus41 import WebcamCapture

        capture = WebcamCapture()
        _, objects = capture.detect_objects(frame_with_objects)

        if objects:  # If any objects detected
            obj = objects[0]
            assert 'id' in obj
            assert 'bbox' in obj
            assert 'center' in obj
            assert 'area' in obj
            assert 'contour' in obj

    @pytest.mark.unit
    def test_detect_objects_bbox_format(self, frame_with_objects):
        """Test bounding box format (x, y, w, h)."""
        from webcam_capture_test_opus41 import WebcamCapture

        capture = WebcamCapture()
        _, objects = capture.detect_objects(frame_with_objects)

        if objects:
            bbox = objects[0]['bbox']
            assert len(bbox) == 4  # x, y, w, h
            x, y, w, h = bbox
            assert all(isinstance(v, (int, np.integer)) for v in bbox)
            assert w > 0 and h > 0

    @pytest.mark.unit
    def test_detect_objects_center_format(self, frame_with_objects):
        """Test center point format (cx, cy)."""
        from webcam_capture_test_opus41 import WebcamCapture

        capture = WebcamCapture()
        _, objects = capture.detect_objects(frame_with_objects)

        if objects:
            center = objects[0]['center']
            assert len(center) == 2  # cx, cy

    @pytest.mark.unit
    def test_detect_objects_area_threshold(self, sample_small_frame):
        """Test that small objects (area < 500) are filtered."""
        from webcam_capture_test_opus41 import WebcamCapture

        # Create frame with very small object
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.rectangle(frame, (100, 100), (110, 110), (255, 255, 255), -1)  # 10x10 = 100 area

        capture = WebcamCapture()
        _, objects = capture.detect_objects(frame)

        # Small objects should be filtered out
        for obj in objects:
            assert obj['area'] >= 500

    @pytest.mark.unit
    def test_detect_objects_empty_frame(self):
        """Test detection on empty frame."""
        from webcam_capture_test_opus41 import WebcamCapture

        frame = np.zeros((480, 640, 3), dtype=np.uint8)

        capture = WebcamCapture()
        _, objects = capture.detect_objects(frame)

        assert objects == []

    @pytest.mark.unit
    def test_detect_objects_preserves_original_shape(self, frame_with_objects):
        """Test that output frame has same shape as input."""
        from webcam_capture_test_opus41 import WebcamCapture

        capture = WebcamCapture()
        frame_result, _ = capture.detect_objects(frame_with_objects)

        assert frame_result.shape == frame_with_objects.shape


class TestWebcamCaptureAddOverlayInfo:
    """Tests for WebcamCapture.add_overlay_info"""

    @pytest.mark.unit
    def test_add_overlay_returns_frame(self, sample_bgr_frame):
        """Test that add_overlay_info returns a frame."""
        from webcam_capture_test_opus41 import WebcamCapture

        capture = WebcamCapture()
        result = capture.add_overlay_info(sample_bgr_frame.copy())

        assert isinstance(result, np.ndarray)
        assert result.shape == sample_bgr_frame.shape

    @pytest.mark.unit
    def test_add_overlay_modifies_frame(self, sample_bgr_frame):
        """Test that overlay modifies the frame."""
        from webcam_capture_test_opus41 import WebcamCapture

        capture = WebcamCapture()
        capture.fps = 30.0
        capture.frame_count = 100

        original = sample_bgr_frame.copy()
        result = capture.add_overlay_info(sample_bgr_frame.copy())

        # Frames should not be identical (overlay was added)
        assert not np.array_equal(original, result)

    @pytest.mark.unit
    def test_add_overlay_recording_indicator(self, sample_bgr_frame):
        """Test recording indicator when recording is active."""
        from webcam_capture_test_opus41 import WebcamCapture

        capture = WebcamCapture()
        capture.recording = True

        result = capture.add_overlay_info(sample_bgr_frame.copy())

        # Should return modified frame without error
        assert result is not None

    @pytest.mark.unit
    def test_add_overlay_with_various_fps(self, sample_bgr_frame):
        """Test overlay with various FPS values."""
        from webcam_capture_test_opus41 import WebcamCapture

        capture = WebcamCapture()

        for fps in [0, 15, 30, 60, 120]:
            capture.fps = fps
            result = capture.add_overlay_info(sample_bgr_frame.copy())
            assert result is not None


class TestWebcamCaptureCaptureImage:
    """Tests for WebcamCapture.capture_image"""

    @pytest.mark.unit
    def test_capture_image_no_frame(self):
        """Test capture_image when no frame available."""
        from webcam_capture_test_opus41 import WebcamCapture

        capture = WebcamCapture()
        capture.last_frame = None

        result = capture.capture_image()

        assert result is None

    @pytest.mark.unit
    def test_capture_image_success(self, sample_bgr_frame, temp_working_dir, mock_cv2_imwrite):
        """Test successful image capture."""
        from webcam_capture_test_opus41 import WebcamCapture

        capture = WebcamCapture()
        capture.last_frame = sample_bgr_frame.copy()

        result = capture.capture_image()

        assert result is not None
        assert 'capture_' in result
        assert result.endswith('.jpg')

    @pytest.mark.unit
    def test_capture_image_file_naming(self, sample_bgr_frame, temp_working_dir, mock_cv2_imwrite):
        """Test that captured image has correct naming format."""
        from webcam_capture_test_opus41 import WebcamCapture
        import re

        capture = WebcamCapture()
        capture.last_frame = sample_bgr_frame.copy()

        result = capture.capture_image()

        # Should match pattern: capture_YYYYMMDD_HHMMSS.jpg
        pattern = r'capture_\d{8}_\d{6}\.jpg'
        assert re.search(pattern, result)

    @pytest.mark.unit
    def test_capture_image_failure(self, sample_bgr_frame, temp_working_dir):
        """Test capture_image when imwrite fails."""
        from webcam_capture_test_opus41 import WebcamCapture

        capture = WebcamCapture()
        capture.last_frame = sample_bgr_frame.copy()

        with patch.object(cv2, 'imwrite', return_value=False):
            result = capture.capture_image()

        assert result is None


class TestWebcamCaptureRecording:
    """Tests for WebcamCapture.start_recording and stop_recording"""

    @pytest.mark.unit
    def test_start_recording_when_already_recording(self, webcam_capture_with_mock_camera):
        """Test start_recording when already recording."""
        capture = webcam_capture_with_mock_camera
        capture.recording = True

        result = capture.start_recording()

        assert result is False

    @pytest.mark.unit
    def test_start_recording_success(self, webcam_capture_with_mock_camera, mock_video_writer):
        """Test successful start recording."""
        capture = webcam_capture_with_mock_camera
        capture.recording = False

        with patch.object(cv2, 'VideoWriter', return_value=mock_video_writer):
            with patch.object(cv2, 'VideoWriter_fourcc', return_value=1):
                result = capture.start_recording()

        assert result is True
        assert capture.recording is True

    @pytest.mark.unit
    def test_start_recording_failure(self, webcam_capture_with_mock_camera, mock_video_writer_fail):
        """Test start_recording when VideoWriter fails."""
        capture = webcam_capture_with_mock_camera
        capture.recording = False

        with patch.object(cv2, 'VideoWriter', return_value=mock_video_writer_fail):
            with patch.object(cv2, 'VideoWriter_fourcc', return_value=1):
                result = capture.start_recording()

        assert result is False
        assert capture.recording is False

    @pytest.mark.unit
    def test_stop_recording_when_not_recording(self):
        """Test stop_recording when not recording."""
        from webcam_capture_test_opus41 import WebcamCapture

        capture = WebcamCapture()
        capture.recording = False

        # Should not raise exception
        capture.stop_recording()

    @pytest.mark.unit
    def test_stop_recording_success(self, mock_video_writer):
        """Test successful stop recording."""
        from webcam_capture_test_opus41 import WebcamCapture

        capture = WebcamCapture()
        capture.recording = True
        capture.video_writer = mock_video_writer

        capture.stop_recording()

        assert capture.recording is False
        assert capture.video_writer is None
        mock_video_writer.release.assert_called_once()


class TestWebcamCaptureCleanup:
    """Tests for WebcamCapture.cleanup"""

    @pytest.mark.unit
    def test_cleanup_releases_camera(self, mock_video_capture, mock_cv2_destroyallwindows):
        """Test that cleanup releases camera resources."""
        from webcam_capture_test_opus41 import WebcamCapture

        capture = WebcamCapture()
        capture.cap = mock_video_capture
        capture.recording = False

        capture.cleanup()

        mock_video_capture.release.assert_called_once()

    @pytest.mark.unit
    def test_cleanup_stops_recording(self, mock_video_capture, mock_video_writer, mock_cv2_destroyallwindows):
        """Test that cleanup stops recording if active."""
        from webcam_capture_test_opus41 import WebcamCapture

        capture = WebcamCapture()
        capture.cap = mock_video_capture
        capture.recording = True
        capture.video_writer = mock_video_writer

        capture.cleanup()

        assert capture.recording is False
        mock_video_writer.release.assert_called_once()

    @pytest.mark.unit
    def test_cleanup_closes_windows(self, mock_video_capture, mock_cv2_destroyallwindows):
        """Test that cleanup closes all windows."""
        from webcam_capture_test_opus41 import WebcamCapture

        capture = WebcamCapture()
        capture.cap = mock_video_capture
        capture.recording = False

        capture.cleanup()

        mock_cv2_destroyallwindows.assert_called_once()

    @pytest.mark.unit
    def test_cleanup_handles_none_cap(self, mock_cv2_destroyallwindows):
        """Test cleanup when cap is None."""
        from webcam_capture_test_opus41 import WebcamCapture

        capture = WebcamCapture()
        capture.cap = None
        capture.recording = False

        # Should not raise exception
        capture.cleanup()


class TestWebcamCaptureRun:
    """Tests for WebcamCapture.run (main loop)"""

    @pytest.mark.unit
    def test_run_fails_without_camera(self, temp_working_dir):
        """Test run exits early when camera initialization fails."""
        from webcam_capture_test_opus41 import WebcamCapture

        with patch.object(cv2, 'VideoCapture') as mock_vc:
            mock_cap = MagicMock()
            mock_cap.isOpened.return_value = False
            mock_vc.return_value = mock_cap

            capture = WebcamCapture()
            capture.run()

        # Should exit without setting is_running
        assert capture.is_running is False

    @pytest.mark.unit
    def test_run_processes_quit_key(
        self, webcam_capture_with_mock_camera, mock_all_cv2_gui
    ):
        """Test that run loop exits on 'q' key."""
        capture = webcam_capture_with_mock_camera

        # Simulate 'q' key press
        mock_all_cv2_gui['waitKey'].return_value = ord('q')

        # Mock initialize_camera to return True
        with patch.object(capture, 'initialize_camera', return_value=True):
            with patch.object(capture, 'cleanup'):
                capture.run()

        assert capture.is_running is False

    @pytest.mark.unit
    def test_run_processes_escape_key(
        self, webcam_capture_with_mock_camera, mock_all_cv2_gui
    ):
        """Test that run loop exits on ESC key."""
        capture = webcam_capture_with_mock_camera

        # Simulate ESC key press (27)
        mock_all_cv2_gui['waitKey'].return_value = 27

        with patch.object(capture, 'initialize_camera', return_value=True):
            with patch.object(capture, 'cleanup'):
                capture.run()

        assert capture.is_running is False
