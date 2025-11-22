# -*- coding: utf-8 -*-
"""
Unit tests for helper functions.
Tests cover camera availability testing, OpenCV verification, and main function.
"""

import pytest
import numpy as np
import cv2
import sys
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch, call

sys.path.insert(0, str(Path(__file__).parent.parent))


class TestCameraAvailability:
    """Tests for test_camera_availability function"""

    @pytest.mark.unit
    def test_returns_list(self):
        """Test that function returns a list."""
        from webcam_capture_test_opus41 import test_camera_availability

        with patch.object(cv2, 'VideoCapture') as mock_vc:
            mock_cap = MagicMock()
            mock_cap.isOpened.return_value = False
            mock_vc.return_value = mock_cap

            result = test_camera_availability()

        assert isinstance(result, list)

    @pytest.mark.unit
    def test_finds_available_camera(self, sample_bgr_frame):
        """Test detection of available camera."""
        from webcam_capture_test_opus41 import test_camera_availability

        with patch.object(cv2, 'VideoCapture') as mock_vc:
            mock_cap = MagicMock()
            mock_cap.isOpened.return_value = True
            mock_cap.read.return_value = (True, sample_bgr_frame)
            mock_cap.get.return_value = 30  # Default return for get calls
            mock_vc.return_value = mock_cap

            result = test_camera_availability()

        assert len(result) > 0
        assert 0 in result  # Camera index 0 should be detected

    @pytest.mark.unit
    def test_no_cameras_available(self):
        """Test when no cameras are available."""
        from webcam_capture_test_opus41 import test_camera_availability

        with patch.object(cv2, 'VideoCapture') as mock_vc:
            mock_cap = MagicMock()
            mock_cap.isOpened.return_value = False
            mock_vc.return_value = mock_cap

            result = test_camera_availability()

        assert result == []

    @pytest.mark.unit
    def test_multiple_cameras(self, sample_bgr_frame):
        """Test detection of multiple cameras."""
        from webcam_capture_test_opus41 import test_camera_availability

        def create_mock_cap(index):
            mock_cap = MagicMock()
            # Only cameras 0 and 2 are "available"
            if index in [0, 2]:
                mock_cap.isOpened.return_value = True
                mock_cap.read.return_value = (True, sample_bgr_frame)
                mock_cap.get.return_value = 30
            else:
                mock_cap.isOpened.return_value = False
            return mock_cap

        with patch.object(cv2, 'VideoCapture', side_effect=create_mock_cap):
            result = test_camera_availability()

        assert 0 in result
        assert 2 in result

    @pytest.mark.unit
    def test_camera_opens_but_no_frame(self):
        """Test when camera opens but cannot read frame."""
        from webcam_capture_test_opus41 import test_camera_availability

        with patch.object(cv2, 'VideoCapture') as mock_vc:
            mock_cap = MagicMock()
            mock_cap.isOpened.return_value = True
            mock_cap.read.return_value = (False, None)  # Can't read frame
            mock_vc.return_value = mock_cap

            result = test_camera_availability()

        assert result == []

    @pytest.mark.unit
    def test_releases_cameras_after_test(self, sample_bgr_frame):
        """Test that cameras are released after testing."""
        from webcam_capture_test_opus41 import test_camera_availability

        with patch.object(cv2, 'VideoCapture') as mock_vc:
            mock_cap = MagicMock()
            mock_cap.isOpened.return_value = True
            mock_cap.read.return_value = (True, sample_bgr_frame)
            mock_cap.get.return_value = 30
            mock_vc.return_value = mock_cap

            test_camera_availability()

        # release should be called for each opened camera
        mock_cap.release.assert_called()

    @pytest.mark.unit
    def test_handles_exception(self):
        """Test exception handling during camera testing."""
        from webcam_capture_test_opus41 import test_camera_availability

        with patch.object(cv2, 'VideoCapture', side_effect=Exception("Camera error")):
            result = test_camera_availability()

        # Should return empty list, not raise exception
        assert result == []

    @pytest.mark.unit
    def test_scans_ten_indices(self, sample_bgr_frame):
        """Test that function scans indices 0-9."""
        from webcam_capture_test_opus41 import test_camera_availability

        call_indices = []

        def track_calls(index):
            call_indices.append(index)
            mock_cap = MagicMock()
            mock_cap.isOpened.return_value = False
            return mock_cap

        with patch.object(cv2, 'VideoCapture', side_effect=track_calls):
            test_camera_availability()

        # Should have tried indices 0-9
        assert call_indices == list(range(10))


class TestVerifyOpenCVInstallation:
    """Tests for verify_opencv_installation function"""

    @pytest.mark.unit
    def test_returns_bool(self, mock_cuda_unavailable):
        """Test that function returns a boolean."""
        from webcam_capture_test_opus41 import verify_opencv_installation

        result = verify_opencv_installation()

        assert isinstance(result, bool)

    @pytest.mark.unit
    def test_success_without_cuda(self, mock_cuda_unavailable):
        """Test successful verification without CUDA."""
        from webcam_capture_test_opus41 import verify_opencv_installation

        result = verify_opencv_installation()

        assert result is True

    @pytest.mark.unit
    def test_success_with_cuda(self, mock_cuda_available):
        """Test successful verification with CUDA."""
        from webcam_capture_test_opus41 import verify_opencv_installation

        result = verify_opencv_installation()

        assert result is True

    @pytest.mark.unit
    def test_checks_opencv_version(self, mock_cuda_unavailable):
        """Test that OpenCV version is checked."""
        from webcam_capture_test_opus41 import verify_opencv_installation

        # Should not raise exception
        result = verify_opencv_installation()
        assert result is True

    @pytest.mark.unit
    def test_tests_basic_operations(self, mock_cuda_unavailable):
        """Test that basic OpenCV operations are tested."""
        from webcam_capture_test_opus41 import verify_opencv_installation

        # This should test grayscale conversion and edge detection
        result = verify_opencv_installation()
        assert result is True

    @pytest.mark.unit
    def test_checks_video_codecs(self, mock_cuda_unavailable):
        """Test that video codecs are checked."""
        from webcam_capture_test_opus41 import verify_opencv_installation

        with patch.object(cv2, 'VideoWriter_fourcc', return_value=1) as mock_fourcc:
            result = verify_opencv_installation()

        # Should have tested codecs
        assert mock_fourcc.call_count > 0

    @pytest.mark.unit
    def test_handles_exception(self):
        """Test exception handling."""
        from webcam_capture_test_opus41 import verify_opencv_installation

        with patch.object(cv2, 'cvtColor', side_effect=Exception("OpenCV error")):
            result = verify_opencv_installation()

        assert result is False

    @pytest.mark.unit
    def test_cuda_device_enumeration(self, mock_cuda_available):
        """Test CUDA device enumeration when available."""
        from webcam_capture_test_opus41 import verify_opencv_installation

        result = verify_opencv_installation()

        mock_cuda_available.getCudaEnabledDeviceCount.assert_called()


class TestMainFunction:
    """Tests for main function"""

    @pytest.mark.unit
    def test_main_exits_on_opencv_failure(self, temp_working_dir):
        """Test that main exits when OpenCV verification fails."""
        from webcam_capture_test_opus41 import main

        with patch('webcam_capture_test_opus41.verify_opencv_installation', return_value=False):
            with pytest.raises(SystemExit) as exc_info:
                main()

        assert exc_info.value.code == 1

    @pytest.mark.unit
    def test_main_exits_on_no_cameras(self, temp_working_dir, mock_cuda_unavailable):
        """Test that main exits when no cameras found."""
        from webcam_capture_test_opus41 import main

        with patch('webcam_capture_test_opus41.verify_opencv_installation', return_value=True):
            with patch('webcam_capture_test_opus41.test_camera_availability', return_value=[]):
                with pytest.raises(SystemExit) as exc_info:
                    main()

        assert exc_info.value.code == 1

    @pytest.mark.unit
    def test_main_creates_webcam_capture(self, temp_working_dir, mock_cuda_unavailable):
        """Test that main creates WebcamCapture instance."""
        from webcam_capture_test_opus41 import main, WebcamCapture

        mock_instance = MagicMock(spec=WebcamCapture)

        with patch('webcam_capture_test_opus41.verify_opencv_installation', return_value=True):
            with patch('webcam_capture_test_opus41.test_camera_availability', return_value=[0]):
                with patch('webcam_capture_test_opus41.WebcamCapture', return_value=mock_instance):
                    main()

        mock_instance.run.assert_called_once()
        mock_instance.cleanup.assert_called_once()

    @pytest.mark.unit
    def test_main_handles_keyboard_interrupt(self, temp_working_dir, mock_cuda_unavailable):
        """Test that main handles keyboard interrupt."""
        from webcam_capture_test_opus41 import main, WebcamCapture

        mock_instance = MagicMock(spec=WebcamCapture)
        mock_instance.run.side_effect = KeyboardInterrupt()

        with patch('webcam_capture_test_opus41.verify_opencv_installation', return_value=True):
            with patch('webcam_capture_test_opus41.test_camera_availability', return_value=[0]):
                with patch('webcam_capture_test_opus41.WebcamCapture', return_value=mock_instance):
                    # Should not raise exception
                    main()

        mock_instance.cleanup.assert_called_once()

    @pytest.mark.unit
    def test_main_handles_general_exception(self, temp_working_dir, mock_cuda_unavailable):
        """Test that main handles general exceptions."""
        from webcam_capture_test_opus41 import main, WebcamCapture

        mock_instance = MagicMock(spec=WebcamCapture)
        mock_instance.run.side_effect = Exception("Runtime error")

        with patch('webcam_capture_test_opus41.verify_opencv_installation', return_value=True):
            with patch('webcam_capture_test_opus41.test_camera_availability', return_value=[0]):
                with patch('webcam_capture_test_opus41.WebcamCapture', return_value=mock_instance):
                    # Should not raise exception
                    main()

        mock_instance.cleanup.assert_called_once()

    @pytest.mark.unit
    def test_main_selects_single_camera(self, temp_working_dir, mock_cuda_unavailable):
        """Test camera selection with single camera."""
        from webcam_capture_test_opus41 import main, WebcamCapture

        mock_instance = MagicMock(spec=WebcamCapture)

        with patch('webcam_capture_test_opus41.verify_opencv_installation', return_value=True):
            with patch('webcam_capture_test_opus41.test_camera_availability', return_value=[1]):
                with patch('webcam_capture_test_opus41.WebcamCapture', return_value=mock_instance) as mock_wc:
                    main()

        # Should use camera index 1
        mock_wc.assert_called_once_with(camera_index=1, resolution=(1280, 720))

    @pytest.mark.unit
    def test_main_selects_first_of_multiple_cameras(self, temp_working_dir, mock_cuda_unavailable):
        """Test camera selection with multiple cameras."""
        from webcam_capture_test_opus41 import main, WebcamCapture

        mock_instance = MagicMock(spec=WebcamCapture)

        with patch('webcam_capture_test_opus41.verify_opencv_installation', return_value=True):
            with patch('webcam_capture_test_opus41.test_camera_availability', return_value=[0, 1, 2]):
                with patch('webcam_capture_test_opus41.WebcamCapture', return_value=mock_instance) as mock_wc:
                    main()

        # Should use first camera (index 0)
        mock_wc.assert_called_once_with(camera_index=0, resolution=(1280, 720))


class TestImageProcessingFunctions:
    """Tests for image processing utility functions"""

    @pytest.mark.unit
    def test_grayscale_conversion(self, sample_bgr_frame):
        """Test BGR to grayscale conversion."""
        gray = cv2.cvtColor(sample_bgr_frame, cv2.COLOR_BGR2GRAY)

        assert gray.ndim == 2
        assert gray.shape[:2] == sample_bgr_frame.shape[:2]

    @pytest.mark.unit
    def test_gaussian_blur(self, sample_bgr_frame):
        """Test Gaussian blur operation."""
        blurred = cv2.GaussianBlur(sample_bgr_frame, (5, 5), 0)

        assert blurred.shape == sample_bgr_frame.shape

    @pytest.mark.unit
    def test_canny_edge_detection(self, sample_gray_frame):
        """Test Canny edge detection."""
        edges = cv2.Canny(sample_gray_frame, 50, 150)

        assert edges.ndim == 2
        assert edges.shape == sample_gray_frame.shape

    @pytest.mark.unit
    def test_binary_threshold(self, sample_gray_frame):
        """Test binary thresholding."""
        _, binary = cv2.threshold(sample_gray_frame, 127, 255, cv2.THRESH_BINARY)

        assert binary.ndim == 2
        unique_vals = np.unique(binary)
        assert all(v in [0, 255] for v in unique_vals)

    @pytest.mark.unit
    def test_adaptive_threshold(self, sample_gray_frame):
        """Test adaptive thresholding."""
        adaptive = cv2.adaptiveThreshold(
            sample_gray_frame, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 11, 2
        )

        assert adaptive.ndim == 2
        assert adaptive.shape == sample_gray_frame.shape

    @pytest.mark.unit
    def test_morphological_operations(self, sample_gray_frame):
        """Test morphological operations."""
        kernel = np.ones((3, 3), np.uint8)

        # Test opening
        opened = cv2.morphologyEx(sample_gray_frame, cv2.MORPH_OPEN, kernel)
        assert opened.shape == sample_gray_frame.shape

        # Test closing
        closed = cv2.morphologyEx(sample_gray_frame, cv2.MORPH_CLOSE, kernel)
        assert closed.shape == sample_gray_frame.shape

    @pytest.mark.unit
    def test_contour_detection(self, frame_with_objects):
        """Test contour detection."""
        gray = cv2.cvtColor(frame_with_objects, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        contours, hierarchy = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        assert isinstance(contours, tuple)
        assert len(contours) >= 1  # Should find at least one contour

    @pytest.mark.unit
    def test_bounding_rect(self, frame_with_objects):
        """Test bounding rectangle calculation."""
        gray = cv2.cvtColor(frame_with_objects, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            x, y, w, h = cv2.boundingRect(contours[0])
            assert all(v >= 0 for v in [x, y, w, h])
            assert w > 0 and h > 0

    @pytest.mark.unit
    def test_image_moments(self, frame_with_objects):
        """Test image moments calculation."""
        gray = cv2.cvtColor(frame_with_objects, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            M = cv2.moments(contours[0])
            assert 'm00' in M
            assert 'm10' in M
            assert 'm01' in M


class TestVideoCodecs:
    """Tests for video codec operations"""

    @pytest.mark.unit
    def test_xvid_codec(self):
        """Test XVID codec availability."""
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        assert fourcc != -1

    @pytest.mark.unit
    def test_mjpg_codec(self):
        """Test MJPG codec availability."""
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        assert fourcc != -1

    @pytest.mark.unit
    def test_mp4v_codec(self):
        """Test MP4V codec availability."""
        fourcc = cv2.VideoWriter_fourcc(*'MP4V')
        assert fourcc != -1


class TestDirectorySetup:
    """Tests for directory setup at module import"""

    @pytest.mark.unit
    def test_output_dir_exists(self, temp_working_dir):
        """Test that output directory is created."""
        output_dir = temp_working_dir / "output"
        assert output_dir.exists()

    @pytest.mark.unit
    def test_captures_dir_exists(self, temp_working_dir):
        """Test that captures directory is created."""
        captures_dir = temp_working_dir / "output" / "captures"
        assert captures_dir.exists()

    @pytest.mark.unit
    def test_videos_dir_exists(self, temp_working_dir):
        """Test that videos directory is created."""
        videos_dir = temp_working_dir / "output" / "videos"
        assert videos_dir.exists()
