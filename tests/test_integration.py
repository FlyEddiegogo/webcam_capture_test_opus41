# -*- coding: utf-8 -*-
"""
Integration tests for webcam_capture_test_opus41.
Tests cover end-to-end workflows and component interactions.
"""

import pytest
import numpy as np
import cv2
import sys
import time
import tempfile
import os
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch, call

sys.path.insert(0, str(Path(__file__).parent.parent))


class TestWebcamCaptureWorkflow:
    """Integration tests for complete WebcamCapture workflows"""

    @pytest.mark.integration
    def test_complete_capture_workflow(
        self, sample_bgr_frame, temp_working_dir, mock_cv2_imwrite
    ):
        """Test complete image capture workflow."""
        from webcam_capture_test_opus41 import WebcamCapture

        # Create mock camera
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.read.return_value = (True, sample_bgr_frame)
        mock_cap.get.return_value = 30

        with patch.object(cv2, 'VideoCapture', return_value=mock_cap):
            capture = WebcamCapture(camera_index=0)

            # Initialize camera
            success = capture.initialize_camera()
            assert success is True

            # Simulate frame capture
            capture.last_frame = sample_bgr_frame.copy()

            # Capture image
            filepath = capture.capture_image()
            assert filepath is not None

            # Cleanup
            with patch.object(cv2, 'destroyAllWindows'):
                capture.cleanup()

    @pytest.mark.integration
    def test_complete_recording_workflow(
        self, sample_bgr_frame, temp_working_dir
    ):
        """Test complete video recording workflow."""
        from webcam_capture_test_opus41 import WebcamCapture

        # Create mock camera
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.read.return_value = (True, sample_bgr_frame)
        mock_cap.get.return_value = 30

        # Create mock video writer
        mock_writer = MagicMock()
        mock_writer.isOpened.return_value = True

        with patch.object(cv2, 'VideoCapture', return_value=mock_cap):
            capture = WebcamCapture(camera_index=0)
            capture.initialize_camera()
            capture.last_frame = sample_bgr_frame.copy()

            with patch.object(cv2, 'VideoWriter', return_value=mock_writer):
                with patch.object(cv2, 'VideoWriter_fourcc', return_value=1):
                    # Start recording
                    success = capture.start_recording()
                    assert success is True
                    assert capture.recording is True

                    # Simulate writing frames
                    for _ in range(10):
                        if capture.recording and capture.video_writer:
                            capture.video_writer.write(sample_bgr_frame)

                    # Stop recording
                    capture.stop_recording()
                    assert capture.recording is False

            # Cleanup
            with patch.object(cv2, 'destroyAllWindows'):
                capture.cleanup()

    @pytest.mark.integration
    def test_frame_processing_pipeline(self, sample_bgr_frame):
        """Test complete frame processing pipeline."""
        from webcam_capture_test_opus41 import WebcamCapture

        capture = WebcamCapture()

        # Preprocess frame
        processed = capture.preprocess_frame(sample_bgr_frame)

        # Verify all processing stages
        assert 'gray' in processed
        assert 'blur' in processed
        assert 'edges' in processed
        assert 'binary' in processed
        assert 'adaptive' in processed
        assert 'morphology' in processed

        # Detect objects
        frame_with_objects, objects = capture.detect_objects(sample_bgr_frame)

        # Verify detection output
        assert isinstance(frame_with_objects, np.ndarray)
        assert isinstance(objects, list)

        # Add overlay
        frame_with_overlay = capture.add_overlay_info(frame_with_objects)
        assert frame_with_overlay is not None

    @pytest.mark.integration
    def test_fps_calculation_over_time(self):
        """Test FPS calculation over multiple frames."""
        from webcam_capture_test_opus41 import WebcamCapture

        capture = WebcamCapture()

        # Simulate 60 frames over ~2 seconds
        start_time = time.time()
        capture.fps_start_time = start_time

        for i in range(60):
            capture.calculate_fps()
            time.sleep(0.01)  # ~100 FPS simulation

        # FPS should have been calculated at least once
        assert capture.fps > 0

    @pytest.mark.integration
    def test_object_detection_with_multiple_objects(self):
        """Test object detection with multiple objects."""
        from webcam_capture_test_opus41 import WebcamCapture

        # Create frame with multiple objects
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.rectangle(frame, (50, 50), (150, 150), (255, 255, 255), -1)
        cv2.rectangle(frame, (200, 100), (350, 250), (255, 255, 255), -1)
        cv2.rectangle(frame, (400, 200), (550, 400), (255, 255, 255), -1)
        cv2.circle(frame, (100, 350), 50, (255, 255, 255), -1)

        capture = WebcamCapture()
        _, objects = capture.detect_objects(frame)

        # Should detect multiple objects
        assert len(objects) >= 1

        # Each object should have required fields
        for obj in objects:
            assert 'id' in obj
            assert 'bbox' in obj
            assert 'center' in obj
            assert 'area' in obj


class TestEndToEndScenarios:
    """End-to-end scenario tests"""

    @pytest.mark.integration
    def test_camera_initialization_and_frame_read(self, sample_bgr_frame, temp_working_dir):
        """Test camera initialization followed by frame reading."""
        from webcam_capture_test_opus41 import WebcamCapture

        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.read.return_value = (True, sample_bgr_frame)
        mock_cap.get.return_value = 30

        with patch.object(cv2, 'VideoCapture', return_value=mock_cap):
            capture = WebcamCapture()
            success = capture.initialize_camera()

            assert success is True

            # Read frame
            ret, frame = capture.cap.read()
            assert ret is True
            assert frame is not None
            assert frame.shape == sample_bgr_frame.shape

            with patch.object(cv2, 'destroyAllWindows'):
                capture.cleanup()

    @pytest.mark.integration
    def test_full_main_loop_simulation(
        self, sample_bgr_frame, temp_working_dir, mock_all_cv2_gui
    ):
        """Test simulated main loop execution."""
        from webcam_capture_test_opus41 import WebcamCapture

        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.read.return_value = (True, sample_bgr_frame)
        mock_cap.get.return_value = 30

        # Simulate quit after a few iterations
        call_count = [0]
        def mock_waitkey(delay):
            call_count[0] += 1
            if call_count[0] >= 3:
                return ord('q')
            return 255  # No key pressed

        mock_all_cv2_gui['waitKey'].side_effect = mock_waitkey

        with patch.object(cv2, 'VideoCapture', return_value=mock_cap):
            capture = WebcamCapture()
            capture.run()

        assert capture.is_running is False

    @pytest.mark.integration
    def test_recording_during_capture(
        self, sample_bgr_frame, temp_working_dir
    ):
        """Test recording while capturing frames."""
        from webcam_capture_test_opus41 import WebcamCapture

        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.read.return_value = (True, sample_bgr_frame)
        mock_cap.get.return_value = 30

        mock_writer = MagicMock()
        mock_writer.isOpened.return_value = True

        with patch.object(cv2, 'VideoCapture', return_value=mock_cap):
            capture = WebcamCapture()
            capture.initialize_camera()
            capture.last_frame = sample_bgr_frame.copy()

            with patch.object(cv2, 'VideoWriter', return_value=mock_writer):
                with patch.object(cv2, 'VideoWriter_fourcc', return_value=1):
                    # Start recording
                    capture.start_recording()

                    # Simulate frame capture and write
                    for _ in range(5):
                        ret, frame = capture.cap.read()
                        if ret and capture.recording:
                            capture.video_writer.write(frame)
                            capture.frame_count += 1

                    capture.stop_recording()

        assert capture.frame_count >= 5
        assert mock_writer.write.call_count >= 5


class TestErrorHandlingIntegration:
    """Integration tests for error handling"""

    @pytest.mark.integration
    def test_recovery_from_frame_read_failure(self, sample_bgr_frame, temp_working_dir):
        """Test recovery when frame read fails temporarily."""
        from webcam_capture_test_opus41 import WebcamCapture

        # Simulate intermittent failures
        read_results = [
            (True, sample_bgr_frame),
            (False, None),  # Failure
            (True, sample_bgr_frame),
        ]
        read_index = [0]

        def mock_read():
            result = read_results[read_index[0] % len(read_results)]
            read_index[0] += 1
            return result

        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.read.side_effect = mock_read
        mock_cap.get.return_value = 30

        with patch.object(cv2, 'VideoCapture', return_value=mock_cap):
            capture = WebcamCapture()
            success = capture.initialize_camera()

            # Should handle failures gracefully
            for _ in range(3):
                ret, frame = capture.cap.read()
                if ret and frame is not None:
                    capture.last_frame = frame

            with patch.object(cv2, 'destroyAllWindows'):
                capture.cleanup()

    @pytest.mark.integration
    def test_cleanup_after_exception(self, sample_bgr_frame, temp_working_dir):
        """Test that cleanup happens even after exception."""
        from webcam_capture_test_opus41 import WebcamCapture

        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.read.return_value = (True, sample_bgr_frame)
        mock_cap.get.return_value = 30

        with patch.object(cv2, 'VideoCapture', return_value=mock_cap):
            capture = WebcamCapture()
            capture.cap = mock_cap

            try:
                raise Exception("Test exception")
            except Exception:
                pass
            finally:
                with patch.object(cv2, 'destroyAllWindows'):
                    capture.cleanup()

        mock_cap.release.assert_called_once()


class TestConcurrentOperations:
    """Tests for concurrent/simultaneous operations"""

    @pytest.mark.integration
    def test_simultaneous_capture_and_detection(self, sample_bgr_frame):
        """Test simultaneous capture and object detection."""
        from webcam_capture_test_opus41 import WebcamCapture

        capture = WebcamCapture()
        capture.last_frame = sample_bgr_frame.copy()

        # Perform both operations
        processed = capture.preprocess_frame(sample_bgr_frame)
        _, objects = capture.detect_objects(sample_bgr_frame)
        overlay_frame = capture.add_overlay_info(sample_bgr_frame.copy())

        # All should succeed
        assert len(processed) > 0
        assert isinstance(objects, list)
        assert overlay_frame is not None

    @pytest.mark.integration
    def test_recording_while_processing(
        self, sample_bgr_frame, temp_working_dir
    ):
        """Test recording while processing frames."""
        from webcam_capture_test_opus41 import WebcamCapture

        mock_writer = MagicMock()
        mock_writer.isOpened.return_value = True

        capture = WebcamCapture()
        capture.last_frame = sample_bgr_frame.copy()

        with patch.object(cv2, 'VideoWriter', return_value=mock_writer):
            with patch.object(cv2, 'VideoWriter_fourcc', return_value=1):
                capture.start_recording()

                # Process and record simultaneously
                for _ in range(5):
                    # Process frame
                    processed = capture.preprocess_frame(sample_bgr_frame)
                    _, objects = capture.detect_objects(sample_bgr_frame)

                    # Write to video
                    if capture.recording:
                        capture.video_writer.write(sample_bgr_frame)

                capture.stop_recording()

        assert mock_writer.write.call_count == 5


class TestPerformanceScenarios:
    """Performance-related integration tests"""

    @pytest.mark.integration
    @pytest.mark.slow
    def test_continuous_frame_processing(self, sample_small_frame):
        """Test continuous frame processing performance."""
        from webcam_capture_test_opus41 import WebcamCapture

        capture = WebcamCapture()

        start_time = time.time()
        frame_count = 0

        # Process frames for ~1 second
        while time.time() - start_time < 1.0:
            processed = capture.preprocess_frame(sample_small_frame)
            _, objects = capture.detect_objects(sample_small_frame)
            frame_count += 1

        # Should be able to process multiple frames per second
        assert frame_count > 5  # At least 5 FPS for small frames

    @pytest.mark.integration
    def test_memory_stability(self, sample_small_frame):
        """Test memory stability during repeated operations."""
        from webcam_capture_test_opus41 import WebcamCapture

        capture = WebcamCapture()

        # Perform many iterations
        for _ in range(100):
            processed = capture.preprocess_frame(sample_small_frame)
            _, objects = capture.detect_objects(sample_small_frame)
            overlay = capture.add_overlay_info(sample_small_frame.copy())

        # Should complete without memory errors


class TestModuleImport:
    """Tests for module import and initialization"""

    @pytest.mark.integration
    def test_module_import_creates_directories(self, temp_working_dir):
        """Test that importing module creates output directories."""
        # Directories should exist from conftest fixture
        output_dir = temp_working_dir / "output"
        captures_dir = output_dir / "captures"
        videos_dir = output_dir / "videos"

        assert output_dir.exists()
        assert captures_dir.exists()
        assert videos_dir.exists()

    @pytest.mark.integration
    def test_logger_initialization(self):
        """Test that logger is properly initialized."""
        import logging
        from webcam_capture_test_opus41 import logger

        assert logger is not None
        assert isinstance(logger, logging.Logger)


class TestCrossComponentInteraction:
    """Tests for interactions between different components"""

    @pytest.mark.integration
    def test_preprocessing_to_detection_flow(self, frame_with_objects):
        """Test data flow from preprocessing to detection."""
        from webcam_capture_test_opus41 import WebcamCapture

        capture = WebcamCapture()

        # Preprocess
        processed = capture.preprocess_frame(frame_with_objects)

        # Verify edges are used in detection
        assert 'edges' in processed

        # Detection should work with preprocessed data
        _, objects = capture.detect_objects(frame_with_objects)

        # Should find objects
        assert len(objects) >= 1

    @pytest.mark.integration
    def test_detection_to_overlay_flow(self, frame_with_objects):
        """Test data flow from detection to overlay."""
        from webcam_capture_test_opus41 import WebcamCapture

        capture = WebcamCapture()
        capture.fps = 30.0
        capture.frame_count = 100

        # Detect objects
        frame_with_detections, objects = capture.detect_objects(frame_with_objects)

        # Add overlay to detected frame
        final_frame = capture.add_overlay_info(frame_with_detections)

        # Final frame should be valid
        assert final_frame is not None
        assert final_frame.shape == frame_with_objects.shape
