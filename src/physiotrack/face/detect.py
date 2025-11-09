"""
Face detection module using YOLO detectors.
"""
from ..detect import ValidatedDetector
from ..models import Models


class Face(ValidatedDetector):
    """
    Face detector using YOLO models.
    
    This class provides face detection capabilities using pre-trained YOLO models.
    It inherits from ValidatedDetector which handles model validation and loading.
    
    Args:
        model: Face detection model from Models.Detection.YOLO.FACE
               Default: Models.Detection.YOLO.FACE.m_face
        device: Device to use for inference ('cpu', 'cuda', or device id)
        OBJECTNESS_CONFIDENCE: Confidence threshold for detections (default: 0.24)
        NMS_THRESHOLD: Non-maximum suppression threshold (default: 0.4)
        classes: List of class indices to detect (default: None, detects all)
        render_labels: Whether to render labels on output (default: True)
        render_box_detections: Whether to render bounding boxes on output (default: True)
        verbose: Whether to print verbose output (default: True)
        **kwargs: Additional arguments passed to the detector
    
    Usage:
        >>> from physiotrack import Face, Models
        >>> 
        >>> # Initialize with default model
        >>> face_detector = Face(device='cuda')
        >>> 
        >>> # Or specify a specific model
        >>> face_detector = Face(
        ...     model=Models.Detection.YOLO.FACE.l_face,
        ...     device='cuda',
        ...     OBJECTNESS_CONFIDENCE=0.3,
        ...     render_box_detections=True
        ... )
        >>> 
        >>> # Detect faces
        >>> results, output_img = face_detector.detect(frame)
        >>> 
        >>> # Batch detection
        >>> batch_results = face_detector.detect_batch(frames)
    """
    expected_subclass = "Face"
    model = Models.Detection.YOLO.FACE.m_face

