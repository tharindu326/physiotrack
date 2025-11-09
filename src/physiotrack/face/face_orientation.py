"""
Face orientation estimation using 6DRepNet360.
"""
import os
import numpy as np
from typing import Optional, Union
from ..modules._6DRepNet360 import HeadPoseEstimator
from ..models import Models


class FaceOrientation:
    """
    Face orientation estimator using 6DRepNet360.
    
    This class provides head pose estimation (roll, pitch, yaw) from face images
    to determine the orientation of the face in 3D space using the 6D rotation representation network.
    
    Args:
        model: Face orientation model from Models.Pose3D.FaceOrientation
               Default: Models.Pose3D.FaceOrientation.default
        device: Device to use for inference. Can be int (e.g., 0 for cuda:0)
               or str (e.g., 'cpu', 'cuda'). Default: 'cpu'
        render_pose: Whether to render pose axes on output (default: False)
        verbose: Whether to print verbose output (default: True)
        **kwargs: Additional arguments passed to the estimator
    
    Usage:
        >>> from physiotrack import FaceOrientation, Face, Models
        >>> import cv2
        >>> 
        >>> # Initialize face detector and face orientation estimator
        >>> face_detector = Face(device='cuda')
        >>> face_orientation = FaceOrientation(
        ...     model=Models.Pose3D.FaceOrientation.default,
        ...     device='cuda',
        ...     render_pose=True
        ... )
        >>> 
        >>> # Detect faces
        >>> frame = cv2.imread('image.jpg')
        >>> det_results, _ = face_detector.detect(frame)
        >>> 
        >>> # Extract bounding boxes
        >>> bboxes = det_results[0].boxes.data.cpu().numpy()[:, :4]
        >>> 
        >>> # Estimate face orientation
        >>> output_img, pose_results = face_orientation.predict(frame, bboxes)
        >>> 
        >>> # Access pose angles
        >>> for detection in pose_results['detections']:
        ...     pose = detection['pose']
        ...     print(f"Yaw: {pose['yaw']:.1f}°")
        ...     print(f"Pitch: {pose['pitch']:.1f}°")
        ...     print(f"Roll: {pose['roll']:.1f}°")
        >>> 
        >>> # Batch processing
        >>> batch_outputs = face_orientation.predict_batch(frames, bboxes_list)
    
    Returns:
        For single frame:
            - output_img: Image with pose visualization (if render_pose=True), None otherwise
            - results: Dictionary containing:
                {
                    "detections": [
                        {
                            "id": int,
                            "bbox": [x1, y1, x2, y2],
                            "pose": {
                                "pitch": float,  # degrees
                                "yaw": float,    # degrees
                                "roll": float    # degrees
                            },
                            "rotation_matrix": [[...], [...], [...]]  # 3x3 matrix
                        },
                        ...
                    ]
                }
        
        For batch:
            - List of (output_img, results) tuples
    """
    
    def __init__(self,
                 model=None,
                 device: Union[str, int] = 'cpu',
                 render_pose: bool = False,
                 verbose: bool = True,
                 **kwargs):
        
        # Validate and get model
        if model is None:
            model = Models.Pose3D.FaceOrientation.default
        
        Models.validate_pose3d_model(model, expected_subclass='FaceOrientation')
        
        # Check if model file exists, download if needed
        model_path = os.path.join(
            os.path.dirname(__file__), 
            '..', 
            'modules', 
            'model_data', 
            model.value
        )
        if not os.path.isfile(model_path):
            # Try to download from HuggingFace, but skip if not available
            # The 6DRepNet360 model will auto-download from its source
            try:
                Models.download_model(model)
            except Exception as e:
                if verbose:
                    print(f"Note: Could not download from HuggingFace ({e}). Model will auto-download from 6DRepNet source.")
        
        # Initialize the head pose estimator
        self.estimator = HeadPoseEstimator(
            model=model,
            device=device,
            render_pose=render_pose,
            verbose=verbose,
            **kwargs
        )
        
        self.model = model
        self.device = device
        self.render_pose = render_pose
        self.verbose = verbose
    
    def __call__(self, img: np.ndarray, bboxes: np.ndarray = None):
        """
        Make the class callable for single-frame inference.
        
        Args:
            img: Input image in BGR format (OpenCV format)
            bboxes: Face bounding boxes [N, 4] with format [x1, y1, x2, y2]
        
        Returns:
            output_img: Image with pose visualization (if render_pose=True)
            results: Dictionary containing pose estimation results
        """
        return self.predict(img, bboxes)
    
    def predict(self, img: np.ndarray, bboxes: np.ndarray = None):
        """
        Perform head pose estimation on a single frame.
        
        Args:
            img: Input image in BGR format (OpenCV format)
            bboxes: Face bounding boxes [N, 4] with format [x1, y1, x2, y2]
                   If None, assumes the entire image is a face
        
        Returns:
            output_img: Image with pose visualization (if render_pose=True), None otherwise
            results: Dictionary with pose predictions for each detected face
        """
        return self.estimator.predict(img, bboxes)
    
    def predict_batch(self, frames, bboxes_list=None):
        """
        Perform batch inference on multiple frames.
        
        Args:
            frames: List of input images in BGR format
            bboxes_list: List of bounding boxes for each frame
        
        Returns:
            List of (output_img, results) tuples for each frame
        """
        return self.estimator.predict_batch(frames, bboxes_list)
    
    def get_avg_inference_time(self):
        """Get average inference time in milliseconds."""
        return self.estimator.get_avg_inference_time()
    
    def get_avg_fps(self):
        """Get average FPS based on inference times."""
        return self.estimator.get_avg_fps()

