import os
import time
from collections import deque
from typing import Optional, List, Tuple, Union

import cv2
import numpy as np
import torch
from torchvision import transforms

from .model import load_model
from . import utils


class HeadPoseEstimator:
    """
    Class for performing head pose estimation using SixDRepNet360 model.
    
    This class provides single-frame and batch inference capabilities for estimating
    head pose (roll, pitch, yaw) from face images.
    
    Args:
        model: Model enum object with .value attribute containing the model filename.
               If None, downloads pretrained weights.
        device (Union[str, int], optional): Device to use for inference. Can be int (e.g., 0 for cuda:0)
               or str (e.g., 'cuda', 'cpu'). Defaults to 'cuda' if available, else 'cpu'.
        render_pose (bool, optional): Whether to render pose visualization on output images. Defaults to False.
        verbose (bool, optional): Whether to print verbose information. Defaults to False.
    """

    def __init__(self,
                 model = None,
                 device: Optional[Union[str, int]] = None,
                 render_pose: bool = False,
                 verbose: bool = False,
                 **kwargs):
        
        # Device priority is cuda / mps / cpu
        if device is None:
            if torch.cuda.is_available():
                device = 'cuda'
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                device = 'mps'
            else:
                device = 'cpu'
        # Handle integer device input (e.g., 0 for cuda:0) - for consistency with YOLO models
        elif isinstance(device, int):
            if device < 0:
                device = 'cpu'
            else:
                device = f'cuda:{device}' if device > 0 else 'cuda'

        self.device = device
        self.render_pose = render_pose
        self.verbose = verbose
        self.extra_args = kwargs
        
        # Load model following the same pattern as Yolo
        model_path = os.path.join(os.path.dirname(__file__), '..', 'model_data')
        if model is not None:
            snapshot_path = os.path.join(model_path, model.value)
            # Check if file exists, if not set to None to trigger auto-download
            if not os.path.isfile(snapshot_path):
                if self.verbose:
                    print(f"Model file not found at {snapshot_path}, will auto-download from source.")
                snapshot_path = None
        else:
            snapshot_path = None  # Will download pretrained weights
        
        self.model = load_model(snapshot_path, device=self.device)
        
        # Image preprocessing
        self.transformations = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # FPS monitoring
        self.inference_times = deque(maxlen=100)
    
    def __call__(self, img: np.ndarray, bboxes: Optional[np.ndarray] = None) -> Tuple[np.ndarray, dict]:
        """
        Make the class callable for single-frame inference.
        
        Args:
            img: Input image in BGR format (OpenCV format)
            bboxes: Optional bounding boxes for faces [N, 4] with format [x1, y1, x2, y2]
                   If None, assumes the entire image is a face
        
        Returns:
            output_img: Image with pose visualization (if render_pose=True)
            results: Dictionary containing pose estimation results
        """
        return self.predict(img, bboxes)
    
    def predict(self, img: np.ndarray, bboxes: Optional[np.ndarray] = None) -> Tuple[Optional[np.ndarray], dict]:
        """
        Perform head pose estimation on a single frame.
        
        Args:
            img: Input image in BGR format (OpenCV format)
            bboxes: Optional bounding boxes for faces [N, 4] with format [x1, y1, x2, y2]
                   If None, assumes the entire image is a face
        
        Returns:
            output_img: Image with pose visualization (if render_pose=True), None otherwise
            results: Dictionary with pose predictions for each detected face
        """
        start = time.perf_counter()
        
        # Convert BGR to RGB
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Initialize results
        results = {"detections": []}
        output_img = None
        
        # Handle bboxes
        if bboxes is None:
            # Use entire image as a single face
            h, w = img.shape[:2]
            bboxes = np.array([[0, 0, w, h]])
        elif len(bboxes) == 0:
            # No faces detected
            inference_time = time.perf_counter() - start
            self.inference_times.append(inference_time)
            if self.render_pose:
                output_img = img.copy()
            return output_img, results
        
        # Ensure bboxes is numpy array
        bboxes = np.array(bboxes, dtype=int)
        if len(bboxes.shape) == 1:
            bboxes = bboxes.reshape(1, -1)
        
        # Crop faces and prepare for batch inference
        cropped_faces = []
        crop_metadata = []
        
        for i, bbox in enumerate(bboxes):
            x1, y1, x2, y2 = bbox[:4]
            
            # Add padding
            pad = 10
            x1 = max(0, x1 - pad)
            y1 = max(0, y1 - pad)
            x2 = min(img_rgb.shape[1], x2 + pad)
            y2 = min(img_rgb.shape[0], y2 + pad)
            
            # Crop face
            face_crop = img_rgb[y1:y2, x1:x2]
            
            if face_crop.size == 0:
                continue
            
            cropped_faces.append(face_crop)
            crop_metadata.append({
                'bbox': bbox,
                'center': ((x1 + x2) // 2, (y1 + y2) // 2),
                'index': i
            })
        
        if len(cropped_faces) == 0:
            inference_time = time.perf_counter() - start
            self.inference_times.append(inference_time)
            if self.render_pose:
                output_img = img.copy()
            return output_img, results
        
        # Batch inference
        rotation_matrices = self._inference_batch(cropped_faces)
        
        # Process results
        for rot_matrix, metadata in zip(rotation_matrices, crop_metadata):
            # Convert rotation matrix to Euler angles
            euler = utils.compute_euler_angles_from_rotation_matrices(
                rot_matrix.unsqueeze(0)
            ) * 180 / np.pi
            
            pitch = float(euler[0, 0].cpu().numpy())
            yaw = float(euler[0, 1].cpu().numpy())
            roll = float(euler[0, 2].cpu().numpy())
            
            detection = {
                "id": metadata['index'],
                "bbox": metadata['bbox'].tolist(),
                "pose": {
                    "pitch": pitch,
                    "yaw": yaw,
                    "roll": roll
                },
                "rotation_matrix": rot_matrix.cpu().numpy().tolist()
            }
            results["detections"].append(detection)
        
        # Render visualization if requested
        if self.render_pose:
            output_img = img.copy()
            for detection in results["detections"]:
                bbox = detection["bbox"]
                pose = detection["pose"]
                center_x = (bbox[0] + bbox[2]) // 2
                center_y = (bbox[1] + bbox[3]) // 2
                
                # Draw bounding box
                cv2.rectangle(output_img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), 
                            (0, 255, 0), 2)
                
                # Draw pose axes
                utils.draw_axis(output_img, pose['yaw'], pose['pitch'], pose['roll'],
                              tdx=center_x, tdy=center_y, size=100)
                
                # Add text annotation
                text = f"Y:{pose['yaw']:.1f} P:{pose['pitch']:.1f} R:{pose['roll']:.1f}"
                cv2.putText(output_img, text, (bbox[0], bbox[1] - 10),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
        
        inference_time = time.perf_counter() - start
        self.inference_times.append(inference_time)
        
        return output_img, results
    
    def predict_batch(self, frames: List[np.ndarray], 
                     bboxes_list: Optional[List[np.ndarray]] = None) -> List[Tuple[Optional[np.ndarray], dict]]:
        """
        Perform batch inference on multiple frames.
        
        This method processes all face crops across all frames in a single batch
        for maximum efficiency.
        
        Args:
            frames: List of input images in BGR format
            bboxes_list: List of bounding boxes for each frame. Each element should be
                        an array of shape [N, 4] with format [x1, y1, x2, y2]
                        If None, assumes each entire image is a face
        
        Returns:
            List of (output_img, results) tuples for each frame
        """
        if bboxes_list is None:
            bboxes_list = [None] * len(frames)
        
        start = time.perf_counter()
        
        # Collect all face crops from all frames
        all_crops = []
        crop_metadata = []
        
        for frame_idx, (frame, bboxes) in enumerate(zip(frames, bboxes_list)):
            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Handle bboxes
            if bboxes is None:
                h, w = frame.shape[:2]
                bboxes = np.array([[0, 0, w, h]])
            elif len(bboxes) == 0:
                continue
            
            bboxes = np.array(bboxes, dtype=int)
            if len(bboxes.shape) == 1:
                bboxes = bboxes.reshape(1, -1)
            
            # Crop faces
            for bbox_idx, bbox in enumerate(bboxes):
                x1, y1, x2, y2 = bbox[:4]
                
                # Add padding
                pad = 10
                x1 = max(0, x1 - pad)
                y1 = max(0, y1 - pad)
                x2 = min(img_rgb.shape[1], x2 + pad)
                y2 = min(img_rgb.shape[0], y2 + pad)
                
                # Crop face
                face_crop = img_rgb[y1:y2, x1:x2]
                
                if face_crop.size == 0:
                    continue
                
                all_crops.append(face_crop)
                crop_metadata.append({
                    'frame_idx': frame_idx,
                    'bbox_idx': bbox_idx,
                    'bbox': bbox,
                    'center': ((x1 + x2) // 2, (y1 + y2) // 2)
                })
        
        # Batch inference on all crops
        if len(all_crops) > 0:
            rotation_matrices = self._inference_batch(all_crops)
        else:
            rotation_matrices = []
        
        # Initialize per-frame results
        per_frame_results = [{"detections": []} for _ in frames]
        
        # Scatter results back to frames
        for rot_matrix, metadata in zip(rotation_matrices, crop_metadata):
            frame_idx = metadata['frame_idx']
            
            # Convert rotation matrix to Euler angles
            euler = utils.compute_euler_angles_from_rotation_matrices(
                rot_matrix.unsqueeze(0)
            ) * 180 / np.pi
            
            pitch = float(euler[0, 0].cpu().numpy())
            yaw = float(euler[0, 1].cpu().numpy())
            roll = float(euler[0, 2].cpu().numpy())
            
            detection = {
                "id": metadata['bbox_idx'],
                "bbox": metadata['bbox'].tolist(),
                "pose": {
                    "pitch": pitch,
                    "yaw": yaw,
                    "roll": roll
                },
                "rotation_matrix": rot_matrix.cpu().numpy().tolist()
            }
            per_frame_results[frame_idx]["detections"].append(detection)
        
        # Build per-frame outputs
        outputs = []
        for frame_idx, (frame, frame_results) in enumerate(zip(frames, per_frame_results)):
            output_img = None
            
            if self.render_pose:
                output_img = frame.copy()
                for detection in frame_results["detections"]:
                    bbox = detection["bbox"]
                    pose = detection["pose"]
                    center_x = (bbox[0] + bbox[2]) // 2
                    center_y = (bbox[1] + bbox[3]) // 2
                    
                    # Draw bounding box
                    cv2.rectangle(output_img, (bbox[0], bbox[1]), (bbox[2], bbox[3]),
                                (0, 255, 0), 2)
                    
                    # Draw pose axes
                    utils.draw_axis(output_img, pose['yaw'], pose['pitch'], pose['roll'],
                                  tdx=center_x, tdy=center_y, size=100)
                    
                    # Add text annotation
                    text = f"Y:{pose['yaw']:.1f} P:{pose['pitch']:.1f} R:{pose['roll']:.1f}"
                    cv2.putText(output_img, text, (bbox[0], bbox[1] - 10),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
            
            outputs.append((output_img, frame_results))
        
        # Update timing metrics
        elapsed = time.perf_counter() - start
        if len(frames) > 0:
            per_frame_time = elapsed / len(frames)
            for _ in frames:
                self.inference_times.append(per_frame_time)
        
        return outputs
    
    def _inference_batch(self, face_crops: List[np.ndarray]) -> List[torch.Tensor]:
        """
        Perform batch inference on a list of face crops.
        
        Args:
            face_crops: List of face images (RGB numpy arrays)
        
        Returns:
            List of rotation matrices (3x3 tensors)
        """
        if len(face_crops) == 0:
            return []
        
        # Preprocess all crops
        batch_tensors = []
        for crop in face_crops:
            # Convert numpy array to PIL Image for torchvision transforms
            from PIL import Image
            pil_img = Image.fromarray(crop)
            tensor = self.transformations(pil_img)
            batch_tensors.append(tensor)
        
        # Stack into batch
        batch = torch.stack(batch_tensors).to(self.device)
        
        # Inference
        with torch.no_grad():
            rotation_matrices = self.model(batch)
        
        # Return list of individual rotation matrices
        return [rotation_matrices[i] for i in range(rotation_matrices.size(0))]
    
    def get_avg_inference_time(self):
        """Get average inference time in milliseconds."""
        if len(self.inference_times) == 0:
            return 0.0
        return (sum(self.inference_times) / len(self.inference_times)) * 1000
    
    def get_avg_fps(self):
        """Get average FPS based on inference times."""
        if len(self.inference_times) == 0:
            return 0.0
        avg_time = sum(self.inference_times) / len(self.inference_times)
        return 1.0 / avg_time if avg_time > 0 else 0.0


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Head pose estimation inference')
    parser.add_argument('--input', type=str, required=True,
                       help='Path to input image or directory')
    parser.add_argument('--output', type=str, default='output',
                       help='Path to output directory')
    parser.add_argument('--device', type=str, default=None,
                       help='Device to use (cuda/cpu, default: auto)')
    parser.add_argument('--render', action='store_true',
                       help='Render pose visualization')
    args = parser.parse_args()
    
    # Initialize estimator (model=None will download pretrained weights)
    estimator = HeadPoseEstimator(
        model=None,
        device=args.device,
        render_pose=args.render,
        verbose=True
    )
    
    os.makedirs(args.output, exist_ok=True)
    
    if os.path.isfile(args.input):
        # Process single image
        img = cv2.imread(args.input)
        if img is None:
            print(f"Error: Unable to load image {args.input}")
        else:
            output_img, results = estimator.predict(img)
            
            # Save results
            output_path = os.path.join(args.output, os.path.basename(args.input))
            if output_img is not None:
                cv2.imwrite(output_path, output_img)
            
            print(f"Processed {args.input}")
            print(f"Results: {results}")
            print(f"Saved to {output_path}")
    
    elif os.path.isdir(args.input):
        # Process directory
        image_files = [f for f in os.listdir(args.input) 
                      if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
        
        for img_file in image_files:
            img_path = os.path.join(args.input, img_file)
            img = cv2.imread(img_path)
            
            if img is None:
                print(f"Skipping {img_file}: Unable to load")
                continue
            
            output_img, results = estimator.predict(img)
            
            # Save results
            output_path = os.path.join(args.output, img_file)
            if output_img is not None:
                cv2.imwrite(output_path, output_img)
            
            print(f"Processed {img_file}: {len(results['detections'])} faces detected")
        
        print(f"\nAverage inference time: {estimator.get_avg_inference_time():.2f} ms")
        print(f"Average FPS: {estimator.get_avg_fps():.2f}")
    
    else:
        print(f"Error: {args.input} is not a valid file or directory")

