from .config import COCO, COCO_WHOLEBODY, HALPE_TO_COCO_KEYPOINT_MAP, HUMAN26M
import json
import os
from tqdm import tqdm
import numpy as np
from pathlib import Path

class Keypoint:
    def __init__(self, id, x, y, confidence, keypoint_names):
        self.id = id
        self.x = x
        self.y = y
        self.confidence = confidence
        self._keypoint_names = keypoint_names
    
    @property
    def name(self):
        return self._keypoint_names.get(str(self.id), f"unknown_{self.id}")
    
class KeypointCollection:
    def __init__(self, keypoints_data, pose_archetecture):
        self._keypoint_names = COCO_WHOLEBODY if pose_archetecture == "WHOLEBODY" else COCO
        self._keypoints = {}
        self._keypoints_by_name = {}
        
        for kp_data in keypoints_data:
            kp = Keypoint(
                kp_data['id'], 
                kp_data['x'], 
                kp_data['y'], 
                kp_data['confidence'],
                self._keypoint_names
            )
            self._keypoints[kp.id] = kp
            self._keypoints_by_name[kp.name] = kp
    
    def id(self, keypoint_id):
        """Return keypoint by ID"""
        return self._keypoints.get(keypoint_id)
    
    def name(self, keypoint_name):
        """Return keypoint by name"""
        return self._keypoints_by_name.get(keypoint_name)

class PoseObject:
    def __init__(self, detection_data, pose_archetecture="WHOLEBODY"):
        self.id = detection_data['id']
        self.box = detection_data['bbox']  # [x1, y1, x2, y2]
        self.keypoints = KeypointCollection(detection_data['keypoints'], pose_archetecture)

class PoseObjectsFrame:
    def __init__(self, frame_data, pose_archetecture="WHOLEBODY"):
        self.frame_data = frame_data
        self.pose_archetecture = pose_archetecture
        self.pose_objects = []
        self.convert_frame_data_to_poses()

    def __iter__(self):
        """Make PoseObjects iterable"""
        return iter(self.pose_objects)
    
    def __len__(self):
        """Get number of poses"""
        return len(self.pose_objects)
    
    def __getitem__(self, index):
        """Get pose by index"""
        return self.pose_objects[index]
    
    def to_json(self):
        return self.frame_data

    def convert_frame_data_to_poses(self):
        """Convert frame_data to list of PoseObject instances"""
        for detection in self.frame_data['detections']:
            pose = PoseObject(detection, self.pose_archetecture)
            self.pose_objects.append(pose)
        return self.pose_objects


def convert_to_halpe_pose_format(frame_data, image_filename):
    halpe_pose_data = []
    
    if "detections" not in frame_data or not frame_data["detections"]:
        return halpe_pose_data
    
    for detection in frame_data["detections"]:
        x1, y1, x2, y2 = detection["bbox"]
        bbox_halpe_format = [x1, y1, x2 - x1, y2 - y1]
        
        keypoints_flat = [0.0] * 78  # 26 keypoints × 3 values
        
        keypoint_dict = {}
        for kp in detection["keypoints"]:
            keypoint_dict[kp["id"]] = (kp["x"], kp["y"], kp["confidence"])
        
        # Helper function to calculate midpoint between two keypoints
        def calculate_midpoint(kp1_id, kp2_id):
            if kp1_id in keypoint_dict and kp2_id in keypoint_dict:
                x1, y1, conf1 = keypoint_dict[kp1_id]
                x2, y2, conf2 = keypoint_dict[kp2_id]
                # Only calculate if both keypoints have reasonable confidence
                if conf1 > 0.1 and conf2 > 0.1:
                    mid_x = (x1 + x2) / 2
                    mid_y = (y1 + y2) / 2
                    mid_conf = min(conf1, conf2)  # Use minimum confidence
                    return mid_x, mid_y, mid_conf
            return None, None, 0.0
        
        # Helper function to calculate center of three points
        def calculate_center_of_three(kp1_id, kp2_id, kp3_id):
            if kp1_id in keypoint_dict and kp2_id in keypoint_dict and kp3_id in keypoint_dict:
                x1, y1, conf1 = keypoint_dict[kp1_id]
                x2, y2, conf2 = keypoint_dict[kp2_id]
                x3, y3, conf3 = keypoint_dict[kp3_id]
                # Only calculate if all keypoints have reasonable confidence
                if conf1 > 0.1 and conf2 > 0.1 and conf3 > 0.1:
                    center_x = (x1 + x2 + x3) / 3
                    center_y = (y1 + y2 + y3) / 3
                    center_conf = min(conf1, conf2, conf3)  # Use minimum confidence
                    return center_x, center_y, center_conf
            return None, None, 0.0
        
        # Map keypoints using original IDs from dict and convert to sequential indices
        output_idx = 0
        for halpe_id in HALPE_TO_COCO_KEYPOINT_MAP.keys():
            coco_id = HALPE_TO_COCO_KEYPOINT_MAP[halpe_id]
            
            if coco_id is None:  # These are calculated keypoints (17, 18, 19)
                if halpe_id == 17:  # head_top - midpoint of left_eye and right_eye
                    x, y, conf = calculate_midpoint(1, 2)  # left_eye, right_eye
                    keypoints_flat[output_idx * 3] = x if x is not None else 0.0
                    keypoints_flat[output_idx * 3 + 1] = y if y is not None else 0.0
                    keypoints_flat[output_idx * 3 + 2] = conf
                elif halpe_id == 18:  # neck - center of nose, left_shoulder, right_shoulder
                    x, y, conf = calculate_center_of_three(0, 5, 6)  # nose, left_shoulder, right_shoulder
                    keypoints_flat[output_idx * 3] = x if x is not None else 0.0
                    keypoints_flat[output_idx * 3 + 1] = y if y is not None else 0.0
                    keypoints_flat[output_idx * 3 + 2] = conf
                elif halpe_id == 19:  # pelvis_point - midpoint of left_hip and right_hip
                    x, y, conf = calculate_midpoint(11, 12)  # left_hip, right_hip
                    keypoints_flat[output_idx * 3] = x if x is not None else 0.0
                    keypoints_flat[output_idx * 3 + 1] = y if y is not None else 0.0
                    keypoints_flat[output_idx * 3 + 2] = conf
            else:
                # Map from original keypoint using the COCO ID
                if coco_id in keypoint_dict:
                    x, y, conf = keypoint_dict[coco_id]
                    keypoints_flat[output_idx * 3] = x
                    keypoints_flat[output_idx * 3 + 1] = y
                    keypoints_flat[output_idx * 3 + 2] = conf
            
            output_idx += 1
        
        confident_scores = [kp["confidence"] for kp in detection["keypoints"] if kp["confidence"] > 0.15]
        overall_score = sum(confident_scores) / len(confident_scores) if confident_scores else 0.0
        
        halpe_detection = {
            "image_id": image_filename,
            "category_id": 1,
            "keypoints": keypoints_flat,
            "score": overall_score,
            "box": bbox_halpe_format,
            "idx": [float(detection["id"])]
        }
        
        halpe_pose_data.append(halpe_detection)
    
    return halpe_pose_data

def COCO2Halpe(input_json_path, output_json_path, video_name=None):
    with open(input_json_path, 'r') as f:
        frames_data = json.load(f)
    
    if video_name is None:
        video_name = os.path.splitext(os.path.basename(input_json_path))[0]
    
    all_converted_data = []
    
    for frame_data in tqdm(frames_data, desc="Converting data to Halpe Pose format", unit="frame"):
        frame_id = frame_data.get('frame_id', 0)
        image_filename = f"{video_name}_frame_{frame_id:06d}.jpg"
        converted_frame_data = convert_to_halpe_pose_format(frame_data, image_filename)
        all_converted_data.extend(converted_frame_data)
    
    with open(output_json_path, 'w') as f:
        json.dump(all_converted_data, f, indent=2)
    return output_json_path

def get_coco_to_h36m_mapping():
    """
    Map COCO keypoints to H36M format
    """
    
    # Create mapping from COCO to H36M
    coco_to_h36m = np.zeros(17, dtype=int)
    
    # Map joints
    coco_to_h36m[0] = 0   # COCO nose -> H36M Hip (approximation - will be computed as hip center)
    coco_to_h36m[1] = 12  # COCO right_hip -> H36M RHip
    coco_to_h36m[2] = 14  # COCO right_knee -> H36M RKnee
    coco_to_h36m[3] = 16  # COCO right_ankle -> H36M RFoot
    coco_to_h36m[4] = 11  # COCO left_hip -> H36M LHip
    coco_to_h36m[5] = 13  # COCO left_knee -> H36M LKnee
    coco_to_h36m[6] = 15  # COCO left_ankle -> H36M LFoot
    coco_to_h36m[7] = 0   # H36M Spine (will be computed)
    coco_to_h36m[8] = 0   # H36M Thorax (will be computed)
    coco_to_h36m[9] = 0   # COCO nose -> H36M Neck/Nose
    coco_to_h36m[10] = 0  # H36M Head (will be computed from nose)
    coco_to_h36m[11] = 5  # COCO left_shoulder -> H36M LShoulder
    coco_to_h36m[12] = 7  # COCO left_elbow -> H36M LElbow
    coco_to_h36m[13] = 9  # COCO left_wrist -> H36M LWrist
    coco_to_h36m[14] = 6  # COCO right_shoulder -> H36M RShoulder
    coco_to_h36m[15] = 8  # COCO right_elbow -> H36M RElbow
    coco_to_h36m[16] = 10 # COCO right_wrist -> H36M RWrist
    
    return coco_to_h36m


def coco2h36m(filepath):
    """Load 2D keypoints from file (NPZ or JSON format)"""
    filepath = Path(filepath)
    
    if filepath.suffix == '.npz':
        data = np.load(filepath)
        keypoints = data['keypoints']  # Shape: (N, 17, 2) or (N, 17, 3)
    elif filepath.suffix == '.json':
        with open(filepath, 'r') as f:
            data = json.load(f)
        keypoints_list = []
        # Handle new format with frame_id and detections
        if isinstance(data, list) and len(data) > 0:
            # Sort by frame_id to ensure correct temporal order
            data_sorted = sorted(data, key=lambda x: x['frame_id'])
            for frame_data in data_sorted:
                if 'detections' in frame_data and len(frame_data['detections']) > 0:
                    # Take the first detection (assuming single person)
                    detection = frame_data['detections'][0]
                    keypoints_dict = {}
                    for kpt in detection['keypoints']:
                        keypoints_dict[kpt['id']] = [kpt['x'], kpt['y'], kpt['confidence']]
                    # Create array in order (0-16 for COCO format)
                    frame_keypoints = []
                    for i in range(17):  # COCO has 17 body keypoints
                        if i in keypoints_dict:
                            frame_keypoints.append(keypoints_dict[i][:2])  # x, y only
                        else:
                            # Handle missing keypoints
                            frame_keypoints.append([0.0, 0.0])
                    keypoints_list.append(frame_keypoints)
            
            keypoints = np.array(keypoints_list)  # Shape: (N, 17, 2)
            print(f"Loaded {len(keypoints)} frames from new JSON format")

            coco_keypoints = keypoints
            N = coco_keypoints.shape[0]
            h36m_keypoints = np.zeros((N, 17, 2), dtype=coco_keypoints.dtype)
            
            # Direct mappings
            h36m_keypoints[:, 1] = coco_keypoints[:, 12]   # RHip
            h36m_keypoints[:, 2] = coco_keypoints[:, 14]   # RKnee
            h36m_keypoints[:, 3] = coco_keypoints[:, 16]   # RFoot
            h36m_keypoints[:, 4] = coco_keypoints[:, 11]   # LHip
            h36m_keypoints[:, 5] = coco_keypoints[:, 13]   # LKnee
            h36m_keypoints[:, 6] = coco_keypoints[:, 15]   # LFoot
            h36m_keypoints[:, 11] = coco_keypoints[:, 5]   # LShoulder
            h36m_keypoints[:, 12] = coco_keypoints[:, 7]   # LElbow
            h36m_keypoints[:, 13] = coco_keypoints[:, 9]   # LWrist
            h36m_keypoints[:, 14] = coco_keypoints[:, 6]   # RShoulder
            h36m_keypoints[:, 15] = coco_keypoints[:, 8]   # RElbow
            h36m_keypoints[:, 16] = coco_keypoints[:, 10]  # RWrist
            h36m_keypoints[:, 0] = (coco_keypoints[:, 11] + coco_keypoints[:, 12]) / 2  # Hip center (average of left and right hips)
            h36m_keypoints[:, 8] = (coco_keypoints[:, 5] + coco_keypoints[:, 6]) / 2  # Thorax (average of left and right shoulders) 
            h36m_keypoints[:, 7] = (h36m_keypoints[:, 0] + h36m_keypoints[:, 8]) / 2  # Spine (midpoint between hip center and thorax)
            h36m_keypoints[:, 9] = coco_keypoints[:, 0]   # Neck/Nose
            h36m_keypoints[:, 10] = coco_keypoints[:, 0]  # Head (slightly above nose)

            keypoints = h36m_keypoints
            print("Converted COCO keypoints to H36M format")
    else:
        raise ValueError(f"Unsupported file format: {filepath.suffix}")
    
    # Ensure (N, 17, 2) shape
    if keypoints.shape[-1] == 3:
        keypoints = keypoints[..., :2]  # Remove confidence scores if present
    
    return keypoints

def add_3d_keypoints(frame_data, npy_data):
    """
    Add 3D keypoints to frame_data structure
    """
    for frame_idx, frame in enumerate(frame_data):
        if frame_idx < npy_data.shape[0]:
            keypoints_3d = npy_data[frame_idx]  # Shape: (17, 3)
            for detection in frame["detections"]:
                detection["keypoints3D"] = []
                
                # Add each 3D keypoint
                for keypoint_idx in range(17):
                    x, y, z = keypoints_3d[keypoint_idx]
                    detection["keypoints3D"].append({
                        "id": keypoint_idx,
                        "x": float(x),
                        "y": float(y),
                        "z": float(z),
                        "name": HUMAN26M[keypoint_idx]
                    })
    return frame_data
