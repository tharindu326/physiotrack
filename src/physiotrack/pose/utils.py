from .config import COCO, COCO_WHOLEBODY


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


def convert_to_alpha_pose_format(frame_data, image_filename):
    HALPE_KEYPOINT_DICT = {
        0: "nose",
        1: "left_eye",
        2: "right_eye",
        3: "left_ear",
        4: "right_ear",
        5: "left_shoulder",
        6: "right_shoulder",
        7: "left_elbow",
        8: "right_elbow",
        9: "left_wrist",
        10: "right_wrist",
        11: "left_hip",
        12: "right_hip",
        13: "left_knee",
        14: "right_knee",
        15: "left_ankle",
        16: "right_ankle",
        
        17: "head_top",  # Calculate as middle of left_eye and right_eye 
        18: "neck",  # Calculate as center of nose, left_shoulder, right_shoulder
        19: "pelvis_point", # Calculate as middle of left_hip and right_hip

        20: "left_big_toe",    # 17
        21: "right_big_toe",   # 20
        22: "left_small_toe",  # 18
        23: "right_small_toe", # 21
        
        24: "left_heel",       # 19
        25: "right_heel"       # 22 (fixed typo: "heal" -> "heel")
    }
    
    HALPE_TO_COCO_KEYPOINT_MAP = {
        0: 0,   # nose -> nose
        1: 1,   # left_eye -> left_eye
        2: 2,   # right_eye -> right_eye
        3: 3,   # left_ear -> left_ear
        4: 4,   # right_ear -> right_ear
        5: 5,   # left_shoulder -> left_shoulder
        6: 6,   # right_shoulder -> right_shoulder
        7: 7,   # left_elbow -> left_elbow
        8: 8,   # right_elbow -> right_elbow
        9: 9,   # left_wrist -> left_wrist
        10: 10, # right_wrist -> right_wrist
        11: 11, # left_hip -> left_hip
        12: 12, # right_hip -> right_hip
        13: 13, # left_knee -> left_knee
        14: 14, # right_knee -> right_knee
        15: 15, # left_ankle -> left_ankle
        16: 16, # right_ankle -> right_ankle
        17: None,  # head_top - calculated
        18: None,  # neck - calculated
        19: None,  # pelvis_point - calculated
        20: 17,    # left_big_toe
        21: 20,    # right_big_toe
        22: 18,    # left_small_toe
        23: 21,    # right_small_toe
        24: 19,    # left_heel
        25: 22,    # right_heel
    }
    
    alpha_pose_data = []
    
    if "detections" not in frame_data or not frame_data["detections"]:
        return alpha_pose_data
    
    for detection in frame_data["detections"]:
        x1, y1, x2, y2 = detection["bbox"]
        bbox_alpha_format = [x1, y1, x2 - x1, y2 - y1]
        
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
        
        # Calculate overall confidence score
        confident_scores = [kp["confidence"] for kp in detection["keypoints"] if kp["confidence"] > 0.15]
        overall_score = sum(confident_scores) / len(confident_scores) if confident_scores else 0.0
        
        alpha_detection = {
            "image_id": image_filename,
            "category_id": 1,
            "keypoints": keypoints_flat,
            "score": overall_score,
            "box": bbox_alpha_format,
            "idx": [float(detection["id"])]
        }
        
        alpha_pose_data.append(alpha_detection)
    
    return alpha_pose_data

