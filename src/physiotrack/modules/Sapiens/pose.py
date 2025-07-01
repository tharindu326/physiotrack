from typing import List
import time
import cv2
import numpy as np
from torchvision import transforms
import torch
import os
import matplotlib.pyplot as plt
from .classes_and_palettes import (
    COCO_KPTS_COLORS,
    COCO_WHOLEBODY_KPTS_COLORS,
    GOLIATH_KPTS_COLORS,
    GOLIATH_SKELETON_INFO,
    GOLIATH_KEYPOINTS
)


class SapiensPoseEstimation:
    def __init__(self,
                 model, 
                 device,
                 dtype: torch.dtype = torch.float32,
                 mean: List[float] = (0.485, 0.456, 0.406),
                 std: List[float] = (0.229, 0.224, 0.225),
                 ):
        # Load the model
        self.device = device
        self.dtype = dtype
        model_folder = os.path.join(os.path.dirname(__file__), '..', 'model_data')
        model_path = os.path.join(model_folder, model.value)
        self.model = torch.jit.load(model_path).eval().to(self.device).to(dtype)
        self.preprocessor = transforms.Compose([transforms.ToPILImage(),
                               transforms.Resize((1024,768)),
                               transforms.ToTensor(),
                               transforms.Normalize(mean=mean, std=std),
                               ])

    @torch.inference_mode()
    def estimate_pose(self, img, bboxes):
        start = time.perf_counter()
        all_keypoints = []
        result_img = img.copy()
        frame_data = {"detections": []}
        
        for i, bbox in enumerate(bboxes):
            cropped_img = self.crop_image(img, bbox)
            tensor = self.preprocessor(cropped_img).unsqueeze(0).to(self.device).to(self.dtype)
            heatmaps = self.model(tensor)
            keypoints = self.heatmaps_to_keypoints(heatmaps[0].cpu().numpy())
    
            keypoints_list = []
            
            # Draw the keypoints on the original image (optional)
            result_img = self.draw_keypoints(result_img, keypoints, bbox)
            
            x1, y1, x2, y2 = map(int, bbox[:4])
            bbox_width, bbox_height = x2 - x1, y2 - y1
            for idx, (name, (x, y, conf)) in enumerate(keypoints.items()):
                if conf > 0.15:  # Only draw confident keypoints
                    x_coord = (x * bbox_width / 192) + x1
                    y_coord = (y * bbox_height / 256) + y1
                    keypoints_list.append({"id": idx, "x": float(x_coord), "y": float(y_coord), "confidence": float(conf)})

            # Store detection info for this bbox
            detection = {
                "id": i,
                "bbox": [x1, y1, x2, y2],  # The original bbox for the detection
                "keypoints": keypoints_list
            }
            frame_data["detections"].append(detection)
        # print(f"Sapiens Pose inference took: {time.perf_counter() - start:.4f} seconds")
        return result_img, frame_data

    def crop_image(self, img: np.ndarray, bbox: List[float]) -> np.ndarray:
        x1, y1, x2, y2 = map(int, bbox[:4])
        return img[y1:y2, x1:x2]


    def heatmaps_to_keypoints(self, heatmaps: np.ndarray) -> dict:
        keypoints = {}
        for i, name in enumerate(GOLIATH_KEYPOINTS):
            if i < heatmaps.shape[0]:
                y, x = np.unravel_index(np.argmax(heatmaps[i]), heatmaps[i].shape)
                conf = heatmaps[i, y, x]
                keypoints[name] = (float(x), float(y), float(conf))
        return keypoints


    def draw_keypoints(self, img: np.ndarray, keypoints: dict, bbox: List[float]) -> np.ndarray:
        x1, y1, x2, y2 = map(int, bbox[:4])
        bbox_width, bbox_height = x2 - x1, y2 - y1
        img_copy = img.copy()

        # Draw keypoints on the image
        for i, (name, (x, y, conf)) in enumerate(keypoints.items()):
            if conf > 0.3:  # Only draw confident keypoints
                x_coord = int(x * bbox_width / 192) + x1
                y_coord = int(y * bbox_height / 256) + y1
                cv2.circle(img_copy, (x_coord, y_coord), 3, GOLIATH_KPTS_COLORS[i], -1)

        # Optionally draw skeleton
        for _, link_info in GOLIATH_SKELETON_INFO.items():
            pt1_name, pt2_name = link_info['link']
            if pt1_name in keypoints and pt2_name in keypoints:
                pt1 = keypoints[pt1_name]
                pt2 = keypoints[pt2_name]
                if pt1[2] > 0.3 and pt2[2] > 0.3:
                    x1_coord = int(pt1[0] * bbox_width / 192) + x1
                    y1_coord = int(pt1[1] * bbox_height / 256) + y1
                    x2_coord = int(pt2[0] * bbox_width / 192) + x1
                    y2_coord = int(pt2[1] * bbox_height / 256) + y1
                    cv2.line(img_copy, (x1_coord, y1_coord), (x2_coord, y2_coord), GOLIATH_KPTS_COLORS[i], 2)

        return img_copy
    
    def inference(self, img, bboxes):
        # Process the image and estimate the pose
        pose_result_image, keypoints_frame_data = self.estimate_pose(img, bboxes)
        return pose_result_image, keypoints_frame_data

if __name__ == "__main__":
    from enum import Enum
    class Pose:
        class Sapiens(Enum):
            # COCO wholebody
            B1_TS_COCOHB = "sapiens_1b_coco_wholebody_best_coco_wholebody_AP_727_torchscript.pt2"  # https://huggingface.co/noahcao/sapiens-pose-coco/resolve/main/sapiens_lite_host/torchscript/pose/checkpoints/sapiens_1b/sapiens_1b_coco_wholebody_best_coco_wholebody_AP_727_torchscript.pt2?download=true
            B06_TS_COCOHB = "sapiens_0.6b_coco_wholebody_best_coco_wholebody_AP_695_torchscript.pt2"  # https://huggingface.co/noahcao/sapiens-pose-coco/resolve/main/sapiens_lite_host/torchscript/pose/checkpoints/sapiens_0.6b/sapiens_0.6b_coco_wholebody_best_coco_wholebody_AP_695_torchscript.pt2?download=true
            B03_TS_COCOHB = "sapiens_0.3b_coco_wholebody_best_coco_wholebody_AP_620_torchscript.pt2"  # https://huggingface.co/noahcao/sapiens-pose-coco/resolve/main/sapiens_lite_host/torchscript/pose/checkpoints/sapiens_0.3b/sapiens_0.3b_coco_wholebody_best_coco_wholebody_AP_620_torchscript.pt2?download=true
            
            # 308
            B1_TS = "sapiens_1b_goliath_best_goliath_AP_639_torchscript.pt2"  # https://huggingface.co/facebook/sapiens-pose-1b-torchscript/resolve/main/sapiens_1b_goliath_best_goliath_AP_639_torchscript.pt2?download=true
            B06_TS = "sapiens_0.6b_goliath_best_goliath_AP_609_torchscript.pt2"  # https://huggingface.co/facebook/sapiens-pose-0.6b-torchscript/resolve/main/sapiens_0.6b_goliath_best_goliath_AP_609_torchscript.pt2?download=true
            B03_TS = "sapiens_0.3b_goliath_best_goliath_AP_573_torchscript.pt2"  # https://huggingface.co/facebook/sapiens-pose-0.3b-torchscript/resolve/main/sapiens_0.3b_goliath_best_goliath_AP_573_torchscript.pt2?download=true
            B1_BF16 = "sapiens_1b_goliath_best_goliath_AP_639_bfloat16.pt2"  # https://huggingface.co/facebook/sapiens-pose-1b-bfloat16/resolve/main/sapiens_1b_goliath_best_goliath_AP_639_bfloat16.pt2?download=true
            B06_BF16 = "sapiens_0.6b_goliath_best_goliath_AP_609_bfloat16.pt2"  # https://huggingface.co/facebook/sapiens-pose-0.6b-bfloat16/resolve/main/sapiens_0.6b_goliath_best_goliath_AP_609_bfloat16.pt2?download=true
            B03_BF16 = "sapiens_0.3b_goliath_best_goliath_AP_573_bfloat16.pt2"  # https://huggingface.co/facebook/sapiens-pose-0.3b-bfloat16/resolve/main/sapiens_0.3b_goliath_best_goliath_AP_573_bfloat16.pt2?download=true
    model = Pose.Sapiens.B1_TS
    device = 'cpu'
    pose_estimator = SapiensPoseEstimation(model, device)
    images = ['samples/BV_S17_frame_1.jpg', 'samples/BV_S17_frame_2.jpg', 'samples/BV_S17_frame_3.jpg']
    output_path = 'test_path'
    os.makedirs(output_path, exist_ok=True)
    for img_path in images:
        output_path = os.path.join(output_path, f"pose_estimated_single_person_{model.name}_{img_path.split('/')[-1]}")
        img = cv2.imread(img_path)
        h, w, _ = img.shape
        img = img[0:h, 0:700]
        # Perform pose estimation
        start_time = time.perf_counter()
        result_img, keypoints = pose_estimator(img)
        cv2.imwrite(output_path, result_img)
        print(f"Time taken: {time.perf_counter() - start_time:.4f} seconds")