import os
import cv2
from . import YoloPose
from . import VitInference
from . import SapiensPoseEstimation
from . import SapiensSegmentation, draw_segmentation_map
from . import Detector
from . import Segmentor

class Inf:
    def __init__(self):
        pass
from . import cfg
from . import filter_by_box, process_segmentation_map
import numpy as np
import time
from . import Tracker
from tqdm import tqdm
script_dir = os.path.dirname(os.path.realpath(__file__))


class Inference:
    def __init__(self, post_process=True):
        self.get_models()
        self.post_process = post_process
        if not self.post_process:
            print('== Runnnig without potprocessing for the segmentation map ==')
        self.tracker = Tracker(cfg)
        
    def get_models(self):
        pose_framework = cfg.pose.model.__class__.__name__
        segmentation_framework_sapiens = cfg.SapiensSegmentation.model.__class__.__name__
        segmentation_framework_yolo = cfg.VRsegmentation.model.__class__.__name__
        detection_framework = cfg.detection.model.__class__.__name__
        self.COLORS = cfg.general.COLORS
        
        print(f'Initiating {detection_framework} {cfg.detection.model.name} for student detection')
        self.detector = Detector()
        
        if cfg.pose.enable:
            print(f'Initiating {pose_framework} {cfg.pose.model.name} for the Pose estimation')
            if pose_framework == 'ViTPose':
                self.pose_estimator = VitInference(model=os.path.join(cfg.general.model_path, cfg.pose.model.value), 
                                        yolo=os.path.join(cfg.general.model_path, cfg.detection.model.value), 
                                        model_name=str(cfg.pose.model.name.split("_")[0]),
                                        device=cfg.general.device)
            elif pose_framework == 'YOLO':
                self.pose_estimator = YoloPose()
            elif pose_framework == 'Sapiens':
                self.pose_estimator = SapiensPoseEstimation()
            else:
                raise ValueError("Invalid model type. Please check the configuration")
        
        if cfg.SapiensSegmentation.enable:
            print(f'Initiating {segmentation_framework_sapiens} {cfg.SapiensSegmentation.model.name} for the Body Segmentation')
            if segmentation_framework_sapiens == 'Sapiens':
                self.SapiensSegmentor = SapiensSegmentation()
            else:
                raise ValueError("Invalid model type. Please check the configuration")
        
        if cfg.VRsegmentation.enable:
            print(f'Initiating {segmentation_framework_yolo} {cfg.VRsegmentation.model.name} for the Body Segmentation')
            if segmentation_framework_yolo == 'YOLO':
                self.segmentor = Segmentor(model=os.path.join(cfg.general.model_path, cfg.VRsegmentation.model.value))
            else:
                raise ValueError("Invalid model type. Please check the configuration")
    
    def plot_class_contours(self, segmentation_map, bbox=None):
        black_canvas = np.zeros_like(segmentation_map, dtype=np.uint8)
        black_canvas = cv2.cvtColor(black_canvas, cv2.COLOR_GRAY2BGR)
        # Iterate over each unique class in the segmentation map
        unique_classes = np.unique(segmentation_map)
        for cls in unique_classes:
            if cls == 0:
                continue
            # Create a binary mask for the current class
            mask = np.uint8(segmentation_map == cls) * 255
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for idx, contour in enumerate(contours):
                cv2.drawContours(black_canvas, [contour], -1, (0, 255, 0), 1)
                M = cv2.moments(contour)
                if M["m00"] != 0:
                    cX = int(M["m10"] / M["m00"])
                    cY = int(M["m01"] / M["m00"])
                    cv2.putText(black_canvas, str(int(cls)), (cX, cY), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        if bbox is not None:     
            x1, y1, x2, y2 = bbox[0]
            # cv2.rectangle(black_canvas, (x1, y1), (x2, y2), (0, 255, 0), 2)
        return black_canvas
    
    def infer(self, frame):
        # Running pose estimation and segmentation
        frame_with_pose = None
        frame_segmented = None
        segmentation_map = None
        segmentation_map_uint8 = None
        VR_segmentation_map = None
        
        frame_data = {"detections": []}
        results = self.detector.detect(frame.copy())
        detections = results[0].boxes.data.cpu().numpy()  # (x1, y1, x2, y2, conf, cls)
        boxes = detections[:, :-2].astype(int)
        
        if cfg.tracker.enable:
            frame_with_track, online_targets = self.tracker.track(frame.copy(), detections)
        
            # filter boxes based on student track to obtain only the student box
            if self.tracker.student_track_id is not None and self.tracker.last_known_bbox is not None:
                # student_box = self.tracker.student_track_history[self.tracker.student_track_id][-1]
                boxes = np.array(self.tracker.last_known_bbox, dtype=int).reshape(1, -1) # student_box = [x1, y1, x2, y2]
            else:
                boxes = []
                
        if cfg.pose.enable:
            frame_with_pose, frame_data = self.pose_estimator.inference(frame_with_track, boxes)
            
        if cfg.flags.overlay_track:
                frame_with_pose = self.tracker.draw_tracks(frame_with_pose)

        if cfg.VRsegmentation.enable:
            segmentation_img_yolo, VR_segmentation_map = self.segmentor.segment(frame)
            
        if cfg.SapiensSegmentation.enable:
            if len(boxes) == 0:
                boxes = None
            segmentation_map = self.SapiensSegmentor.inference(frame.copy())
            start = time.perf_counter()
            segmentation_map = filter_by_box(segmentation_map, bbox=boxes)
            
            # black_canvas = self.plot_class_contours(segmentation_map, boxes)
            # output_path = "results/output_frame_004_black_canvas_before_process.jpg"
            # cv2.imwrite(output_path, black_canvas)
            # if cfg.sam_post_process.enable:
            if self.post_process:
                segmentation_map = process_segmentation_map(segmentation_map, 
                                                            exclude=cfg.sam_post_process.exclude, 
                                                            combine=cfg.sam_post_process.combine, 
                                                            min_area=cfg.sam_post_process.min_area, 
                                                            max_area=cfg.sam_post_process.max_area, 
                                                            exclude_in_area_filtering=cfg.sam_post_process.exclude_in_area_filtering)
            
            # np.save("BV_S17_cut1_segmentation_map.npy", segmentation_map)
            # black_canvas = self.plot_class_contours(segmentation_map, boxes)
            # output_path = "results/output_frame_004_black_canvas_after_process.jpg"
            # cv2.imwrite(output_path, black_canvas)
            
            # combine seg map 
            if VR_segmentation_map is not None:
                if len(cfg.SapiensSegmentation.preserve_classes) > 0:
                    #  IDs that should always be preserved. Ex face neck 
                    # mask where the preserved classes exist in segmentation_map
                    preserve_mask = np.isin(segmentation_map, cfg.SapiensSegmentation.preserve_classes)
                    # Update segmentation_map while keeping the preserve classes unchanged
                    segmentation_map = np.where(preserve_mask, segmentation_map, np.maximum(segmentation_map, VR_segmentation_map))
                    segmentation_map = np.maximum(segmentation_map, VR_segmentation_map)
                    
                else:
                    segmentation_map = np.maximum(segmentation_map, VR_segmentation_map)
                
            segmentation_img = draw_segmentation_map(segmentation_map)
            
            alpha = 0.6  # More weight to the original frame (less transparency)
            beta = 0.4   # Less weight to the segmentation map
            gamma = 0    # No brightness adjustment
            frame_segmented = cv2.addWeighted(frame.copy(), alpha, segmentation_img, beta, gamma)
            
            print(f"Segmentation Postprocessing took: {time.perf_counter() - start:.4f} seconds")
            if segmentation_map is not None:
                segmentation_map_uint8 = segmentation_map.astype(np.uint8)
            
        # If Sapiens segmentation is disabled but VR segmentation is enabled,
        # use the VR segmentation map directly for visualization.
        elif VR_segmentation_map is not None:
            alpha = 0.6
            beta = 0.4
            gamma = 0
            frame_segmented = cv2.addWeighted(frame.copy(), alpha, segmentation_img_yolo, beta, gamma)
            segmentation_map_uint8 = VR_segmentation_map.astype(np.uint8)
            
        # Merging both results into a single frame
        combined_frame = None
        if frame_segmented is not None:
            if frame_with_pose is not None:
                combined_frame = self.create_combined_view(frame_with_pose, frame_segmented)
            else:
                combined_frame = frame_segmented
        elif frame_with_pose is not None:
            if frame_segmented is not None:
                combined_frame = self.create_combined_view(frame_with_pose, frame_segmented)
            else:
                combined_frame = frame_with_pose
        return frame_with_pose, frame_data, frame_segmented, combined_frame, segmentation_map_uint8

    def create_combined_view(self, pose_image, segmentation_image):
        # Resize images to match in height if they differ
        height = max(pose_image.shape[0], segmentation_image.shape[0])
        width = pose_image.shape[1] + segmentation_image.shape[1]
        # Resize pose and segmentation images to same height and width
        pose_resized = cv2.resize(pose_image, (width // 2, height))
        segmentation_resized = cv2.resize(segmentation_image, (width // 2, height))
        # Concatenate images
        combined_image = np.hstack((pose_resized, segmentation_resized))
        return combined_image


if __name__ == '__main__':
    inference_instance = Inference()
    im_path = 'samples/output_frame_004.png'
    frame = cv2.imread(im_path)
    frame_with_pose, keypoints, segmentation_img, combined_frame, segmentation_map = inference_instance.infer(frame)
    os.makedirs('results', exist_ok=True)
    output_path = f"results/predicted_{im_path.split('/')[-1]}"
    cv2.imwrite(output_path, combined_frame)
    
    # input_folder = 'S2_to_S14_GT_Segmentation_HIAugmentedCare/segmentation_GT_images'
    # output_folder = 'S2_to_S14_GT_Segmentation_HIAugmentedCare/yolov11-new-preds'
    # os.makedirs(output_folder, exist_ok=True)

    # selects = {"vr": 28, "face_neck": 29}
    # inference_instance = Inference()
    
    # image_files = [f for f in os.listdir(input_folder) if f.endswith(('.png', '.jpg', '.jpeg'))]
    # total_images = len(image_files)

    # with tqdm(total=total_images, desc="Processing images", unit="image") as pbar:
    #     for idx, image_name in enumerate(image_files, start=1):
    #         image_path = os.path.join(input_folder, image_name)
    #         frame = cv2.imread(image_path)

    #         _, _, _, _, segmentation_map = inference_instance.infer(frame)

    #         if segmentation_map is not None:
    #             for class_name, class_id in selects.items():
    #                 mask = (segmentation_map == class_id).astype(np.uint8) * 255

    #                 mask_filename = f"{os.path.splitext(image_name)[0]}_mask_{class_name}.jpg"
    #                 mask_output_path = os.path.join(output_folder, mask_filename)

    #                 cv2.imwrite(mask_output_path, mask)
            
    #         pbar.set_description(f"Processing frame {idx}/{total_images}")
    #         pbar.update(1)