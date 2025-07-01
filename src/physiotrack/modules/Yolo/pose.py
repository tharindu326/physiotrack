
from ultralytics import YOLO
import os 
from .classes_and_palettes import COLORS
import cv2
import numpy as np
import time


class YoloPose:
    def __init__(self, model, device, OBJECTNESS_CONFIDENCE, NMS_THRESHOLD, classes, overlay_keypoints=True, 
                 render_labels=False, render_box_detections=False, verbose=False, **kwargs):
        model_path = os.path.join(os.path.dirname(__file__), '..', 'model_data')
        self.model = YOLO(os.path.join(model_path, model.value))
        self.device = device
        self.conf = OBJECTNESS_CONFIDENCE
        self.iou = NMS_THRESHOLD
        self.classes = classes
        self.verbose = verbose
        self.COLORS = COLORS
        self.render_box_detections = render_box_detections
        self.render_labels = render_labels
        self.overlay_keypoints = overlay_keypoints
        self.extra_args = kwargs 
        
    def compute_iou(self, boxA, boxB):
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])

        interArea = max(0, xB - xA) * max(0, yB - yA)
        boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
        boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
        
        iou = interArea / float(boxAArea + boxBArea - interArea)
        return iou
    
    def inference(self, frame, boxes=None, **kwargs):
        """
        YOLO-pose inference with flexibility for extra prediction parameters.
        If `boxes` is provided, only return pose for those corresponding bounding boxes.
        """
        processed_boxes = []
        processed_confidence = []
        processed_class_id = []
        names = []
        keypoints = []
        start = time.perf_counter()
        frame_data = {"detections": []}

        all_kwargs = {
            "conf": self.conf,
            "iou": self.iou,
            "classes": self.classes,
            "device": self.device,
            "verbose": self.verbose,
            **self.extra_args,
            **kwargs,
        }

        results = self.model.predict(source=frame, **all_kwargs)

        for result in results:
            pose = result.keypoints
            detected_boxes = np.array(result.boxes.xywh.tolist())
            class_ids = result.boxes.cls.tolist()
            confidences = result.boxes.conf.tolist()

            if detected_boxes.size == 0:
                continue

            # Convert (cx, cy, w, h) -> (x1, y1, x2, y2)
            # detected_boxes[:, 0] -= detected_boxes[:, 2] / 2
            # detected_boxes[:, 1] -= detected_boxes[:, 3] / 2
            # detected_boxes[:, 2] = detected_boxes[:, 0] + detected_boxes[:, 2]
            # detected_boxes[:, 3] = detected_boxes[:, 1] + detected_boxes[:, 3]
            
            detected_boxes[:, 0] = detected_boxes[:, 0] - detected_boxes[:, 2] / 2
            detected_boxes[:, 1] = detected_boxes[:, 1] - detected_boxes[:, 3] / 2
            detected_boxes[:, 2] = detected_boxes[:, 0] + detected_boxes[:, 2]
            detected_boxes[:, 3] = detected_boxes[:, 1] + detected_boxes[:, 3]

            selected_indices = []

            if boxes is not None:
                # Use IoU to match detections to input boxes
                for j, target_box in enumerate(boxes):
                    max_iou = -1
                    best_index = -1
                    for i, det_box in enumerate(detected_boxes):
                        iou = self.compute_iou(det_box, target_box)
                        if iou > max_iou:
                            max_iou = iou
                            best_index = i
                    if best_index != -1:
                        selected_indices.append(best_index)
            else:
                selected_indices = list(range(len(detected_boxes)))

            for i in selected_indices:
                box = detected_boxes[i]
                class_id = int(class_ids[i])
                confidence = confidences[i]

                p1, p2 = (int(box[0]), int(box[1])), (int(box[2]), int(box[3]))
                line_width = max(round(sum(frame.shape) / 2 * 0.003), 2)
                color = self.COLORS[list(self.COLORS)[class_id % len(self.COLORS)]]

                if self.render_box_detections:
                    cv2.rectangle(frame, p1, p2, color, thickness=line_width, lineType=cv2.LINE_AA)
                if self.render_labels:
                    label = f"{self.model.names[class_id]}: {confidence:.4f}"
                    cv2.putText(frame, label, (int(box[0]), int(box[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 2, color, line_width)

                processed_boxes.append([box[0], box[1], box[2] - box[0], box[3] - box[1]])
                processed_confidence.append(confidence)
                processed_class_id.append(class_id)

                keypoints_list = []
                if pose is not None and i < len(pose.xy):
                    kpts = pose.xy[i].tolist()
                    conf = pose.conf[i].tolist()

                    keypoints_list = [
                        {"id": kp_idx, "x": float(kpt[0]), "y": float(kpt[1]), "confidence": float(conf[kp_idx])}
                        for kp_idx, (kpt, confidence) in enumerate(zip(kpts, conf))
                        if kpt[0] != 0 and kpt[1] != 0
                    ]

                detection = {
                    "id": i,
                    "bbox": [box[0], box[1], box[2] - box[0], box[3] - box[1]],
                    "keypoints": keypoints_list,
                }
                frame_data["detections"].append(detection)

                if pose is not None and self.overlay_keypoints:
                    frame = self.plot_skeleton_kpts(frame, [pose.xy[i].tolist()])

        print(f"YOLO Pose inference took: {time.perf_counter() - start:.4f} seconds")
        return frame, frame_data

    
    def plot_skeleton_kpts(self, im, all_kpts):
        """
        Plot skeleton keypoints on the image.
        """
        palette = np.array([[255, 128, 0], [255, 153, 51], [255, 178, 102],
                            [230, 230, 0], [255, 153, 255], [153, 204, 255],
                            [255, 102, 255], [255, 51, 255], [102, 178, 255],
                            [51, 153, 255], [255, 153, 153], [255, 102, 102],
                            [255, 51, 51], [153, 255, 153], [102, 255, 102],
                            [51, 255, 51], [0, 255, 0], [0, 0, 255], [255, 0, 0],
                            [255, 255, 255]])

        skeleton = [[16, 14], [14, 12], [17, 15], [15, 13], [12, 13], [6, 12],
                    [7, 13], [6, 7], [6, 8], [7, 9], [8, 10], [9, 11], [2, 3],
                    [1, 2], [1, 3], [2, 4], [3, 5], [4, 6], [5, 7]]

        pose_limb_color = palette[[9, 9, 9, 9, 7, 7, 7, 0, 0, 0, 0, 0, 16, 16, 16, 16, 16, 16, 16]]
        pose_kpt_color = palette[[16, 16, 16, 16, 16, 0, 0, 0, 0, 0, 0, 9, 9, 9, 9, 9, 9]]
        radius = 5

        # Iterate over each set of keypoints
        for kpts in all_kpts:
            for kid, kpt in enumerate(kpts):
                if kpt[0] == 0 and kpt[1] == 0:
                    continue  # Skip keypoints that are [0.0, 0.0]

                r, g, b = pose_kpt_color[kid % len(pose_kpt_color)]
                cv2.circle(im, (int(kpt[0]), int(kpt[1])), radius, (int(r), int(g), int(b)), -1)

            # Draw skeleton connections
            for sk_id, sk in enumerate(skeleton):
                pos1 = kpts[sk[0] - 1]
                pos2 = kpts[sk[1] - 1]

                if (pos1[0] == 0 and pos1[1] == 0) or (pos2[0] == 0 and pos2[1] == 0):
                    continue  # Skip lines where either keypoint is missing

                r, g, b = pose_limb_color[sk_id % len(pose_limb_color)]
                cv2.line(im, (int(pos1[0]), int(pos1[1])), (int(pos2[0]), int(pos2[1])), (int(r), int(g), int(b)),
                         thickness=2)
        return im