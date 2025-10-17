from ultralytics import YOLO
import cv2
import time
import numpy as np
import os
from pathlib import Path
from collections import deque
from .classes_and_palettes import COLORS
import sys
main_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(main_dir))


class Detector:
    def __init__(self, model, device, OBJECTNESS_CONFIDENCE, NMS_THRESHOLD, classes,
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
        self.extra_args = kwargs

        # FPS monitoring
        self.inference_times = deque(maxlen=100) 


    def __call__(self, img: np.ndarray) -> np.ndarray:
        return self.detect(img)

    def detect(self, frame: np.ndarray, **kwargs) -> np.ndarray:
        all_kwargs = {
            "conf": self.conf,
            "iou": self.iou,
            "classes": self.classes,
            "device": self.device,
            "verbose": self.verbose,
            **self.extra_args,
            **kwargs,
        }
        start = time.perf_counter()
        results = self.model.predict(source=frame, **all_kwargs)
        inference_time = time.perf_counter() - start
        self.inference_times.append(inference_time)

        detections = results[0].boxes.data.cpu().numpy()  # (x1, y1, x2, y2, conf, cls)
        boxes = detections[:, :-2].astype(int)

        labels = [self.model.names[int(cls)] for cls in results[0].boxes.cls]
        confidences = results[0].boxes.conf.cpu().numpy()
        output_img = self.draw_boxes(frame, boxes, labels, confidences)

        return results, output_img

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
    
    def draw_boxes(self, img, boxes, labels, confidences, color=(0, 255, 0), thickness=2):
        draw_img = img.copy()
        for box, label, conf in zip(boxes, labels, confidences):
            x1, y1, x2, y2 = map(int, box)
            color = self.COLORS[list(self.COLORS)[int(0) % len(self.COLORS)]]
            if self.render_box_detections:
                cv2.rectangle(draw_img, (x1, y1), (x2, y2), color, thickness, lineType=cv2.LINE_AA)
            if self.render_labels:
                text = f"{label}: {conf:.2f}"
                cv2.putText(draw_img, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                                    0.5, color, 1, cv2.LINE_AA)
        return draw_img


if __name__ == '__main__':
    from enum import Enum
    class Detection:
        class YOLO(Enum):
            n11 = "yolo11x.pt"  # https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11x.pt
            m11 = "yolo11m.pt"  # https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11m.pt
            l11 = "yolo11l.pt"  # https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11l.pt
            x11 = "yolo11x.pt"  # https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11x.pt
            m11_VR = 'best_VR_yolom.pt'
            l11_VR = 'best_VR_yolol.pt'
            x11_VR = 'best_VR_yolox.pt'
        class RLDETR(Enum):
            RLDETRl_VR = 'best_VR_RLDETRl.pt'
            RLDETRx_VR = 'best_VR_RLDETRx.pt'
            
    output = 'test_path'
    model = Detection.YOLO.m11_VR
    device = 'cpu'
    os.makedirs(output, exist_ok=True)
    # input = 'samples/'
    input = 'samples/BV_S17_cut1_frame_0.jpg'
    classes = [0, 1, 2]
    
    detector = Detector(model, device, OBJECTNESS_CONFIDENCE=0.24, NMS_THRESHOLD=0.4, classes=classes, 
                 render_labels=False, render_box_detections=False, verbose=False)
    
    if os.path.isfile(input):
        # Process a single image
        img = cv2.imread(input)
        if img is None:
            print(f"Error: Unable to load image {input}")
        else:
            results, output_img = detector.detect(img)
            output_path = os.path.join(output, os.path.basename(input))
            cv2.imwrite(output_path, output_img)
            print(f"Processed {input}, saved to {output_path}")
    elif os.path.isdir(input):
        # Process all images in the folder
        os.makedirs(output, exist_ok=True)
        for file_name in os.listdir(input):
            file_path = os.path.join(input, file_name)
            img = cv2.imread(file_path)
            if img is None:
                print(f"Skipping {file_name}: Unable to load image")
                continue
            results, output_img = detector.detect(img)
            
            output_path = os.path.join(output, file_name)
            cv2.imwrite(output_path, output_img)
            print(f"Processed {file_name}, saved to {output_path}")
    else:
        print(f"Error: {input} is not a valid file or directory")

    