import os
from ultralytics import YOLO
import cv2
import numpy as np
import time
from pathlib import Path
from collections import deque
import sys
from .classes_and_palettes import COLORS
main_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(main_dir))


class Segmentor:
    def __init__(self, model, device, OBJECTNESS_CONFIDENCE, NMS_THRESHOLD, classes,
                 render_segmenttion_map=False, verbose=False, **kwargs):
        model_path = os.path.join(os.path.dirname(__file__), '..', 'model_data')
        self.model = YOLO(os.path.join(model_path, model.value))
        self.names = self.model.names
        self.COLORS = COLORS
        random = np.random.RandomState(11)
        # Handle classes being None
        num_classes = len(classes) if classes is not None else len(self.names)
        colors = random.randint(0, 255, (num_classes - 1, 3))
        colors = np.vstack((np.array([128, 128, 128]), colors)).astype(np.uint8)  # Add background color
        self.colors = colors[:, ::-1]

        self.device = device
        self.conf = OBJECTNESS_CONFIDENCE
        self.iou = NMS_THRESHOLD
        self.classes = classes
        self.verbose = verbose
        self.render_segmenttion_map = render_segmenttion_map
        self.extra_args = kwargs

        # FPS monitoring
        self.inference_times = deque(maxlen=100) 

    def segment(self, frame, **kwargs):
        start = time.perf_counter()

        all_kwargs = {
            "conf": self.conf,
            "iou": self.iou,
            "classes": self.classes,
            "device": self.device,
            "verbose": self.verbose,
            "retina_masks": True,  # Enable high-precision masks
            **self.extra_args,
            **kwargs,
        }

        results = self.model.predict(
            source=frame,
            task="segment",
            **all_kwargs
        )

        inference_time = time.perf_counter() - start
        self.inference_times.append(inference_time)

        output_image = frame.copy()
        segmentation_map = np.zeros(frame.shape[:2], dtype=np.uint8)
        h, w = segmentation_map.shape #single channel
        segmentation_img = np.zeros((h, w, 3), dtype=np.uint8) #3 channels

        for result in results:
            if hasattr(result, 'masks') and result.masks is not None:
                masks = result.masks.data.cpu().numpy()
                class_ids = result.boxes.cls.cpu().numpy()

                for i, mask in enumerate(masks):
                    class_id = int(class_ids[i])
                    # Resize mask to match frame dimensions
                    mask_resized = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
                    segmentation_map[mask_resized > 0.5] = class_id + 28 # appending from Sapiens classes

                    if self.render_segmenttion_map:
                        color = np.array(self.COLORS[list(self.COLORS)[(class_id + 28) % len(self.COLORS)]], dtype=np.uint8)
                        segmentation_img[segmentation_map == class_id + 28] = color

        return segmentation_img, segmentation_map

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
    from enum import Enum
    class Segmentation:  
        class YOLO(Enum):
            m11 = "yolo11m_VR_faceNeck.pt"
            
    output_path = 'test_path'
    model = Segmentation.YOLO.m11
    device = 'cpu'
    os.makedirs(output_path, exist_ok=True)
    # input = 'samples/'
    input = 'samples/BV_S17_cut1_frame_0.jpg'
    classes = [0, 1, 2]
    
    inference = Segmentor(model, device, OBJECTNESS_CONFIDENCE=0.24, NMS_THRESHOLD=0.4, classes=classes, 
                 render_segmenttion_map=True, verbose=False)
    
    if os.path.isdir(input):
        for file in os.listdir(input):
            im_path = os.path.join(input, file)
            out_path = os.path.join(output_path, f'segmented_{file}')
            print(im_path)
            
            im = cv2.imread(im_path)
            output_image_seg, seg_map = inference.segment(im)
            output_image = cv2.addWeighted(im.copy(), 0.5, output_image_seg, 0.5, 0)
            cv2.imwrite(out_path, output_image)
            
    elif os.path.isfile(input):
        out_path = os.path.join(output_path, f'segmented_{input.split("/")[-1]}')
        im = cv2.imread(input)
        output_image_seg, seg_map = inference.segment(im)
        output_image = cv2.addWeighted(im.copy(), 0.5, output_image_seg, 0.5, 0)
        cv2.imwrite(out_path, output_image)