import os
from ultralytics import YOLO
import cv2
import numpy as np
from pathlib import Path
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
        colors = random.randint(0, 255, (len(classes) - 1, 3))
        colors = np.vstack((np.array([128, 128, 128]), colors)).astype(np.uint8)  # Add background color
        self.colors = colors[:, ::-1]

        self.device = device
        self.conf = OBJECTNESS_CONFIDENCE
        self.iou = NMS_THRESHOLD
        self.classes = classes
        self.verbose = verbose
        self.render_segmenttion_map = render_segmenttion_map
        self.extra_args = kwargs 

    def segment(self, frame, **kwargs):
        
        all_kwargs = {
            "conf": self.conf,
            "iou": self.iou,
            "classes": self.classes,
            "device": self.device,
            "verbose": self.verbose,
            **self.extra_args,
            **kwargs,
        }
        
        results = self.model.predict(
            source=frame,
            task="segment",
            **all_kwargs
        )

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
                    segmentation_map[mask > 0] = class_id + 28 # appending from Spaiens classes 
                
                    if self.render_segmenttion_map:
                        color = np.array(self.COLORS[list(self.COLORS)[(class_id + 28) % len(self.COLORS)]], dtype=np.uint8)
                        segmentation_img[segmentation_map == class_id + 28] = color
            
        return segmentation_img, segmentation_map
    

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