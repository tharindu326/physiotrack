from . import Segmentor, SapiensSegmentation, draw_segmentation_map, Models
import os
import numpy as np


class SegmentationBase:
    default_model = None

    def __init__(self, model=None, device='cpu', OBJECTNESS_CONFIDENCE=0.24, NMS_THRESHOLD=0.4,
                 classes=None, render_segmenttion_map=False, segmentation_filter=None, verbose=True, **kwargs):
        if model is None:
            if self.default_model is None:
                raise ValueError("Model must be provided either as parameter or class attribute")
            model = self.default_model

        Models.validate_seg_model(model)

        model_path = os.path.join(os.path.dirname(__file__), '..', 'modules', 'model_data', model.value)
        if not os.path.isfile(model_path):
            Models.download_model(model)

        self.minfo = Models._get_model_info(model)
        self.segmentation_framework = self.minfo['backend']
        print(f'Initiating {self.segmentation_framework} {model.name} for Segmentation')

        if self.segmentation_framework == 'Yolo':
            self.segmentor = Segmentor(model, device, OBJECTNESS_CONFIDENCE, NMS_THRESHOLD, classes,
                                       render_segmenttion_map, verbose, **kwargs)
        elif self.segmentation_framework == 'Sapiens':
            self.segmentor = SapiensSegmentation(model, device)
            self.render_segmenttion_map = render_segmenttion_map
        else:
            raise ValueError("Invalid model type. Please check the configuration")

        self.model = model
        self.device = device
        self.OBJECTNESS_CONFIDENCE = OBJECTNESS_CONFIDENCE
        self.NMS_THRESHOLD = NMS_THRESHOLD
        self.classes = classes
        self.verbose = verbose

        # Parse segmentation filter parameters
        if segmentation_filter is None:
            segmentation_filter = {}

        self.bbox_filter = segmentation_filter.get('bbox_filter', False)
        self.detector_index = segmentation_filter.get('detector_index', None)
        self.detector_class_filter = segmentation_filter.get('detector_class_filter', None)

    def segment(self, frame, bboxes=None):
        """
        Perform segmentation on a frame

        Args:
            frame: Input frame
            bboxes: Optional list of bounding boxes [[x1, y1, x2, y2], ...] to filter segmentation.
                    Only segmentation within these boxes will be kept.

        Returns:
            For Yolo: (segmentation_img, segmentation_map)
            For Sapiens: segmentation_map
        """
        if self.segmentation_framework == 'Yolo':
            segmentation_img, segmentation_map = self.segmentor.segment(frame)
        elif self.segmentation_framework == 'Sapiens':
            segmentation_map = self.segmentor.inference(frame)
            if self.render_segmenttion_map:
                segmentation_img = draw_segmentation_map(segmentation_map)
            else:
                segmentation_img = np.zeros((frame.shape[0], frame.shape[1], 3), dtype=np.uint8)

        # Filter segmentation by bounding boxes if provided
        if bboxes is not None and len(bboxes) > 0:
            h, w = segmentation_map.shape[:2]
            filtered_map = np.zeros((h, w), dtype=segmentation_map.dtype)
            filtered_img = np.zeros((h, w, 3), dtype=np.uint8)

            # Keep only segmentation within bounding boxes
            for bbox in bboxes:
                x1, y1, x2, y2 = map(int, bbox[:4])
                # Ensure bbox is within frame bounds
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)

                # Copy segmentation within bbox
                filtered_map[y1:y2, x1:x2] = segmentation_map[y1:y2, x1:x2]
                filtered_img[y1:y2, x1:x2] = segmentation_img[y1:y2, x1:x2]

            return filtered_img, filtered_map

        return segmentation_img, segmentation_map

    def get_avg_inference_time(self):
        """Get average inference time in milliseconds."""
        if hasattr(self.segmentor, 'get_avg_inference_time'):
            return self.segmentor.get_avg_inference_time()
        return 0.0

    def get_avg_fps(self):
        """Get average FPS based on inference times."""
        if hasattr(self.segmentor, 'get_avg_fps'):
            return self.segmentor.get_avg_fps()
        return 0.0


class Segmentation:
    class Custom(SegmentationBase):
        def __init__(self, model, device='cpu', OBJECTNESS_CONFIDENCE=0.24, NMS_THRESHOLD=0.4,
                     classes=None, render_segmenttion_map=False, segmentation_filter=None, verbose=True, **kwargs):

            Models.validate_seg_model(model)
            super().__init__(
                model=model,
                device=device,
                OBJECTNESS_CONFIDENCE=OBJECTNESS_CONFIDENCE,
                NMS_THRESHOLD=NMS_THRESHOLD,
                classes=classes,
                render_segmenttion_map=render_segmenttion_map,
                segmentation_filter=segmentation_filter,
                verbose=verbose,
                **kwargs
            )

    class VRHEAD(SegmentationBase):
        default_model = Models.Segmentation.Yolo.VRHEAD.M11

    class Person(SegmentationBase):
        default_model = Models.Segmentation.Yolo.PERSON.m_person

    class BodyPart(SegmentationBase):
        default_model = Models.Segmentation.Sapiens.BodyPart.B1_TS_SEG
