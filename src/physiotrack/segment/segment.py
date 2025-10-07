from . import Segmentor, SapiensSegmentation, draw_segmentation_map, Models
import os
import numpy as np


class SegmentationBase:
    default_model = None

    def __init__(self, model=None, device='cpu', OBJECTNESS_CONFIDENCE=0.24, NMS_THRESHOLD=0.4,
                 classes=None, render_segmenttion_map=False, verbose=True, **kwargs):
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

    def segment(self, frame):
        """
        Perform segmentation on a frame

        Returns:
            For Yolo: (segmentation_img, segmentation_map)
            For Sapiens: segmentation_map
        """
        if self.segmentation_framework == 'Yolo':
            segmentation_img, segmentation_map = self.segmentor.segment(frame)
            return segmentation_img, segmentation_map
        elif self.segmentation_framework == 'Sapiens':
            segmentation_map = self.segmentor.inference(frame)
            if self.render_segmenttion_map:
                segmentation_img = draw_segmentation_map(segmentation_map)
                return segmentation_img, segmentation_map
            else:
                # Return zeros for consistency with Yolo output
                segmentation_img = np.zeros((frame.shape[0], frame.shape[1], 3), dtype=np.uint8)
                return segmentation_img, segmentation_map


class Segmentation:
    class Custom(SegmentationBase):
        def __init__(self, model, device='cpu', OBJECTNESS_CONFIDENCE=0.24, NMS_THRESHOLD=0.4,
                     classes=None, render_segmenttion_map=False, verbose=True, **kwargs):

            Models.validate_seg_model(model)
            super().__init__(
                model=model,
                device=device,
                OBJECTNESS_CONFIDENCE=OBJECTNESS_CONFIDENCE,
                NMS_THRESHOLD=NMS_THRESHOLD,
                classes=classes,
                render_segmenttion_map=render_segmenttion_map,
                verbose=verbose,
                **kwargs
            )

    class VRHEAD(SegmentationBase):
        default_model = Models.Segmentation.Yolo.VRHEAD.M11

    class Person(SegmentationBase):
        default_model = Models.Segmentation.Yolo.PERSON.m_person

    class BodyPart(SegmentationBase):
        default_model = Models.Segmentation.Sapiens.BodyPart.B1_TS_SEG
