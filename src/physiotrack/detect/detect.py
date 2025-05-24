from . import Detector, Models

class ValidatedDetector(Detector):
    expected_subclass = None
    model = None

    def __init__(self, model=None, device='cpu', OBJECTNESS_CONFIDENCE=0.24, NMS_THRESHOLD=0.4,
                 classes=None, render_labels=True, render_box_detections=True, verbose=True, **kwargs):
        if self.expected_subclass is None:
            raise NotImplementedError("expected_subclass must be set in subclass")

        if model is None:
            model = self.model
            
        if model is None:
            raise ValueError("Model must be provided either as parameter or class attribute")
            
        Models.validate_det_model(model, expected_subclass=self.expected_subclass)
        super().__init__(
            model=model,
            device=device,
            OBJECTNESS_CONFIDENCE=OBJECTNESS_CONFIDENCE,
            NMS_THRESHOLD=NMS_THRESHOLD,
            classes=classes,
            render_labels=render_labels,
            render_box_detections=render_box_detections,
            verbose=verbose,
            **kwargs
        )

class Detection:
    class Custom(Detector):
        def __init__(self, model, device='cpu', OBJECTNESS_CONFIDENCE=0.24, NMS_THRESHOLD=0.4,
                     classes=None, render_labels=True, render_box_detections=True, verbose=True, **kwargs):
            
            # This should call a method with parameters (was missing arguments)
            Models.validate_det_model(model)
            super().__init__(
                model=model,
                device=device,
                OBJECTNESS_CONFIDENCE=OBJECTNESS_CONFIDENCE,
                NMS_THRESHOLD=NMS_THRESHOLD,
                classes=classes,
                render_labels=render_labels,
                render_box_detections=render_box_detections,
                verbose=verbose,
                **kwargs
            )
        
    class VRStudent(ValidatedDetector):
        expected_subclass = "VRStudent"
        model = Models.Detection.YOLO.VRSTUDENT.m_VRstudent

    class Face(ValidatedDetector):
        expected_subclass = "Face"
        model = Models.Detection.YOLO.FACE.m_face

    class Person(ValidatedDetector):
        expected_subclass = "Person"
        model = Models.Detection.YOLO.PERSON.m_person

    class VR(ValidatedDetector):
        expected_subclass = "VR"
        model = Models.Detection.YOLO.VR.m_VR