from . import YoloPose, VitInference, SapiensPoseEstimation, Models, Detection
import os 

class PoseBase:
    detector = None
    default_model = None
    detector_class = None
    
    def __init__(self, model=None, device='cpu', OBJECTNESS_CONFIDENCE=0.24, NMS_THRESHOLD=0.4,
                 classes=None, overlay_keypoints=True, render_labels=True, render_box_detections=True, verbose=True, 
                 model_DETECTION=None, OBJECTNESS_CONFIDENCE_DETECTION=0.24, NMS_THRESHOLD_DETECTION=0.4,
                 **kwargs):
        if model is None:
            if self.default_model is None:
                raise ValueError("Model must be provided either as parameter or class attribute")
            model = self.default_model
        Models.validate_pose_model(model)
        
        model_path = os.path.join(os.path.dirname(__file__), '..', 'modules', 'model_data', model.value)
        if not os.path.isfile(model_path):
            Models.download_model(model)
        
        minfo = Models._get_model_info(model)
        self.pose_framework = minfo['backend']
        print(f'Initiating {self.pose_framework} {model.name} for the Pose estimation')
        
        if self.pose_framework == 'ViTPose':
            self.pose_estimator = VitInference(model, device, overlay_keypoints)
        elif self.pose_framework == 'YOLO':
            self.pose_estimator = YoloPose(model, device, OBJECTNESS_CONFIDENCE, NMS_THRESHOLD, classes, overlay_keypoints, 
                                            render_labels, render_box_detections, verbose, **kwargs)
        elif self.pose_framework == 'Sapiens':
            self.pose_estimator = SapiensPoseEstimation(model, device)
        else:
            raise ValueError("Invalid model type. Please check the configuration")
        
        if self.detector_class is not None:
            self.detector = self.detector_class(
                            model=model_DETECTION,
                            device=device,
                            OBJECTNESS_CONFIDENCE=OBJECTNESS_CONFIDENCE_DETECTION,
                            NMS_THRESHOLD=NMS_THRESHOLD_DETECTION,
                            classes=classes,
                            render_labels=render_labels,
                            render_box_detections=render_box_detections,
                            verbose=verbose,
                            **kwargs
                    )
 
    def estimate(self, frame, boxes=None):
        if boxes is None and self.pose_framework in ["ViTPose", "Sapiens"]:
            if self.detector is None:
                if self.detector_class is not None:
                    self.detector = self.detector_class()
                else:
                    self.detector = Detection.Person()
            
            results, frame = self.detector.detect(frame)
            detections = results[0].boxes.data.cpu().numpy()  # (x1, y1, x2, y2, conf, cls)
            boxes = detections[:, :-2].astype(int)
        
        frame_with_pose, frame_data = self.pose_estimator.inference(frame, boxes)
        return frame_with_pose, frame_data
    
class Pose:
    class Custom(PoseBase):
        def __init__(self, model, device='cpu', OBJECTNESS_CONFIDENCE=0.24, NMS_THRESHOLD=0.4,
                 classes=None, overlay_keypoints=True, render_labels=True, render_box_detections=True, verbose=True, **kwargs):
            
            Models.validate_pose_model(model)
            super().__init__(
                model=model,
                device=device,
                OBJECTNESS_CONFIDENCE=OBJECTNESS_CONFIDENCE,
                NMS_THRESHOLD=NMS_THRESHOLD,
                overlay_keypoints=overlay_keypoints,
                classes=classes,
                render_labels=render_labels,
                render_box_detections=render_box_detections,
                verbose=verbose,
                **kwargs
            )
        
    class VRStudent(PoseBase):
        detector_class = Detection.VRStudent
        default_model = Models.Pose.ViTPose.WholeBody.b_WHOLEBODY
        
    class Person(PoseBase):
        detector_class = Detection.Person
        default_model = Models.Pose.ViTPose.WholeBody.b_WHOLEBODY