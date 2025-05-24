from enum import Enum
import inspect


class Models:
    class Detection:
        class YOLO:
            class PERSON(Enum):
                m_person = "yolo11m.pt"
                l_person = "yolo11l.pt"

            class FACE(Enum):
                m_face = "yolo11m_face.pt"
                l_face = "yolo11l_face.pt"

            class VR(Enum):
                m_VR = "yolo11m_vr.pt"
                l_VR = "yolo11l_vr.pt"

            class VRSTUDENT(Enum):
                m_VRstudent = "yolo11m_vrstudent.pt"
                l_VRstudent = "yolo11l_vrstudent.pt"

        class RLDETR:
            # class VR(Enum):
            #     RLDETRl_VR = "best_VR_RLDETRl.pt"
            #     RLDETRx_VR = "best_VR_RLDETRx.pt"
            
            class PERSON(Enum):
                x_person = "rtdetr-x.pt"
                l_person = "rtdetr-l.pt"

            class VRSTUDENT(Enum):
                x_person = "best_VR_RLDETRx.pt"
                l_person = "best_VR_RLDETRl.pt"
                
    class Pose:
        class YOLO:
            class COCO(Enum):
                M11 = "yolo11m-pose.pt"
                L11 = "yolo11l-pose.pt"

        """
            COCO WholeBody:
                Keypoints: 133 keypoints
                Keypoint Groups:
                    Face: 68 facial landmarks (covering contour, eyes, nose, mouth)
                    Hands: 21 keypoints per hand (similar to hand tracking)
                    Body: 17 COCO body keypoints
        """
            
        class Sapiens:
            class WholeBody(Enum):
                # COCO wholebody
                B1_TS_COCOHB = "sapiens_1b_coco_wholebody_best_coco_wholebody_AP_727_torchscript.pt2"  # https://huggingface.co/noahcao/sapiens-pose-coco/resolve/main/sapiens_lite_host/torchscript/pose/checkpoints/sapiens_1b/sapiens_1b_coco_wholebody_best_coco_wholebody_AP_727_torchscript.pt2?download=true
                B06_TS_COCOHB = "sapiens_0.6b_coco_wholebody_best_coco_wholebody_AP_695_torchscript.pt2"  # https://huggingface.co/noahcao/sapiens-pose-coco/resolve/main/sapiens_lite_host/torchscript/pose/checkpoints/sapiens_0.6b/sapiens_0.6b_coco_wholebody_best_coco_wholebody_AP_695_torchscript.pt2?download=true
                B03_TS_COCOHB = "sapiens_0.3b_coco_wholebody_best_coco_wholebody_AP_620_torchscript.pt2"  # https://huggingface.co/noahcao/sapiens-pose-coco/resolve/main/sapiens_lite_host/torchscript/pose/checkpoints/sapiens_0.3b/sapiens_0.3b_coco_wholebody_best_coco_wholebody_AP_620_torchscript.pt2?download=true
            
        class ViTPose:
            class WholeBody(Enum):
                s_WHOLEBODY = "vitpose-s-wholebody.pth"  # https://huggingface.co/JunkyByte/easy_ViTPose/resolve/main/torch/wholebody/vitpose-s-wholebody.pth?download=true
                b_WHOLEBODY = "vitpose-b-wholebody.pth"  # https://huggingface.co/JunkyByte/easy_ViTPose/resolve/main/torch/wholebody/vitpose-b-wholebody.pth?download=true
                l_WHOLEBODY = "vitpose-l-wholebody.pth"  # https://huggingface.co/JunkyByte/easy_ViTPose/resolve/main/torch/wholebody/vitpose-l-wholebody.pth?download=true
                h_WHOLEBODY = "vitpose-h-wholebody.pth"  # https://huggingface.co/JunkyByte/easy_ViTPose/resolve/main/torch/wholebody/vitpose-h-wholebody.pth?download=true
            class COCO(Enum):
                b_WHOLEBODY = "vitpose-b-coco.pth" # https://huggingface.co/JunkyByte/easy_ViTPose/resolve/main/torch/coco/vitpose-b-coco.pth?download=true
                h_WHOLEBODY = "vitpose-h-coco.pth" # https://huggingface.co/JunkyByte/easy_ViTPose/resolve/main/torch/coco/vitpose-h-coco.pth?download=true
                l_WHOLEBODY = "vitpose-l-coco.pth" # https://huggingface.co/JunkyByte/easy_ViTPose/resolve/main/torch/coco/vitpose-l-coco.pth?download=true
                s_WHOLEBODY = "vitpose-s-coco.pth" # https://huggingface.co/JunkyByte/easy_ViTPose/resolve/main/torch/coco/vitpose-s-coco.pth?download=true
    
    @staticmethod
    def validate_det_model(model, expected_subclass: str):
        """
        Verifies that `model` is a member of the Enum named `expected_subclass`
        under either Detection.YOLO or Detection.RLDETR.
        """
        if not isinstance(model, Enum):
            raise ValueError(f"Expected an Enum member for `model`, got {type(model).__name__}")
        target = expected_subclass.strip().upper()
        enum_classes = []
        for backend in (Models.Detection.YOLO, Models.Detection.RLDETR):
            if hasattr(backend, target):
                enum_classes.append(getattr(backend, target))
        if not enum_classes:
            raise ValueError(f"No detection subclass named '{expected_subclass}' in YOLO or RLDETR.")
        for enum_cls in enum_classes:
            if isinstance(model, enum_cls):
                return  # ✅ valid
        all_valid = []
        for enum_cls in enum_classes:
            names = ", ".join(e.name for e in enum_cls)
            all_valid.append(f"{enum_cls.__module__.split('.')[-1]}.{enum_cls.__name__}: [{names}]")
        valid_str = "\n  ".join(all_valid)
        raise ValueError(
            f"Model '{model.name}' is not valid for subclass '{expected_subclass}'.\n"
            f"Valid members are:\n  {valid_str}"
        )
        
    def is_valid_detection_model(model):
            """
            Returns True if the given model is a valid enum member under Models.Detection.<Backend>.<EnumClass>.
            For example: Models.Detection.YOLO.VRSTUDENT.m_VRstudent
            """
            for _, backend_cls in inspect.getmembers(Models.Detection, inspect.isclass):
                for _, enum_cls in inspect.getmembers(backend_cls, inspect.isclass):
                    if issubclass(enum_cls, Enum):
                        if isinstance(model, enum_cls):
                            return True
            raise ValueError(
            f"Invalid model '{model}'. Expected a model from Models.Detection.<YOLO|RLDETR>.<enum>."
        )
    
    
    @staticmethod
    def validate_pose_model(model):
        """
        Validates whether the given model is one of the defined physiotrack.Models.Pose enum members.

        ✔ Example valid models:
            physiotrack.Models.Pose.YOLO.COCO.M11
            physiotrack.Models.Pose.Sapiens.WholeBody.B1_TS_COCOHB
            physiotrack.Models.Pose.ViTPose.WholeBody.l_WHOLEBODY

        ✘ Invalid examples:
            "yolo11m-pose.pt"  (str)
            Models.Detection.YOLO.FACE.m_face  (not from Pose)
            None

        :param model: Enum member from physiotrack.Models.Pose.*
        :raises ValueError: if the model is not part of any valid Pose enum
        """
        # Loop over all top-level categories in Models.Pose (YOLO, Sapiens, ViTPose, etc.)
        for backend in Models.Pose.__dict__.values():
            if not isinstance(backend, type):
                continue

            # Loop over all nested Enum classes (e.g. COCO, WholeBody)
            for sub in backend.__dict__.values():
                if isinstance(sub, type) and issubclass(sub, Enum):
                    if isinstance(model, sub):
                        return  # ✅ Valid model found

        raise ValueError(
            f"Invalid pose model: {repr(model)}.\n"
            f"Expected a valid enum member from physiotrack.Models.Pose.<Backend>.<EnumClass>, "
            f"like Models.Pose.YOLO.COCO.M11 or Models.Pose.ViTPose.WholeBody.s_WHOLEBODY."
        )
            
def get_model_category(param):
    if isinstance(param, Models.Detection.YOLO):
        return "Detection"
    elif isinstance(param, Models.Segmentation.Sapiens):
        return "Segmentation"
    elif isinstance(param, (Models.Pose.YOLO, Models.Pose.Sapiens, Models.Pose.ViTPose)):
        return "Pose"
    else:
        return "Unknown Category"