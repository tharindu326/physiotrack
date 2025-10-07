from enum import Enum
import inspect
import os
import requests
from tqdm import tqdm
from pathlib import Path


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
                m_VRstudent = "yolo11m_VRstudent.pt"
                l_VRstudent = "yolo11l_VRstudent.pt"

        class RLDETR:
            class PERSON(Enum):
                x_person = "rtdetr-x.pt"
                l_person = "rtdetr-l.pt"

            class VRSTUDENT(Enum):
                x_person = "yolo11x_RLDETR_VRstudent.pt"
                l_person = "yolo11l_RLDETR_VRstudent.pt"
                
    class Pose:
        class YOLO:
            class COCO(Enum):
                M11 = "yolo11m-pose.pt"
                L11 = "yolo11l-pose.pt"
            
        class Sapiens:
            class WholeBody(Enum):
                # COCO wholebody
                B1_TS_COCOHB = "sapiens_1b_coco_wholebody_best_coco_wholebody_AP_727_torchscript.pt2"
                B06_TS_COCOHB = "sapiens_0.6b_coco_wholebody_best_coco_wholebody_AP_695_torchscript.pt2"
                B03_TS_COCOHB = "sapiens_0.3b_coco_wholebody_best_coco_wholebody_AP_620_torchscript.pt2"
            
        class ViTPose:
            class WholeBody(Enum):
                s_WHOLEBODY = "vitpose-s-wholebody.pth"
                b_WHOLEBODY = "vitpose-b-wholebody.pth"
                l_WHOLEBODY = "vitpose-l-wholebody.pth"
                h_WHOLEBODY = "vitpose-h-wholebody.pth"
                
            class COCO(Enum):
                b_COCO = "vitpose-b-coco.pth"
                h_COCO = "vitpose-h-coco.pth"
                l_COCO = "vitpose-l-coco.pth"
                s_COCO = "vitpose-s-coco.pth"

    class Pose3D:
        class MotionBERT(Enum):
            MB_ft_h36m_global_lite = 'FT_MB_lite_MB_ft_h36m_global_lite/best_epoch.bin'
            MB_ft_h36m = 'FT_MB_release_MB_ft_h36m/best_epoch.bin'
            # MB_ft_h36m_global = ''
            MB_train_h36m = 'MB_train_h36m/best_epoch.bin'

        class DDH(Enum):
            best = 'best_epoch_DDHPose.bin'

        class Canonicalizer:
            class Models(Enum):
                _3DPCNetS2 = 'best_model_3DPCNetS2.pth'
                _3DPCNetS3 = 'best_model_3DPCNetS3.pth'
                GEOMETRIC = ''

            class Configs(Enum):
                _3DPCNetS2 = 'best_model_3DPCNetS2.yaml'
                _3DPCNetS3 = 'best_model_3DPCNetS3.yaml'
            
            class View(Enum):
                FRONT = "front"
                BACK = "back" 
                LEFT_SIDE = "left_side"
                RIGHT_SIDE = "right_side"

    class Segmentation:
        class Sapiens:
            class BodyPart(Enum):
                B1_TS_SEG = "sapiens_1b_goliath_best_goliath_mIoU_7994_epoch_151_torchscript.pt2"
                B06_TS_SEG = "sapiens_0.6b_goliath_best_goliath_mIoU_7777_epoch_178_torchscript.pt2"
                B03_TS_SEG = "sapiens_0.3b_goliath_best_goliath_mIoU_7673_epoch_194_torchscript.pt2"

        class Yolo: 
            class VRHEAD(Enum):
                M11 = "yolo11m_VR_head.pt"

            class PERSON(Enum):
                m_person = "yolo11m-seg.pt"
                l_person = "yolo11l-seg.pt"
            

    @staticmethod
    def _get_model_info(model_enum):
        """Extract model information from enum instance"""
        if not isinstance(model_enum, Enum):
            return None
            
        for category_name in ['Detection', 'Pose', 'Segmentation', 'Pose3D']:
            category = getattr(Models, category_name, None)
            if not category:
                continue
            for backend_name in dir(category):
                if backend_name.startswith('_'):
                    continue
                    
                backend = getattr(category, backend_name)
                if not inspect.isclass(backend):
                    continue
                if category_name == "Pose3D":
                    if issubclass(backend, Enum) and isinstance(model_enum, backend):
                        return {
                            'category': category_name,
                            'backend': backend_name,
                            'enum_class': backend_name,  # For Pose3D, backend and enum_class are the same
                            'model_name': model_enum.name,
                            'file_name': model_enum.value
                        }
                    # Check for Canonicalizer models
                    elif backend_name == 'Canonicalizer':
                        for enum_class_name in dir(backend):
                            if enum_class_name.startswith('_'):
                                continue
                            enum_class = getattr(backend, enum_class_name)
                            if (inspect.isclass(enum_class) and 
                                issubclass(enum_class, Enum) and 
                                isinstance(model_enum, enum_class)):
                                return {
                                    'category': category_name,
                                    'backend': 'Canonicalizer',
                                    'enum_class': enum_class_name,
                                    'model_name': model_enum.name,
                                    'file_name': model_enum.value
                                }
                else:
                    for enum_class_name in dir(backend):
                        if enum_class_name.startswith('_'):
                            continue
                        enum_class = getattr(backend, enum_class_name)
                        if (inspect.isclass(enum_class) and 
                            issubclass(enum_class, Enum) and 
                            isinstance(model_enum, enum_class)):
                            return {
                                'category': category_name,
                                'backend': backend_name,
                                'enum_class': enum_class_name,
                                'model_name': model_enum.name,
                                'file_name': model_enum.value
                            }
        return None
    
    @staticmethod
    def _download_yolo_model(model_info, download_path):
        """Download ViTPose models from HuggingFace"""
        file_name = model_info['file_name']
        base_url = f"https://huggingface.co/tharindu326/physiotrack/resolve/main"
        download_url = f"{base_url}/{file_name}?download=true"
        return Models._download_file(download_url, file_name, download_path)

    @staticmethod
    def _download_sapiens_model(model_info, download_path):
        """Download Sapiens models from HuggingFace"""
        file_name = model_info['file_name']

        parts = file_name.split('_')
        size = parts[1] if len(parts) > 1 else "1b"

        size_map = {"03b": "0.3b", "06b": "0.6b", "1b": "1b"}
        size = size_map.get(size, size)

        if model_info['category'] == 'Pose':
            task = "pose-coco"
            format_type = "torchscript"
            base_url = f"https://huggingface.co/noahcao/sapiens-{task}/resolve/main/sapiens_lite_host/{format_type}/pose/checkpoints/sapiens_{size}"
        elif model_info['category'] == 'Segmentation':
            # Sapiens segmentation models - all use facebook repos
            task = "seg"
            format_type = "torchscript"
            base_url = f"https://huggingface.co/facebook/sapiens-{task}-{size}-{format_type}/resolve/main"
        download_url = f"{base_url}/{file_name}?download=true"
        return Models._download_file(download_url, file_name, download_path)

    @staticmethod
    def _download_vitpose_model(model_info, download_path):
        """Download ViTPose models from HuggingFace"""
        file_name = model_info['file_name']
        dataset = model_info['enum_class'].lower()  # 'wholebody' or 'coco'
        base_url = f"https://huggingface.co/JunkyByte/easy_ViTPose/resolve/main/torch/{dataset}"
        download_url = f"{base_url}/{file_name}?download=true"
        return Models._download_file(download_url, file_name, download_path)
    
    @staticmethod
    def _download_motionbert_model(model_info, download_path):
        """Download MotionBERT models from HuggingFace"""
        file_name = model_info['file_name']
        file_dir = os.path.dirname(file_name)
        actual_filename = os.path.basename(file_name)
        full_download_path = os.path.join(download_path, file_dir)
        os.makedirs(full_download_path, exist_ok=True)
        base_url = f"https://huggingface.co/walterzhu/MotionBERT/resolve/main/checkpoint/pose3d"
        download_url = f"{base_url}/{file_name}?download=true"
        
        return Models._download_file(download_url, actual_filename, full_download_path)
    
    def _download_ddh_model(model_info, download_path):
        """Download MotionBERT models from HuggingFace"""
        file_name = model_info['file_name']
        file_dir = os.path.dirname(file_name)
        actual_filename = os.path.basename(file_name)
        full_download_path = os.path.join(download_path, file_dir)
        os.makedirs(full_download_path, exist_ok=True)
        base_url = f"https://huggingface.co/tharindu326/physiotrack/resolve/main"
        download_url = f"{base_url}/{file_name}?download=true"
        
        return Models._download_file(download_url, actual_filename, full_download_path)
    
    @staticmethod
    def _download_canonicalizer_model(model_info, download_path):
        """Download Canonicalizer (3DPCNet) models and configs from HuggingFace"""
        file_name = model_info['file_name']
        file_dir = os.path.dirname(file_name)
        full_download_path = os.path.join(download_path, file_dir)
        os.makedirs(full_download_path, exist_ok=True)
        base_url = f"https://huggingface.co/tharindu326/physiotrack/resolve/main"
        
        # Download model file
        model_download_url = f"{base_url}/{file_name}?download=true"
        model_path = Models._download_file(model_download_url, file_name, full_download_path)
        
        # Also download corresponding config file if it's a 3DPCNet model
        if file_name.startswith('best_model_3DPCNet'):
            # Extract model name (e.g., '3DPCNetS2' from 'best_model_3DPCNetS2.pth')
            model_name = file_name.replace('best_model_', '').replace('.pth', '')
            config_name = f"best_model_{model_name}.yaml"
            config_download_url = f"{base_url}/{config_name}?download=true"
            Models._download_file(config_download_url, config_name, full_download_path)
        
        return model_path

    @staticmethod
    def _download_file(url, file_name, download_path):
        """Generic file download with progress bar"""
        os.makedirs(download_path, exist_ok=True)
        file_path = os.path.join(download_path, file_name)
        
        if os.path.exists(file_path):
            print(f"File {file_name} already exists at {file_path}")
            return file_path
                
        try:
            headers = {
                'Authorization': 'Bearer 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',
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
                'Accept-Language': 'en-US,en;q=0.5',
                'Accept-Encoding': 'gzip, deflate',
                'Connection': 'keep-alive',
                'Upgrade-Insecure-Requests': '1',
            }

            response = requests.get(url, stream=True, headers=headers)
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            block_size = 8192  # 8KB blocks
            
            with tqdm(total=total_size, unit='iB', unit_scale=True, desc=file_name) as pbar:
                with open(file_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=block_size):
                        if chunk:  # Filter out keep-alive chunks
                            f.write(chunk)
                            pbar.update(len(chunk))
            
            # print(f"Successfully downloaded {file_name} to {file_path}")
            return file_path
            
        except requests.exceptions.RequestException as e:
            print(f"Failed to download {file_name}: {e}")
            if os.path.exists(file_path):
                os.remove(file_path)
            raise

    @staticmethod
    def download_model(model_enum, download_path=f"{os.path.join(os.path.dirname(__file__))}/modules/model_data"):
        """
        Download a model based on its enum instance.
        """
        if not isinstance(model_enum, Enum):
            raise ValueError(f"Expected an Enum instance, got {type(model_enum)}")
        
        model_info = Models._get_model_info(model_enum)
        if not model_info:
            raise ValueError(f"Could not determine model information for {model_enum}")
        
        # print(f"Downloading {model_info['category']} model: {model_info['backend']}.{model_info['enum_class']}.{model_info['model_name']}")
        if model_info['backend'] == 'YOLO' or model_info['backend'] == 'RLDETR':
            if model_info['category'] == 'Pose' or (model_info['category'] == 'Detection' and model_info['enum_class'] == 'PERSON'):
                return None
            else:
                return Models._download_yolo_model(model_info, download_path)
        elif model_info['backend'] == 'Yolo':
            # Handle Segmentation.Yolo backend (note the capital 'Y')
            if model_info['category'] == 'Segmentation':
                # VRHEAD needs custom download, PERSON uses standard ultralytics
                if model_info['enum_class'] == 'PERSON':
                    return None  # Standard YOLO segmentation models auto-download
                else:
                    return Models._download_yolo_model(model_info, download_path)
        elif model_info['backend'] == 'Sapiens':
            return Models._download_sapiens_model(model_info, download_path)
        elif model_info['backend'] == 'ViTPose':
            return Models._download_vitpose_model(model_info, download_path)
        elif model_info['backend'] == 'MotionBERT':
            return Models._download_motionbert_model(model_info, download_path)
        elif model_info['backend'] == 'DDH':
            return Models._download_ddh_model(model_info, download_path)
        elif model_info['backend'] == 'Canonicalizer':
            return Models._download_canonicalizer_model(model_info, download_path)
        else:
            raise ValueError(f"Unknown backend: {model_info['backend']}")

    @staticmethod
    def validate_det_model(model, expected_subclass: str):
        """Verifies that `model` is a member of the Enum named `expected_subclass`"""
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

    @staticmethod
    def validate_seg_model(model, expected_subclass: str = None):
        """Verifies that `model` is a member of the Segmentation Enum"""
        if not isinstance(model, Enum):
            raise ValueError(f"Expected an Enum member for `model`, got {type(model).__name__}")

        # If expected_subclass is provided, validate against specific subclass
        if expected_subclass:
            target = expected_subclass.strip().upper()
            enum_classes = []
            for backend in (Models.Segmentation.Yolo, Models.Segmentation.Sapiens):
                # Check if the target exists in the backend
                for attr_name in dir(backend):
                    if attr_name.upper() == target:
                        enum_classes.append(getattr(backend, attr_name))

            if not enum_classes:
                raise ValueError(f"No segmentation subclass named '{expected_subclass}' in Yolo or Sapiens.")

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
        else:
            # General validation - check if it's any valid segmentation model
            for backend_name in dir(Models.Segmentation):
                if backend_name.startswith('_'):
                    continue
                backend = getattr(Models.Segmentation, backend_name)
                if not inspect.isclass(backend):
                    continue

                for enum_class_name in dir(backend):
                    if enum_class_name.startswith('_'):
                        continue
                    enum_class = getattr(backend, enum_class_name)
                    if (inspect.isclass(enum_class) and
                        issubclass(enum_class, Enum) and
                        isinstance(model, enum_class)):
                        return  # ✅ valid

            raise ValueError(
                f"Invalid segmentation model: {repr(model)}.\n"
                f"Expected a valid enum member from Models.Segmentation.<Backend>.<EnumClass>"
            )

    @staticmethod
    def validate_pose_model(model):
        """Validates whether the given model is one of the defined pose enum members"""
        if not isinstance(model, Enum):
            raise ValueError(f"Expected an Enum instance, got {type(model)}")
            
        for attr_name in dir(Models.Pose):
            if attr_name.startswith('_'):
                continue
                
            backend = getattr(Models.Pose, attr_name)
            if not inspect.isclass(backend):
                continue
                
            for sub_attr_name in dir(backend):
                if sub_attr_name.startswith('_'):
                    continue
                    
                sub = getattr(backend, sub_attr_name)
                if (inspect.isclass(sub) and 
                    issubclass(sub, Enum) and 
                    isinstance(model, sub)):
                    return  # ✅ Valid model found
                    
        raise ValueError(
            f"Invalid pose model: {repr(model)}.\n"
            f"Expected a valid enum member from Models.Pose.<Backend>.<EnumClass>"
        )

    @staticmethod
    def validate_pose3d_model(model):
        """Validates whether the given model is one of the defined pose3d enum members"""
        if not isinstance(model, Enum):
            raise ValueError(f"Expected an Enum instance, got {type(model)}")
            
        for attr_name in dir(Models.Pose3D):
            if attr_name.startswith('_'):
                continue
                
            backend = getattr(Models.Pose3D, attr_name)
            if not inspect.isclass(backend):
                continue
                
            # Check if this backend is an Enum class itself
            if issubclass(backend, Enum) and isinstance(model, backend):
                return  # ✅ Valid model found
                
            # Check sub-classes within the backend
            for sub_attr_name in dir(backend):
                if sub_attr_name.startswith('_'):
                    continue
                    
                sub = getattr(backend, sub_attr_name)
                if (inspect.isclass(sub) and 
                    issubclass(sub, Enum) and 
                    isinstance(model, sub)):
                    return  # ✅ Valid model found
                    
        # If we reach here, the model is not valid
        valid_models = []
        for attr_name in dir(Models.Pose3D):
            if attr_name.startswith('_'):
                continue
            backend = getattr(Models.Pose3D, attr_name)
            if inspect.isclass(backend) and issubclass(backend, Enum):
                for member in backend:
                    valid_models.append(f"Models.Pose3D.{attr_name}.{member.name}")
                    
        valid_str = "\n  ".join(valid_models)
        raise ValueError(
            f"Invalid pose3d model: {repr(model)}.\n"
            f"Expected a valid enum member from Models.Pose3D.<Backend>.<model_name>\n"
            f"Valid models are:\n  {valid_str}"
        )


if __name__ == "__main__":
    try:
        vitpose_path = Models.download_model(Models.Pose.ViTPose.WholeBody.s_WHOLEBODY)
        sapiens_path = Models.download_model(Models.Pose.Sapiens.WholeBody.B03_TS_COCOHB)
        yolo_path = Models.download_model(Models.Detection.YOLO.VRSTUDENT.m_VRstudent)
        
    except Exception as e:
        print(f"Error: {e}")