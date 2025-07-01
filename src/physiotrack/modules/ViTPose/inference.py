import os
from typing import Optional
import cv2
import numpy as np
import torch
from thop import profile
from .configs.ViTPose_common import data_cfg
from .vit_models.model import ViTPose
from .vit_utils.inference import pad_image
from .vit_utils.top_down_eval import keypoints_from_heatmaps
from .vit_utils.util import dyn_model_import, infer_dataset_by_path
from .vit_utils.visualization import draw_points_and_skeleton, joints_dict
import time 

try:
    import onnxruntime
except ModuleNotFoundError:
    pass

__all__ = ['VitInference']
np.bool = np.bool_
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

class VitInference:
    """
    Class for performing inference using ViTPose models with YOLOv8 human detection.

    Args:
        model (str): Path to the ViT model file (.pth, .onnx, .engine).
        yolo (str): Path of the YOLOv8 model to load.
        model_name (str, optional): Name of the ViT model architecture to use.
                                    Valid values are 's', 'b', 'l', 'h'.
                                    Defaults to None, is necessary when using .pth checkpoints.
        device (str, optional): Device to use for inference. Defaults to 'cuda' if available, else 'cpu'.
    """

    def __init__(self, model_obj,
                 device = 'cpu', 
                 overlay_keypoints = False):

        model_name = model_obj.name.split("_")[0]
        model_path = os.path.join(os.path.dirname(__file__), '..', 'model_data')
        model = os.path.join(model_path, model_obj.value)
        if not os.path.isfile(model):
            raise ValueError(f'The model file {model} does not exist. Download it to model_data/')
        # assert os.path.isfile(yolo), f'The YOLOv model {yolo} does not exist'

        # Device priority is cuda / mps / cpu
        if device is None:
            if torch.cuda.is_available():
                device = 'cuda'
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                device = 'mps'
            else:
                device = 'cpu'

        self.device = device
        self.overlay_keypoints = overlay_keypoints
        
        # State saving during inference
        self.save_state = True  # Can be disabled manually
        self._img = None
        self._keypoints = None

        # Use extension to decide which kind of model has been loaded
        use_onnx = model.endswith('.onnx')
        use_trt = model.endswith('.engine')

        # Extract dataset name
        dataset = infer_dataset_by_path(model)
        assert dataset in ['mpii', 'coco', 'coco_25', 'wholebody', 'aic', 'ap10k', 'apt36k'], \
            'The specified dataset is not valid'

        self.dataset = dataset

        assert model_name in [None, 's', 'b', 'l', 'h'], \
            f'The model name {model_name} is not valid'

        # onnx / trt models do not require model_cfg specification
        if model_name is None:
            assert use_onnx or use_trt, \
                'Specify the model_name if not using onnx / trt'
        else:
            model_cfg = dyn_model_import(self.dataset, model_name)

        self.target_size = data_cfg['image_size']
        if use_onnx:
            self._ort_session = onnxruntime.InferenceSession(model,
                                                             providers=['CUDAExecutionProvider',
                                                                        'CPUExecutionProvider'])
            inf_fn = self._inference_onnx
        else:
            self._vit_pose = ViTPose(model_cfg)
            self._vit_pose.eval()

            if use_trt:
                self._vit_pose = torch.jit.load(model)
            else:
                ckpt = torch.load(model, map_location='cpu')
                if 'state_dict' in ckpt:
                    self._vit_pose.load_state_dict(ckpt['state_dict'])
                else:
                    self._vit_pose.load_state_dict(ckpt)
                self._vit_pose.to(torch.device(device))

            inf_fn = self._inference_torch

        # Override _inference abstract with selected engine
        self._inference = inf_fn  # type: ignore

    @classmethod
    def postprocess(cls, heatmaps, org_w, org_h):
        """
        Postprocess the heatmaps to obtain keypoints and their probabilities.

        Args:
            heatmaps (ndarray): Heatmap predictions from the model.
            org_w (int): Original width of the image.
            org_h (int): Original height of the image.

        Returns:
            ndarray: Processed keypoints with probabilities.
        """
        points, prob = keypoints_from_heatmaps(heatmaps=heatmaps,
                                               center=np.array([[org_w // 2,
                                                                 org_h // 2]]),
                                               scale=np.array([[org_w, org_h]]),
                                               unbiased=True, use_udp=True)
        return np.concatenate([points[:, :, ::-1], prob], axis=2)

    def inference(self, img, bboxes) -> dict:
        """
        Perform inference on the input image.

        Args:
            img (ndarray): Input image for inference in RGB format.

        Returns:
            dict: Inference results.
        """
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        frame_keypoints = {}
        start = time.perf_counter()
        frame_data = {"detections": []}
        output_frame = None
        for i, bbox in enumerate(bboxes):
            pad_bbox = 10
            bbox[[0, 2]] = np.clip(bbox[[0, 2]] + [-pad_bbox, pad_bbox], 0, img.shape[1])
            bbox[[1, 3]] = np.clip(bbox[[1, 3]] + [-pad_bbox, pad_bbox], 0, img.shape[0])

            img_inf = img[bbox[1]:bbox[3], bbox[0]:bbox[2]]
            img_inf, (left_pad, top_pad) = pad_image(img_inf, 3 / 4)

            keypoints = self._inference(img_inf)[0]
            keypoints[:, :2] += bbox[:2][::-1] - [top_pad, left_pad]
            frame_keypoints[len(frame_keypoints)] = keypoints
            
            keypoints_list = [
                                {"id": idx, "x": float(kp[1]), "y": float(kp[0]), "confidence": float(kp[2])}
                                for idx, kp in enumerate(keypoints)
                            ]
            detection = {
                            "id": i,
                            "bbox": bbox.tolist(),
                            "keypoints": keypoints_list
                        }
            frame_data["detections"].append(detection)

        if self.save_state:
            self._img = img
            self._keypoints = frame_keypoints
            
        if self.overlay_keypoints:
            output_frame = self.draw()
        # print(f"ViTPose inference took: {time.perf_counter() - start:.4f} seconds")
        return output_frame, frame_data

    def draw(self, confidence_threshold=0.5):
        """
        Draw keypoints on the image.

        Args:
            confidence_threshold (float, optional): Confidence threshold for keypoints. Default is 0.5.

        Returns:
            ndarray: Image with keypoints drawn.
        """
        self._img = cv2.cvtColor(self._img, cv2.COLOR_RGB2BGR)
        img = self._img.copy()

        for idx, k in self._keypoints.items():
            img = draw_points_and_skeleton(img.copy(), k,
                                           joints_dict()[self.dataset]['skeleton'],
                                           person_index=idx,
                                           points_color_palette='gist_rainbow',
                                           skeleton_color_palette='jet',
                                           points_palette_samples=10,
                                           confidence_threshold=confidence_threshold)
        return img

    @torch.no_grad()
    def _inference_torch(self, img: np.ndarray) -> np.ndarray:
        img_input = cv2.resize(img, self.target_size, interpolation=cv2.INTER_LINEAR) / 255
        img_input = ((img_input - MEAN) / STD).transpose(2, 0, 1)[None].astype(np.float32)
        img_input = torch.from_numpy(img_input).to(torch.device(self.device))
        heatmaps = self._vit_pose(img_input).detach().cpu().numpy()
        return self.postprocess(heatmaps, img.shape[1], img.shape[0])

    def _inference_onnx(self, img: np.ndarray) -> np.ndarray:
        img_input = cv2.resize(img, self.target_size, interpolation=cv2.INTER_LINEAR) / 255
        img_input = ((img_input - MEAN) / STD).transpose(2, 0, 1)[None].astype(np.float32)
        ort_inputs = {self._ort_session.get_inputs()[0].name: img_input}
        heatmaps = self._ort_session.run(None, ort_inputs)[0]
        return self.postprocess(heatmaps, img.shape[1], img.shape[0])
