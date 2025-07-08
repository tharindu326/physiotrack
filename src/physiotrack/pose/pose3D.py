from . import Models
from .. import MotionBERTInference
import os
import numpy as np
from .utils import COCO2Halpe, add_3d_keypoints
import json


class Pose3D:
    def __init__(self, model=None, device='cpu', 
                 config=None,
                 clip_len=243,
                 pixel=False,
                 render_video=True,
                 save_npy=True,
                 testloader_params=None,
                 **kwargs):
        
        if model is None:
            model = Models.Pose3D.MotionBERT.MB_ft_h36m_global_lite

        model_path = os.path.join(os.path.dirname(__file__), '..', 'modules', 'model_data', model.value)
        if not os.path.isfile(model_path):
            Models.download_model(model)
        
        if config is None:
            config = os.path.join(os.path.dirname(__file__), '..', 'modules', 'MotionBERT', 'configs', f'{model.name}.yaml')

        Models.validate_pose3d_model(model)
    
        self.minfo = Models._get_model_info(model)
        self.pose3d_framework = self.minfo['backend']
        print(f'Initiating {self.pose3d_framework} {model.name} for 3D Pose estimation')
        
        # Initialize 3D pose estimator based on framework
        if self.pose3d_framework == 'MotionBERT':
            self.pose3d_estimator = MotionBERTInference(
                config_path=config,
                checkpoint_path=model_path,
                clip_len=clip_len,
                testloader_params=testloader_params,
                device=device
            )
        else:
            raise ValueError(f"Invalid 3D model type: {self.pose3d_framework}")
        
        # Store parameters
        self.model = model
        self.device = device
        self.pixel = pixel
        self.render_video = render_video
        self.save_npy = save_npy
        self.clip_len = clip_len
    
    def estimate(self, json_path, vid_path, out_path=None, focus=None, 
                 scale_range=None, keep_imgs=False, no_conf=None, 
                 flip=None, rootrel=None, gt_2d=None, convert2alpha=True):
        """
        Estimate 3D poses from 2D pose detection JSON file
        """
        with open(json_path, 'r') as f:
            frames_data = json.load(f)

        if convert2alpha:
            dir_path = os.path.dirname(json_path)
            base_name = os.path.splitext(os.path.basename(json_path))[0]
            temp_output_json_path = os.path.join(dir_path, f"{base_name}_temp_alphapose.json")
            json_path = COCO2Halpe(json_path, temp_output_json_path) # converted temporary json file path

        results_3d = self.pose3d_estimator.infer(
            json_path=json_path,
            vid_path=vid_path,
            out_path=out_path,
            pixel=self.pixel,
            focus=focus,
            scale_range=scale_range,
            render_video=self.render_video,
            save_npy=self.save_npy,
            keep_imgs=keep_imgs,
            no_conf=no_conf,
            flip=flip,
            rootrel=rootrel,
            gt_2d=gt_2d
        )
        if convert2alpha:
            os.remove(temp_output_json_path)
        frames_data = add_3d_keypoints(frames_data, results_3d)

        if out_path:
            dir_path = os.path.dirname(out_path)
            base_name = os.path.splitext(os.path.basename(json_path))[0]
            output_json_path = os.path.join(dir_path, f"{base_name}_with_3d_keypoints.json")

            # Save updated frame data
            with open(output_json_path, 'w') as f:
                json.dump(frames_data, f, indent=2)
        
        return frames_data
    
    def process_batch(self, json_paths, vid_paths, out_paths=None, **kwargs):
        """Process multiple videos in batch"""
        results = []
        poses = []
        
        if out_paths is None:
            out_paths = [None] * len(json_paths)
        
        for json_path, vid_path, out_path in zip(json_paths, vid_paths, out_paths):
            result_3d, pose_3d = self.estimate(
                json_path=json_path,
                vid_path=vid_path,
                out_path=out_path,
                **kwargs
            )
            results.append(result_3d)
            poses.append(pose_3d)
        
        return results, poses