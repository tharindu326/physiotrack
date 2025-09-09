from . import Models
from .. import MotionBERTInference, DDHPoseInference
import os
import numpy as np
from .utils import COCO2Halpe, add_3d_keypoints, coco2h36m
import json
from .canonicalizer import canonicalize_pose, CanonicalView
from ..modules.MotionBERT.utils.vismo import render_and_save
from datetime import datetime


class Pose3D:
    def __init__(self, model=None, device='cpu', 
                 config=None,
                 clip_len=243,
                 pixel=False,
                 render_video=True,
                 save_npy=True,
                 testloader_params=None,
                 # DDHPose specific parameters
                 boneindex_h36m='0,1,1,2,2,3,0,4,4,5,5,6,0,7,7,8,8,9,9,10,8,11,11,12,12,13,8,14,14,15,15,16',
                 number_of_frames=243,
                 test_time_augmentation=True,
                 timestep=1000,
                 scale=1.0,
                 cs=512,
                 dep=8,
                 joints_left=[4, 5, 6, 11, 12, 13],
                 joints_right=[1, 2, 3, 14, 15, 16],
                 num_proposals=300,
                 sampling_timesteps=5,
                 **kwargs):
        
        if model is None:
            model = Models.Pose3D.MotionBERT.MB_ft_h36m_global_lite

        model_path = os.path.join(os.path.dirname(__file__), '..', 'modules', 'model_data', model.value)
        if not os.path.isfile(model_path):
            Models.download_model(model)

        Models.validate_pose3d_model(model)
    
        self.minfo = Models._get_model_info(model)
        self.pose3d_framework = self.minfo['backend']
        print(f'Initiating {self.pose3d_framework} {model.name} for 3D Pose estimation')
        
        # Initialize 3D pose estimator based on framework
        if self.pose3d_framework == 'MotionBERT':
            if config is None:
                config = os.path.join(os.path.dirname(__file__), '..', 'modules', 'MotionBERT', 'configs', f'{model.name}.yaml')
            self.pose3d_estimator = MotionBERTInference(
                config_path=config,
                checkpoint_path=model_path,
                clip_len=clip_len,
                testloader_params=testloader_params,
                device=device
            )
        elif self.pose3d_framework == 'DDH':
            self.pose3d_estimator = DDHPoseInference(
                boneindex_h36m=boneindex_h36m,
                number_of_frames=number_of_frames,
                test_time_augmentation=test_time_augmentation,
                timestep=timestep,
                scale=scale,
                cs=cs,
                dep=dep,
                joints_left=joints_left,
                joints_right=joints_right,
                num_proposals=num_proposals,
                sampling_timesteps=sampling_timesteps,
                checkpoint_path=model_path,
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
                 flip=None, rootrel=None, gt_2d=None, convert2alpha=True, canonical_view=None, canonical_method=None,
                 # DDHPose specific parameters
                 batch_size=64):
        """
        Estimate 3D poses from 2D pose detection JSON file
        
        Args:
            json_path: Path to 2D pose detection JSON file
            vid_path: Path to input video
            out_path: Optional output directory path
            canonical_view: Optional canonical view to apply (CanonicalView.FRONT, BACK, LEFT_SIDE, RIGHT_SIDE).
                          If None, no canonical transformation is applied.
                          Note: Canonical view can also be applied separately using CanonicalViewProcessor
            Other args: Various model-specific parameters
        
        Returns:
            Tuple of (frames_data, results_3d):
            - frames_data: Detection data with 3D keypoints
            - results_3d: Raw 3D poses array (N, 17, 3)
        """
        with open(json_path, 'r') as f:
            frames_data = json.load(f)

        if self.pose3d_framework == 'MotionBERT':
            if convert2alpha:
                dir_path = os.path.dirname(json_path)
                base_name = os.path.splitext(os.path.basename(json_path))[0]
                temp_output_json_path = os.path.join(dir_path, f"{base_name}_temp_alphapose.json")
                json_path = COCO2Halpe(json_path, temp_output_json_path) # converted temporary json file path

            results_3d = self.pose3d_estimator.infer(
                json_path=json_path,
                vid_path=vid_path,
                pixel=self.pixel,
                focus=focus,
                scale_range=scale_range,
                no_conf=no_conf,
                flip=flip,
                rootrel=rootrel,
                gt_2d=gt_2d
            )
            
            if convert2alpha:
                os.remove(temp_output_json_path)
                
        elif self.pose3d_framework == 'DDH':

            keypoints_2d = coco2h36m(json_path)
            results_3d = self.pose3d_estimator.infer(
                keypoints_2d=keypoints_2d,
                vid_path=vid_path,
                batch_size=batch_size,
            )
        
        if canonical_view:
            results_3d = canonicalize_pose(results_3d, view=canonical_view, method=canonical_method)

        if out_path:
            os.makedirs(out_path, exist_ok=True)
            
        if self.render_video and out_path:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            render_and_save(
                results_3d, 
                f'{out_path}/X3D_{timestamp}.mp4', 
                fps=self.pose3d_estimator.fps_in
            )

        if self.save_npy and out_path:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            np.save(f'{out_path}/X3D_{timestamp}.npy', results_3d)

        frames_data = add_3d_keypoints(frames_data, results_3d)

        if out_path:
            dir_path = os.path.dirname(out_path)
            base_name = os.path.splitext(os.path.basename(json_path))[0]
            output_json_path = os.path.join(dir_path, f"{base_name}_with_3d_keypoints.json")

            # Save updated frame data
            with open(output_json_path, 'w') as f:
                json.dump(frames_data, f, indent=2)
        
        return frames_data, results_3d
    
    def process_batch(self, json_paths, vid_paths, out_paths=None, **kwargs):
        """Process multiple videos in batch"""
        results = []
        poses = []
        
        if out_paths is None:
            out_paths = [None] * len(json_paths)
        
        for json_path, vid_path, out_path in zip(json_paths, vid_paths, out_paths):
            frames_data, results_3d = self.estimate(
                json_path=json_path,
                vid_path=vid_path,
                out_path=out_path,
                **kwargs
            )
            results.append(frames_data)
            poses.append(results_3d)
        
        return results, poses