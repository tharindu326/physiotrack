import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from .common.camera import normalize_screen_coordinates
from .common.ddhpose import DDHPose
import imageio
from tqdm import tqdm


class DDHPoseInference:
    def __init__(
                    self, 
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
                    checkpoint_path='best_epoch_DDHPose.bin',
                    device='cuda' if torch.cuda.is_available() else 'cpu'):
        
        self.device = device
        self.checkpoint_path = checkpoint_path
        self.model = self._load_model(
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
                                        sampling_timesteps=sampling_timesteps
                                        )
        self.number_of_frames = number_of_frames
        self.fps_in = 30

    def _load_model(self, 
                    boneindex_h36m='0,1,1,2,2,3,0,4,4,5,5,6,0,7,7,8,8,9,9,10,8,11,11,12,12,13,8,14,14,15,15,16',
                    number_of_frames=27,
                    test_time_augmentation=True,
                    timestep=1000,
                    scale=1.0,
                    cs=512,
                    dep=8,
                    joints_left=[4, 5, 6, 11, 12, 13],
                    joints_right=[1, 2, 3, 14, 15, 16],
                    num_proposals=10,
                    sampling_timesteps=5):
        """Load and initialize the model"""
        
        model_backbone = DDHPose(
                                boneindex_h36m=boneindex_h36m,
                                number_of_frames=number_of_frames,
                                test_time_augmentation=test_time_augmentation,
                                timestep=timestep,
                                scale=scale,
                                cs=cs,
                                dep=dep,
                                joints_left=joints_left,
                                joints_right=joints_right,
                                is_train=False,
                                num_proposals=num_proposals,
                                sampling_timesteps=sampling_timesteps
                    )

        if self.device == 'cuda' and torch.cuda.is_available():
            model_backbone = nn.DataParallel(model_backbone)
            model_backbone = model_backbone.cuda()
        else:
            model_backbone = model_backbone.to(self.device)
        
        # Load checkpoint
        print(f'Loading checkpoint: {self.checkpoint_path}')
        self.checkpoint = torch.load(self.checkpoint_path, map_location=self.device, weights_only=False)
        model_backbone.load_state_dict(self.checkpoint['model_pos'], strict=False)
        model_backbone.eval()
        
        return model_backbone

    def prepare_input_2d(self, keypoints_2d, vid_path):
        """Normalize 2D keypoints to [-1, 1] range"""

        vid = imageio.get_reader(vid_path, 'ffmpeg')
        self.fps_in = vid.get_meta_data()['fps']
        vid_size = vid.get_meta_data()['size']
        width, height = vid_size
        keypoints_2d = normalize_screen_coordinates(keypoints_2d, w=width, h=height)
        return keypoints_2d


    def eval_data_prepare(self, receptive_field, inputs_2d):
        """Prepare data for evaluation with sliding window"""
        inputs_2d = torch.squeeze(inputs_2d)
        if inputs_2d.shape[0] / receptive_field > inputs_2d.shape[0] // receptive_field:
            out_num = inputs_2d.shape[0] // receptive_field + 1
        else:
            out_num = inputs_2d.shape[0] // receptive_field
        eval_input_2d = torch.empty(out_num, receptive_field, inputs_2d.shape[1], inputs_2d.shape[2])
        for i in range(out_num - 1):
            eval_input_2d[i] = inputs_2d[i * receptive_field:(i + 1) * receptive_field]
        # Handle last batch with padding if necessary
        if inputs_2d.shape[0] < receptive_field:
            pad_right = receptive_field - inputs_2d.shape[0]
            inputs_2d = rearrange(inputs_2d, 'b f c -> f c b')
            inputs_2d = F.pad(inputs_2d, (0, pad_right), mode='replicate')
            inputs_2d = rearrange(inputs_2d, 'f c b -> b f c')
        eval_input_2d[-1] = inputs_2d[-receptive_field:]
        return eval_input_2d


    def reconstruct_poses(self, predictions, receptive_field, original_length):
        """Reconstruct full sequence from windowed predictions"""
        if original_length / receptive_field > original_length // receptive_field:
            batch_num = (original_length // receptive_field) + 1
            output = np.empty((original_length, 17, 3))
            for i in range(batch_num - 1):
                output[i * receptive_field:(i + 1) * receptive_field] = predictions[i]
            left_frames = original_length - (batch_num - 1) * receptive_field
            output[-left_frames:] = predictions[-1, -left_frames:]
        else:
            output = predictions.reshape(original_length, 17, 3)
        return output

    def infer(self, keypoints_2d, vid_path, batch_size=64):
        kps_left = [4, 5, 6, 11, 12, 13]
        kps_right = [1, 2, 3, 14, 15, 16]
        
        original_length = keypoints_2d.shape[0]
        # Normalize 2D keypoints
        keypoints_2d = self.prepare_input_2d(keypoints_2d, vid_path)
        
    
        # Prepare input
        inputs_2d = torch.from_numpy(keypoints_2d).float().unsqueeze(0)  # Add batch dimension
        
        # Apply test-time augmentation
        inputs_2d_flip = inputs_2d.clone()
        inputs_2d_flip[:, :, :, 0] *= -1
        inputs_2d_flip[:, :, kps_left + kps_right, :] = inputs_2d_flip[:, :, kps_right + kps_left, :]
        
        # Prepare data with sliding window
        receptive_field = self.number_of_frames
        inputs_2d = self.eval_data_prepare(receptive_field, inputs_2d)
        inputs_2d_flip = self.eval_data_prepare(receptive_field, inputs_2d_flip)
        
        # Create dummy 3D input (zeros)
        inputs_3d = torch.zeros(inputs_2d.shape[0], inputs_2d.shape[1], 17, 3)
        
        if self.device == 'cuda':
            inputs_2d = inputs_2d.to(self.device)
            inputs_2d_flip = inputs_2d_flip.to(self.device)
            inputs_3d = inputs_3d.to(self.device)
        
        all_predictions = []
        
        with torch.no_grad():
            bs = batch_size
            total_batches = (inputs_2d.shape[0] + bs - 1) // bs
            
            for batch_idx in tqdm(range(total_batches), desc="Processing batches"):
                start_idx = batch_idx * bs
                end_idx = min((batch_idx + 1) * bs, inputs_2d.shape[0])
                
                batch_2d = inputs_2d[start_idx:end_idx]
                batch_2d_flip = inputs_2d_flip[start_idx:end_idx]
                batch_3d = inputs_3d[start_idx:end_idx]
                predictions = self.model(batch_2d, batch_3d, input_2d_flip=batch_2d_flip)
                
                # Simple selection: take last timestep, first proposal
                best_poses = predictions[:, -1, 0, :, :, :].cpu().numpy()
                all_predictions.append(best_poses)
        
        # Concatenate all predictions
        all_predictions = np.concatenate(all_predictions, axis=0)
        # Reconstruct full sequence
        output_3d = self.reconstruct_poses(all_predictions, receptive_field, original_length)
        return output_3d


if __name__ == "__main__":
    pose_inference = DDHPoseInference(num_proposals=5, sampling_timesteps=5)
    from physiotrack.pose.utils import coco2h36m
    input_2d = 'BV_S17_cut1_result.json',
    keypoints_2d = coco2h36m(input_2d)
    # Run inference
    results = pose_inference.infer(
        input_2d=keypoints_2d,
        out_path='output/',
        render_video=True,
        save_npy=True
    )