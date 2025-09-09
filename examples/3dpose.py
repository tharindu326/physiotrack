from physiotrack.pose.pose3D import Pose3D
from physiotrack import Models

# Use the View enum from Models
CanonicalView = Models.Pose3DCanonicalizer.View

# ------------------- MotionBERT -------------------

pose3D = Pose3D(    model=Models.Pose3D.MotionBERT.MB_ft_h36m_global_lite, 
                    config=None,
                    device='cuda', 
                    clip_len=243,
                    pixel=False,
                    render_video=True,
                    save_npy=True,
                    testloader_params=None)

# Returns both frames_data and results_3d
frames_data, results_3d = pose3D.estimate(json_path='output/BV_S17_cut1_result.json', vid_path='BV_S17_cut1.mp4', out_path='output/', canonical_view=CanonicalView.FRONT)

# ------------------- DDHPose -------------------

pose3D = Pose3D(    model=Models.Pose3D.DDH.best, 
                    device='cuda', 
                    render_video=True,
                    save_npy=True,
                    num_proposals=10,
                    sampling_timesteps=5
                    )

# Returns both frames_data and results_3d
frames_data, results_3d = pose3D.estimate(json_path='output/BV_S17_cut1_result.json', 
                          vid_path='BV_S17_cut1.mp4',
                          out_path='output/', 
                          batch_size=8, 
                          canonical_view=CanonicalView.FRONT)