from physiotrack import Pose, Video, Models, Detection, Tracker
from physiotrack.trackers import Config
from physiotrack.pose.config import COCO_WHOLEBODY_NAMES, HUMAN26M_NAMES
from physiotrack.signals.motion import extract_key_point_sequence_3d, extract_key_point_sequence_2d, add_body_centroid, add_head_centroid, resample_dataframe_by_interpolation
from physiotrack.signals.normalize import min_max_normalize
from physiotrack.signals.filters import band_pass_filter
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import json
from physiotrack.pose.pose3D import Pose3D

# pose estimator with no detector
pose_estimator = Pose.Custom(model=Models.Pose.ViTPose.WholeBody.b_WHOLEBODY, render_box_detections=False, render_labels=True, overlay_keypoints=True, verbose=False, device=0)  
detector = Detection.VRStudent(model=Models.Detection.YOLO.VRSTUDENT.m_VRstudent, render_box_detections=False, render_labels=False, verbose=False, device=0)
TrackerConfig = Config()
TrackerConfig.tracker_type = 'ocsort'
TrackerConfig.debug_mode = True
TrackerConfig.classes = [0]
TrackerConfig.enable_student_tracking =True
tracker = Tracker(config=TrackerConfig)

input_video = 'BV_S17_cut1.mp4'
output_directory = 'output'
input_path = Path(input_video)
video_name = input_path.stem
required_fps=None,
frame_resize=None,
frame_rotate=False,

video_processor = Video(
    video_path=input_video,
    pose_estimator=pose_estimator,
    detector=detector,
    tracker=tracker,
    required_fps=None,
    frame_resize=None,
    frame_rotate=False,
    output_path=output_directory,
    verbose=True
)

video_output_path = Path(output_directory) / f"{video_name}_poses.mp4"
json_output_path = Path(output_directory) / f"{video_name}_result.json"

detection_data_2D = video_processor.run(video_output_path, json_output_path)

sampling_freq = video_processor.video_fps

# 3D pose generation
pose3D = Pose3D(    model=Models.Pose3D.MotionBERT.MB_ft_h36m_global_lite, 
                    config=None,
                    device='cuda', 
                    clip_len=243,
                    pixel=False,
                    render_video=True,
                    save_npy=True,
                    testloader_params=None)

detection_data_3D = pose3D.estimate(json_path=json_output_path, vid_path=input_video, out_path='output/')
# file_path = 'output/BV_S17_cut1_result_temp_alphapose_with_3d_keypoints.json'
# with open(file_path, 'r', encoding='utf-8') as file:
#     detection_data_3D = json.load(file)

detection_data = add_body_centroid(detection_data_3D, pose_estimator.archetecture)
detection_data = add_head_centroid(detection_data, pose_estimator.archetecture)

# keypoint extraction
keypoint_id = int(COCO_WHOLEBODY_NAMES['left_hand_wrist'])
keypoint_df_2d = extract_key_point_sequence_2d(detection_data, keypoint_id, original_fps=video_processor.video_fps)
keypoint_id = int(HUMAN26M_NAMES['left_wrist'])
keypoint_df_3d = extract_key_point_sequence_3d(detection_data, keypoint_id, original_fps=video_processor.video_fps)
 

keypoint_df_2d = keypoint_df_2d.dropna(subset=['x', 'y'])
keypoint_df_2d = resample_dataframe_by_interpolation(keypoint_df_2d, input_fs=sampling_freq, output_fs=30, columns_to_resample=None)
keypoint_df_2d['x_norm'] = min_max_normalize(keypoint_df_2d['x'], feature_range=(0, 1))
keypoint_df_2d['y_norm'] = min_max_normalize(keypoint_df_2d['y'], feature_range=(0, 1))

x = keypoint_df_2d['x_norm'].to_numpy()
y = keypoint_df_2d['y_norm'].to_numpy()
time = keypoint_df_2d['time'].to_numpy()

# x = band_pass_filter(x, [0.05, 0.35], sampling_freq)
# y = band_pass_filter(y, [0.05, 0.35], sampling_freq)

# Plotting
plt.figure(figsize=(12, 6))
plt.plot(time, x, label="X coords (left_hand_wrist) - ViTPose", lw=1.8)
plt.plot(time, y, label="Y coords (left_hand_wrist) - ViTPose", lw=1.8)

plt.grid(True, which='major', color='#666666', linestyle='-')
plt.legend(loc='upper left', fontsize=16)
plt.minorticks_on()
plt.grid(True, which='minor', color='#999999', linestyle='-', alpha=0.3)
plt.ylabel('Normalized Amplitude', fontsize=14)
plt.xlabel('time (s)', fontsize=14)
plt.title("Estimated Motion Signals (left wrist) from Video Analysis.", fontsize=22)

plt.tight_layout()
plt.savefig(Path(output_directory) / f"{video_name}_motion_signals.png", dpi=300)
plt.show()

# 3D

keypoint_df_3d = keypoint_df_3d.dropna(subset=['x', 'y', 'z'])
keypoint_df_3d = resample_dataframe_by_interpolation(keypoint_df_3d, input_fs=sampling_freq, output_fs=30, columns_to_resample=None)
keypoint_df_3d['x_norm'] = min_max_normalize(keypoint_df_3d['x'], feature_range=(0, 1))
keypoint_df_3d['y_norm'] = min_max_normalize(keypoint_df_3d['y'], feature_range=(0, 1))
keypoint_df_3d['z_norm'] = min_max_normalize(keypoint_df_3d['z'], feature_range=(0, 1))

x_3d = keypoint_df_3d['x_norm'].to_numpy()
y_3d = keypoint_df_3d['y_norm'].to_numpy()
z_3d = keypoint_df_3d['z_norm'].to_numpy()
time_3d = keypoint_df_3d['time'].to_numpy()

# x_3d = band_pass_filter(x_3d, [0.05, 0.35], sampling_freq)
# y_3d = band_pass_filter(y_3d, [0.05, 0.35], sampling_freq)
# z_3d = band_pass_filter(z_3d, [0.05, 0.35], sampling_freq)

# Plotting 3D data
plt.figure(figsize=(12, 6))
plt.plot(time_3d, x_3d, label="X coords (left_hand_wrist) - 3D", lw=1.8)
plt.plot(time_3d, y_3d, label="Y coords (left_hand_wrist) - 3D", lw=1.8)
plt.plot(time_3d, z_3d, label="Z coords (left_hand_wrist) - 3D", lw=1.8)

plt.grid(True, which='major', color='#666666', linestyle='-')
plt.legend(loc='upper left', fontsize=16)
plt.minorticks_on()
plt.grid(True, which='minor', color='#999999', linestyle='-', alpha=0.3)
plt.ylabel('Normalized Amplitude', fontsize=14)
plt.xlabel('time (s)', fontsize=14)
plt.title("Estimated 3D Motion Signals (left wrist) from Video Analysis.", fontsize=22)

plt.tight_layout()
plt.savefig(Path(output_directory) / f"{video_name}_3d_motion_signals.png", dpi=300)
plt.show()


# Comparison Plot 1: X coordinates (2D vs 3D)
plt.figure(figsize=(12, 6))
plt.plot(time, x, label="X coords - 2D ViTPose", lw=1.8, color='blue')
plt.plot(time_3d, x_3d, label="X coords - 3D", lw=1.8, color='red')

plt.grid(True, which='major', color='#666666', linestyle='-')
plt.legend(loc='upper left', fontsize=16)
plt.minorticks_on()
plt.grid(True, which='minor', color='#999999', linestyle='-', alpha=0.3)
plt.ylabel('Normalized Amplitude', fontsize=14)
plt.xlabel('time (s)', fontsize=14)
plt.title("X Coordinate Comparison: 2D vs 3D (left wrist)", fontsize=22)

plt.tight_layout()
plt.savefig(Path(output_directory) / f"{video_name}_x_comparison_2d_vs_3d.png", dpi=300)
plt.show()

# Comparison Plot 2: Y coordinates (2D vs 3D)
plt.figure(figsize=(12, 6))
plt.plot(time, y, label="Y coords - 2D ViTPose", lw=1.8, color='green')
plt.plot(time_3d, y_3d, label="Y coords - 3D", lw=1.8, color='orange')

plt.grid(True, which='major', color='#666666', linestyle='-')
plt.legend(loc='upper left', fontsize=16)
plt.minorticks_on()
plt.grid(True, which='minor', color='#999999', linestyle='-', alpha=0.3)
plt.ylabel('Normalized Amplitude', fontsize=14)
plt.xlabel('time (s)', fontsize=14)
plt.title("Y Coordinate Comparison: 2D vs 3D (left wrist)", fontsize=22)

plt.tight_layout()
plt.savefig(Path(output_directory) / f"{video_name}_y_comparison_2d_vs_3d.png", dpi=300)
plt.show()


