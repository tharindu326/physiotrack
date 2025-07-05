from physiotrack import Pose, Video, Models, Detection, Tracker
from physiotrack.trackers import Config
from physiotrack.pose.config import COCO_WHOLEBODY_NAMES
from physiotrack.signals.motion import extract_key_point_sequence, add_body_centroid, add_head_centroid, resample_dataframe_by_interpolation
from physiotrack.signals.normalize import min_max_normalize
from physiotrack.signals.filters import band_pass_filter
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import json

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

# detection_data = video_processor.run(video_output_path, json_output_path)
with open(json_output_path, 'r') as f:
    detection_data = json.load(f)
print(f"Successfully processed video with {len(detection_data)} total detections")

sampling_freq = video_processor.video_fps

detection_data = add_body_centroid(detection_data, pose_estimator.archetecture)
detection_data = add_head_centroid(detection_data, pose_estimator.archetecture)


keypoint_id = int(COCO_WHOLEBODY_NAMES['left_hand_wrist'])
keypoint_df = extract_key_point_sequence(detection_data, keypoint_id, original_fps=video_processor.video_fps)
keypoint_df = keypoint_df.dropna(subset=['x', 'y'])
keypoint_df = resample_dataframe_by_interpolation(keypoint_df, input_fs=sampling_freq, output_fs=30, columns_to_resample=None)
keypoint_df['x_norm'] = min_max_normalize(keypoint_df['x'], feature_range=(0, 1))
keypoint_df['y_norm'] = min_max_normalize(keypoint_df['y'], feature_range=(0, 1))

x = keypoint_df['x_norm'].to_numpy()
y = keypoint_df['y_norm'].to_numpy()
time = keypoint_df['time'].to_numpy()

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
plt.title("Estimated Motion Signals (left wrist) from Video Analysis. Bandpass filtered.", fontsize=22)

plt.tight_layout()
plt.savefig(Path(output_directory) / f"{video_name}_motion_signals.png", dpi=300)
plt.show()


