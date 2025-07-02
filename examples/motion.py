from physiotrack import Pose, Video, Models, Detection, Tracker
from physiotrack.trackers import Config
from physiotrack.pose.config import COCO_WHOLEBODY_NAMES
from physiotrack.signals.motion import extract_key_point_sequence
from pathlib import Path


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

detection_data = video_processor.run(video_output_path, json_output_path)
print(f"Successfully processed video with {len(detection_data)} total detections")

keypoint_id = COCO_WHOLEBODY_NAMES['left_hand_wrist']
keypoint_df = extract_key_point_sequence(detection_data, keypoint_id, original_fps=video_processor.video_fps)