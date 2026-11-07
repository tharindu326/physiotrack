from physiotrack import Pose, Video, Models, Detection, Tracker, TrackerConfig
from pathlib import Path


# pose estimator with explicit detector + tracker
pose_estimator = Pose.Custom(model=Models.Pose.ViTPose.WholeBody.b_wholebody, verbose=False, device=0)
detector = Detection.VRStudent(model=Models.Detection.YOLO.VRSTUDENT.m_vrstudent, verbose=False, device=0)
tracker_config = TrackerConfig()
tracker_config.tracker_type = 'ocsort'
tracker_config.debug_mode = True
tracker_config.classes = [0]
tracker_config.enable_subject_lock = True
tracker = Tracker(config=tracker_config)

input_video = 'BV_S17_cut1.mp4'
output_directory = 'output'
input_path = Path(input_video)
video_name = input_path.stem

video_processor = Video(
    source=input_video,
    pose=pose_estimator,
    detector=detector,
    tracker=tracker,
    fps=None,
    resize=None,
    rotate=False,
    output_dir=output_directory,
    verbose=True
)

video_output_path = Path(output_directory) / f"{video_name}_poses.mp4"
json_output_path = Path(output_directory) / f"{video_name}_result.json"

frame_results = video_processor.run(video_output_path, json_output_path)
print(
    f"Successfully processed {len(frame_results)} frames with "
    f"{sum(len(frame) for frame in frame_results)} total pose instances"
)
