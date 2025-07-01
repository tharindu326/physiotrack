from physiotrack import Pose, Video
from pathlib import Path

processor = Pose.VRStudent(render_box_detections=False, render_labels=True, overlay_keypoints=True, verbose=False, device=0)  
input_video = 'BV_S17_cut1.mp4'
output_directory = 'output'
input_path = Path(input_video)
video_name = input_path.stem
required_fps=None,
frame_resize=None,
frame_rotate=False,

video_processor = Video(
    video_path=input_video,
    pose_estimator=processor,
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
