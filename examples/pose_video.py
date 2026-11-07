from physiotrack import Pose, Video
from pathlib import Path

processor = Pose.VRStudent(verbose=False, device=0)
input_video = 'BV_S17_cut1.mp4'
output_directory = 'output'
input_path = Path(input_video)
video_name = input_path.stem

video_processor = Video(
    source=input_video,
    pose=processor,
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
