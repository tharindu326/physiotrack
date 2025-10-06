"""
Full inference pipeline: Detection -> Tracking -> Pose Estimation -> Segmentation
Processes a video and outputs all results overlaid on the same frame
"""

from physiotrack import Pose, Video, Models, Detection, Tracker, Segmentation
from physiotrack.trackers import Config
from pathlib import Path
import argparse


def run_full_inference(video_path, output_dir='output/full_inference'):
    """
    Run full inference pipeline on a video

    Args:
        video_path: Path to input video
        output_dir: Directory to save output video
    """

    # Setup paths
    input_path = Path(video_path)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    video_name = input_path.stem
    output_video_path = output_path / f"{video_name}_full_inference.mp4"
    output_json_path = output_path / f"{video_name}_result.json"

    print("="*60)
    print("Full Inference Pipeline")
    print("="*60)
    print(f"Input video: {video_path}")
    print(f"Output video: {output_video_path}")
    print(f"Output JSON: {output_json_path}")
    print("="*60)

    # Initialize models
    print("\n[1/4] Initializing VRStudent Detector...")
    detector = Detection.Person(
        model=Models.Detection.YOLO.PERSON.m_person,
        render_box_detections=True,
        render_labels=True,
        verbose=True,
        device=0
    )

    print("[2/4] Initializing Pose Estimator...")
    pose_estimator = Pose.Custom(
        model=Models.Pose.ViTPose.WholeBody.b_WHOLEBODY,
        render_box_detections=True,
        render_labels=True,
        overlay_keypoints=True,
        verbose=False,
        device=0
    )

    print("[3/4] Initializing Tracker...")
    tracker_config = Config()
    tracker_config.tracker_type = 'ocsort'
    tracker_config.debug_mode = False
    tracker_config.classes = [0]
    tracker = Tracker(config=tracker_config)

    print("[4/4] Initializing Segmentor...")
    segmentor = Segmentation.Person(
        device=0,
        OBJECTNESS_CONFIDENCE=0.24,
        NMS_THRESHOLD=0.4,
        classes=[0],  # Person class only
        render_segmenttion_map=True,
        verbose=False
    )

    print("\n✓ All models initialized successfully!")

    # Process video using Video processor
    print("\n" + "="*60)
    print("Processing Video")
    print("="*60)

    video_processor = Video(
        video_path=video_path,
        pose_estimator=pose_estimator,
        detector=detector,  # Pass VRStudent detector to work with Pose.Custom
        tracker=tracker,
        segmentator=segmentor,
        required_fps=None,
        frame_resize=None,
        frame_rotate=False,
        output_path=output_dir,
        verbose=True
    )

    # Run the pipeline
    detection_data = video_processor.run(output_video_path, output_json_path)

    print(f"\n✓ Processing complete!")
    print(f"✓ Output video saved to: {output_video_path}")
    print(f"✓ Output JSON saved to: {output_json_path}")
    print(f"✓ Total frames processed: {len(detection_data)}")
    print("="*60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Full inference pipeline for video processing')
    parser.add_argument('video_path', type=str, help='Path to input video')
    parser.add_argument('--output_dir', type=str, default='output/full_inference',
                        help='Directory to save output video (default: output/full_inference)')

    args = parser.parse_args()

    run_full_inference(args.video_path, args.output_dir)
