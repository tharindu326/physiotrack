"""
Full inference pipeline: Detection -> Tracking -> Pose Estimation -> Segmentation
Processes a video and outputs all results overlaid on the same frame
"""

from physiotrack import Pose, Video, Models, Detection, Tracker, Segmentation
from physiotrack.trackers import Config
from pathlib import Path
import argparse


def run_full_inference(video_path, output_dir='output/full_inference', floor_map=None, 
                       floor_map_background=None, batch_size=1):
    """
    Run full inference pipeline on a video

    Args:
        video_path: Path to input video
        output_dir: Directory to save output video
        floor_map: List of 4 corner points [(x1,y1), (x2,y2), (x3,y3), (x4,y4)] defining floor area
        floor_map_background: Background mode for floor map canvas:
            - None or "default": Black canvas with gray background (default)
            - "auto" or "extract": Extract floor area from first video frame with homography
            - Path string: Load pre-made floor plan image from file path
        batch_size: Number of frames to process in batch (default: 1)
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
    print(f"Batch size: {batch_size}")
    print("="*60)

    # Initialize models
    print("\n[1/4] Initializing VRStudent Detector...")
    detector = Detection.Person(
        model=Models.Detection.YOLO.PERSON.m_person,
        render_box_detections=True,
        render_labels=True,
        verbose=False,
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

    print("[4/4] Initializing Segmentors...")
    segmentor_person = Segmentation.Person(
        device=0,
        OBJECTNESS_CONFIDENCE=0.24,
        NMS_THRESHOLD=0.4,
        classes=[0],  # Person class only
        render_segmenttion_map=True,
        segmentation_filter={'bbox_filter': False},
        verbose=False
    )

    segmentor_vrhead = Segmentation.VRHEAD(
        device=0,
        OBJECTNESS_CONFIDENCE=0.24,
        NMS_THRESHOLD=0.4,
        # classes=[0],
        render_segmenttion_map=True,
        segmentation_filter={
            'bbox_filter': True,
            'detector_index': 0,  # Use detector index 0
            'detector_class_filter': None  # Use all classes
        },
        verbose=False
    )

    # Combine multiple segmentators
    segmentors = [segmentor_person, segmentor_vrhead]

    print("\n✓ All models initialized successfully!")
    print(f"  - Segmentators: {len(segmentors)} (Person + VRHEAD)")

    # Process video using Video processor
    print("\n" + "="*60)
    print("Processing Video")
    print("="*60)

    video_processor = Video(
        video_path=video_path,
        pose_estimator=pose_estimator,
        detector=detector,  # Pass VRStudent detector to work with Pose.Custom
        tracker=tracker,
        segmentator=segmentors,  # Pass list of segmentators (Person + VRHEAD)
        required_fps=None,
        frame_resize=None,
        frame_rotate=False,
        floor_map=floor_map,  # Floor area for radar view
        floor_map_background=floor_map_background,  # Background mode: None/"default", "auto"/"extract", or path to image
        output_path=output_dir,
        verbose=True,
        show_fps=True,
        batch_size=batch_size  # Enable batch processing
    )

    # Run the pipeline
    detection_data = video_processor.run(output_video_path, output_json_path)

    print(f"\n✓ Processing complete!")
    print(f"✓ Output video saved to: {output_video_path}")
    print(f"✓ Output JSON saved to: {output_json_path}")
    print(f"✓ Total frames processed: {len(detection_data)}")
    print("="*60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Full inference pipeline for video processing',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Background Mode Examples:
  --floor_map_background "default"              # Black canvas (default)
  --floor_map_background "auto"                 # Extract from first video frame
  --floor_map_background "path/to/floorplan.png" # Load from image file
        """
    )
    parser.add_argument('video_path', type=str, help='Path to input video')
    parser.add_argument('--output_dir', type=str, default='output/full_inference',
                        help='Directory to save output video (default: output/full_inference)')
    parser.add_argument('--floor_map', type=str, default=None,
                        help='Floor area coordinates as "x1,y1,x2,y2,x3,y3,x4,y4" (e.g., "314,824,778,402,1140,456,936,1035")')
    parser.add_argument('--floor_map_background', type=str, default=None,
                        help='Background mode: "default" (black), "auto"/"extract" (from first frame), or path to floor plan image')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='Number of frames to process in batch (default: 1)')

    args = parser.parse_args()

    # Parse floor_map if provided
    floor_map = None
    if args.floor_map:
        coords = [int(x) for x in args.floor_map.split(',')]
        if len(coords) == 8:
            floor_map = [(coords[i], coords[i+1]) for i in range(0, 8, 2)]
        else:
            print("Warning: floor_map must have 8 values (4 points with x,y). Ignoring floor_map.")

    run_full_inference(args.video_path, args.output_dir, floor_map, 
                      args.floor_map_background, args.batch_size)
