"""
Full inference pipeline: Detection -> Tracking -> Pose Estimation -> Segmentation -> Face Orientation
Processes a video and outputs all results overlaid on the same frame
"""

from physiotrack import Pose, Video, Models, Detection, Tracker, Segmentation, Face, FaceOrientation
from physiotrack.face import draw_axis
from physiotrack.trackers import Config
from pathlib import Path
import argparse
import cv2
import numpy as np


def run_full_inference(video_path, output_dir='output/full_inference', floor_map=None, 
                       floor_map_background=None, floor_map_rotation=0, 
                       plot_keypoint=None, plot_keypoint_name=None, batch_size=1):
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
        floor_map_rotation: Rotation angle in degrees (0, 90, 180, 270) to orient the radar view
        plot_keypoint: COCO keypoint ID to plot motion (e.g., 9=left_wrist, 10=right_wrist)
        plot_keypoint_name: Name of keypoint for plot label
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
    print("\n[1/5] Initializing VRStudent Detector...")
    detector = Detection.Person(
        model=Models.Detection.YOLO.PERSON.m_person,
        render_box_detections=True,
        render_labels=True,
        verbose=False,
        device=0
    )

    print("[2/5] Initializing Pose Estimator...")
    pose_estimator = Pose.Custom(
        model=Models.Pose.ViTPose.WholeBody.b_WHOLEBODY,
        render_box_detections=True,
        render_labels=True,
        overlay_keypoints=True,
        verbose=False,
        device=0
    )

    print("[3/5] Initializing Tracker...")
    tracker_config = Config()
    tracker_config.tracker_type = 'ocsort'
    tracker_config.debug_mode = False
    tracker_config.classes = [0]
    tracker = Tracker(config=tracker_config)

    print("[4/5] Initializing Segmentors...")
    segmentor_person = Segmentation.Person(
        device=0,
        OBJECTNESS_CONFIDENCE=0.24,
        NMS_THRESHOLD=0.4,
        classes=[0],  # Person class only
        render_segmenttion_map=True,
        segmentation_filter={'bbox_filter': False},
        verbose=False
    )

    segmentor_vrhead = Segmentation.Custom(
        model=Models.Segmentation.Yolo.VRHEAD.M8_251029,
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

    print("[5/5] Initializing Face Orientation Estimator...")
    face_detector = Face(device=0, verbose=False)
    face_orientation = FaceOrientation(device=0, render_pose=False, verbose=False)

    print("\n✓ All models initialized successfully!")
    print(f"  - Segmentators: {len(segmentors)} (Person + VRHEAD)")
    print(f"  - Face Orientation: Enabled")

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
        face_detector=face_detector,  # Face detector for face orientation
        face_orientation=face_orientation,  # Face orientation estimator
        required_fps=None,
        frame_resize=None,
        frame_rotate=False,
        floor_map=floor_map,  # Floor area for radar view
        floor_map_background=floor_map_background,  # Background mode: None/"default", "auto"/"extract", or path to image
        floor_map_rotation=floor_map_rotation,  # Rotation: 0, 90, 180, or 270 degrees
        plot_keypoint=plot_keypoint,  # Keypoint ID to plot motion (relative to pelvis)
        plot_keypoint_name=plot_keypoint_name,  # Keypoint name for plot label
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
                Examples:
                # Basic: Floor map with radar view
                python full_inference.py video.mp4 --floor_map "314,824,778,402,1140,456,936,1035"
                
                # With auto-extracted floor and motion plotting (left wrist)
                python full_inference.py video.mp4 --floor_map "314,824,778,402,1140,456,936,1035" \\
                    --floor_map_background "auto" --floor_map_rotation 90 \\
                    --plot_keypoint 9 --plot_keypoint_name "left_wrist"
                
                # Motion plotting only (without floor map)
                python full_inference.py video.mp4 --plot_keypoint 9 --plot_keypoint_name "left_wrist"
                
                Common COCO Keypoint IDs:
                  0=nose, 5=left_shoulder, 6=right_shoulder, 7=left_elbow, 8=right_elbow
                  9=left_wrist, 10=right_wrist, 11=left_hip, 12=right_hip
                  13=left_knee, 14=right_knee, 15=left_ankle, 16=right_ankle
                """
                )
    parser.add_argument('video_path', type=str, help='Path to input video')
    parser.add_argument('--output_dir', type=str, default='output/full_inference',
                        help='Directory to save output video (default: output/full_inference)')
    parser.add_argument('--floor_map', type=str, default=None,
                        help='Floor area coordinates as "x1,y1,x2,y2,x3,y3,x4,y4" (e.g., "314,824,778,402,1140,456,936,1035")')
    parser.add_argument('--floor_map_background', type=str, default=None,
                        help='Background mode: "default" (black), "auto"/"extract" (from first frame), or path to floor plan image')
    parser.add_argument('--floor_map_rotation', type=int, default=90, choices=[0, 90, 180, 270],
                        help='Rotation angle in degrees to align radar view orientation (default: 90)')
    parser.add_argument('--plot_keypoint', type=int, default=None,
                        help='COCO keypoint ID to plot motion (e.g., 9=left_wrist, 10=right_wrist, 7=left_elbow)')
    parser.add_argument('--plot_keypoint_name', type=str, default=None,
                        help='Name of keypoint for plot label (default: auto-detected)')
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
                      args.floor_map_background, args.floor_map_rotation,
                      args.plot_keypoint, args.plot_keypoint_name, args.batch_size)
