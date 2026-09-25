"""
Full inference pipeline: Detection -> Tracking -> Pose Estimation -> Segmentation -> (Optional) Face analysis -> (Optional) Depth
Processes a video and outputs all results overlaid on the same frame
"""

from physiotrack import (Pose, Video, Models, Detection, Tracker, TrackerConfig, Segmentation,
                         VRFace, FaceOrientation, FaceLandmarks, FaceExpression,
                         GazeEstimator, FaceQuality, FaceRegions, Depth)
from pathlib import Path
import argparse

# --face_stages names -> constructors, in the order they must run (gaze needs landmarks).
FACE_STAGES = {
    "orientation": lambda: FaceOrientation(model=Models.Face.Orientation.VR, device=0),
    "landmarks": lambda: FaceLandmarks(),
    "expression": lambda: FaceExpression(device=0),
    "gaze": lambda: GazeEstimator(device=0),
    "quality": lambda: FaceQuality(),
    "regions": lambda: FaceRegions(device=0),
}


def run_full_inference(video_path, output_dir='output/full_inference', floor_map=None,
                       floor_map_background=None, floor_map_rotation=0,
                       plot_keypoint=None, plot_keypoint_name=None, batch_size=1,
                       enable_face_detection=False, face_stages=None,
                       enable_depth=False, ego_video_path=None, show_output=False,
                       plot_angles=False, angle_joints=None, rom=None, rom_render=True,
                       enable_rppg=False, enable_hrv=False, enable_respiration=False,
                       respiration_source="motion", rppg_method="POS",
                       rppg_window_sec=None):
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
        enable_face_detection: Detect faces and tie each to its tracked person
            (default: False).
        face_stages: Names from ``FACE_STAGES`` to run on every face, e.g.
            ``["orientation", "landmarks"]``; implies face detection (default: None).
        enable_depth: Enable depth estimation (default: False)
        ego_video_path: Path to ego-centric video to overlay (default: None)
        show_output: Display output in real-time during processing (default: False)
        plot_angles: Overlay a live joint-angle panel on the left side (default: False)
        angle_joints: Optional subset of joints to show, e.g.
            ["leftElbow", "rightElbow", "leftKnee", "rightKnee"]; None shows all 8.
        enable_rppg: Overlay contactless rPPG panels (BVP pulse + heart rate) and add
            ``vitals`` to the JSON. The skin ROI is a SegFace segmentation built into
            Video; it reuses the face boxes when face detection is on. Default: False.
        enable_hrv: Overlay a heart-rate-variability panel (RMSSD/SDNN/pNN50/SD1/SD2/
            LF-HF); uses a longer rPPG window. Default: False.
        enable_respiration: Overlay a respiration-rate panel (breaths/min) derived from
            shoulder/torso motion, reusing the pose keypoints (needs pose, not a face).
            Default: False.
        rppg_method: rPPG extraction method: "POS" (default), "CHROM", "LGI" or "OMIT".
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
    print(f"Real-time display: {'Enabled (press q to quit)' if show_output else 'Disabled'}")
    print("="*60)

    # Initialize models
    print("\n[1/5] Initializing Person Detector...")
    detector = Detection.Person(
        model=Models.Detection.YOLO.PERSON.m_person,
        conf=0.25,
        iou=0.45,
        verbose=False,
        device=0
    )

    print("[2/5] Initializing Pose Estimator...")
    pose_estimator = Pose.Custom(
        model=Models.Pose.ViTPose.WholeBody.b_wholebody,
        verbose=False,
        device=0
    )

    print("[3/5] Initializing Tracker...")
    tracker_config = TrackerConfig()
    tracker_config.tracker_type = 'ocsort'
    tracker_config.debug_mode = False
    tracker_config.classes = [0]
    tracker = Tracker(config=tracker_config)

    print("[4/5] Initializing Segmentors...")
    segmentor_person = Segmentation.Person(
        device=0,
        conf=0.24,
        iou=0.4,
        classes=[0],  # Person class only
        filter={'bbox_filter': False},
        verbose=False
    )

    segmentor_vrhead = Segmentation.Custom(
        model=Models.Segmentation.YOLO.VRHEAD.M8_251029,
        device=0,
        conf=0.24,
        iou=0.4,
        # classes=[0],
        filter={
            'bbox_filter': True,
            'detector_index': 0,  # Use detector index 0
            'detector_class_filter': None  # Use all classes
        },
        verbose=False
    )

    # Combine multiple segmentators
    segmentors = [segmentor_person, segmentor_vrhead]

    # Faces: detected per frame, tied to the tracked person, then analysed by the chosen
    # face stages. rPPG / HRV get their skin ROI from SegFace inside Video and reuse
    # these face boxes when present, so a face detector is optional for them.
    stage_names = list(face_stages or [])
    unknown = [name for name in stage_names if name not in FACE_STAGES]
    if unknown:
        raise ValueError(f"Unknown face stage(s) {unknown}; choose from {list(FACE_STAGES)}.")
    face_detector = None
    stages = []
    if enable_face_detection or stage_names:
        print(f"[5/6] Initializing Face Detector{' + ' + ', '.join(stage_names) if stage_names else ''}...")
        face_detector = VRFace(device=0, verbose=False)
        stages = [FACE_STAGES[name]() for name in stage_names]

    # Initialize depth estimator if enabled
    depth_estimator = None
    if enable_depth:
        print("[6/6] Initializing Depth Estimator (DepthAnythingV2; Depth.ZipDepth is a lighter drop-in)...")
        depth_estimator = Depth.DepthAnythingV2Base(
            device=0,
            input_size=518,
            verbose=False
        )

    print("\n✓ All models initialized successfully!")
    print(f"  - Segmentators: {len(segmentors)} (Person + VRHEAD)")
    print(f"  - Face Detection: {'Enabled' if face_detector else 'Disabled'}")
    print(f"  - Face stages: {', '.join(stage_names) if stage_names else 'None'}")
    print(f"  - Depth Estimation: {'Enabled' if depth_estimator else 'Disabled'}")
    print(f"  - Ego Video Overlay: {'Enabled' if ego_video_path else 'Disabled'}")
    print(f"  - Joint-Angle Panel: {'Enabled' if plot_angles else 'Disabled'}")
    print(f"  - Clinical ROM: {'Enabled' if rom else 'Disabled'}"
          + (" (skeleton panel off)" if rom and not rom_render else ""))
    print(f"  - rPPG heart rate: {'Enabled (' + rppg_method + ')' if enable_rppg else 'Disabled'}")
    print(f"  - HRV: {'Enabled' if enable_hrv else 'Disabled'}")
    print(f"  - Respiration: {'Enabled' if enable_respiration else 'Disabled'}")

    # Process video using Video processor
    print("\n" + "="*60)
    print("Processing Video")
    print("="*60)

    video_processor = Video(
        source=video_path,
        pose=pose_estimator,
        detector=detector,  # Person detector to work with Pose.Custom
        tracker=tracker,
        segmenter=segmentors,  # Pass list of segmenters (Person + VRHEAD)
        face=face_detector,  # Faces, tied to the tracked person
        face_stages=stages,  # Per-face analysis, in order
        depth=depth_estimator,  # Depth estimator (DepthAnythingV2)
        ego_video=ego_video_path,  # Ego-centric video overlay
        fps=None,
        resize=None,
        rotate=False,
        floor_map=floor_map,  # Floor area for radar view
        floor_map_background=floor_map_background,  # Background mode: None/"default", "auto"/"extract", or path to image
        floor_map_rotation=floor_map_rotation,  # Rotation: 0, 90, 180, or 270 degrees
        plot_keypoint=plot_keypoint,  # Keypoint ID to plot motion (relative to pelvis)
        plot_keypoint_name=plot_keypoint_name,  # Keypoint name for plot label
        plot_angles=plot_angles,  # Live joint-angle panel on the left side
        angle_joints=angle_joints,  # Optional subset of joints (None = all 8)
        rom=rom,  # Clinical ROM (flexion/extension/abduction/adduction)
        rom_render=rom_render,  # render the white-background ROM skeleton panel (right side)
        rppg=enable_rppg,  # contactless rPPG heart-rate panels + vitals JSON
        hrv=enable_hrv,  # heart-rate-variability panel (uses a longer rPPG window)
        respiration=enable_respiration,  # respiration-rate panel (breaths/min)
        respiration_source=respiration_source,  # "motion" (pose/shoulders) or "pulse" (rPPG)
        rppg_method=rppg_method,  # POS / CHROM / LGI / OMIT
        rppg_window_sec=rppg_window_sec,  # rPPG sliding-window seconds (None -> 60 if hrv else 15)
        output_dir=output_dir,
        verbose=True,
        show_fps=True,
        show=show_output,  # Display output in real-time
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

                # With face detection only (faces carry the tracked person's id)
                python full_inference.py video.mp4 --face

                # Face analysis: head orientation, face mesh, expression and gaze
                python full_inference.py video.mp4 --face_stages orientation,landmarks,expression,gaze

                # With depth estimation
                python full_inference.py video.mp4 --depth

                # Contactless vitals: rPPG heart rate
                python full_inference.py video.mp4 --rppg

                # Heart rate + HRV + respiration together
                python full_inference.py video.mp4 --rppg --hrv --respiration --rppg_method POS

                # With live joint-angle panel (left side; all 8 joints)
                python full_inference.py video.mp4 --angles

                # Joint-angle panel for a subset of joints
                python full_inference.py video.mp4 --angles --angle_joints "leftElbow,rightElbow,leftKnee,rightKnee"

                # Clinical ROM red arcs (default: hip flexion + abduction per side)
                python full_inference.py video.mp4 --rom

                # ROM for specific movements only
                python full_inference.py video.mp4 --rom "leftHipFlexion,rightHipFlexion"

                # ROM computed but the skeleton panel hidden
                python full_inference.py video.mp4 --rom --no_rom_render

                # With ego video overlay
                python full_inference.py video.mp4 --ego_video path/to/ego_video.mp4

                # Display output in real-time while processing
                python full_inference.py video.mp4 --show

                # Combine multiple options
                python full_inference.py --floor_map "314,824,778,402,1140,456,936,1035" "kinect_s1_v3.mkv" \\
                    --batch_size 2 --plot_keypoint 9 --plot_keypoint_name "left_wrist" \\
                    --show --face_stages orientation --depth --ego_video ego.mp4

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
    parser.add_argument('--face', action='store_true',
                        help='Detect faces and tie each to its tracked person')
    parser.add_argument('--face_stages', type=str, default=None,
                        help='Comma-separated face stages to run on every face, in order '
                             '(implies --face): ' + ', '.join(FACE_STAGES))
    parser.add_argument('--depth', action='store_true',
                        help='Enable depth estimation using DepthAnythingV2')
    parser.add_argument('--ego_video', type=str, default=None,
                        help='Path to ego-centric video to overlay on output')
    parser.add_argument('--show', action='store_true',
                        help='Display output in real-time during processing (press q to quit)')
    parser.add_argument('--angles', action='store_true',
                        help='Overlay a live joint-angle panel (L/R shoulder, elbow, hip, knee) on the left side')
    parser.add_argument('--angle_joints', type=str, default=None,
                        help='Comma-separated subset of joints, e.g. "leftElbow,rightElbow,leftKnee,rightKnee" (default: all 8)')
    parser.add_argument('--rom', nargs='?', const='__default__', default=None,
                        help='Clinical ROM. Use alone for the default set (hip flexion + abduction '
                             'per side), or pass a comma-separated movement list, e.g. '
                             '--rom "leftHipFlexion,rightHipFlexion,leftHipAbduction,rightHipAbduction". '
                             'Valid: {left,right}Hip{Flexion,Extension,Abduction,Adduction}.')
    parser.add_argument('--no_rom_render', action='store_true',
                        help='With --rom: compute ROM but hide the right-side ROM skeleton panel')
    parser.add_argument('--rppg', action='store_true',
                        help='Overlay contactless rPPG heart-rate panels')
    parser.add_argument('--hrv', action='store_true',
                        help='Overlay a heart-rate-variability panel (RMSSD/SDNN/SD1/SD2/LF-HF; needs a face)')
    parser.add_argument('--respiration', action='store_true',
                        help='Overlay a respiration-rate panel (breaths/min); source '
                             'set by --respiration_source')
    parser.add_argument('--respiration_source', type=str, default='motion',
                        choices=['motion', 'pulse'],
                        help='Respiration signal: "motion" (shoulder/torso motion, reuses '
                             'pose keypoints, no face) or "pulse" (rPPG amplitude, needs a '
                             'face). Default: motion')
    parser.add_argument('--rppg_method', type=str, default='POS',
                        choices=['POS', 'CHROM', 'LGI', 'OMIT'],
                        help='rPPG extraction method (default: POS)')
    parser.add_argument('--rppg_window_sec', type=float, default=None,
                        help='rPPG sliding-window length in seconds (default: 60 with '
                             '--hrv, else 15). Use a smaller value (e.g. 10) for short clips.')

    args = parser.parse_args()

    # Parse floor_map if provided
    floor_map = None
    if args.floor_map:
        coords = [int(x) for x in args.floor_map.split(',')]
        if len(coords) == 8:
            floor_map = [(coords[i], coords[i+1]) for i in range(0, 8, 2)]
        else:
            print("Warning: floor_map must have 8 values (4 points with x,y). Ignoring floor_map.")

    # Parse optional joint subset for the angle panel
    angle_joints = [j.strip() for j in args.angle_joints.split(',')] if args.angle_joints else None

    # Resolve ROM from the single --rom flag:
    #   absent          -> None  (off)
    #   present, no val -> True  (default movement set)
    #   present + list  -> explicit movement list
    if args.rom is None:
        rom = None
    elif args.rom == '__default__':
        rom = True
    else:
        rom = [m.strip() for m in args.rom.split(',')]

    run_full_inference(args.video_path, args.output_dir, floor_map,
                      args.floor_map_background, args.floor_map_rotation,
                      args.plot_keypoint, args.plot_keypoint_name, args.batch_size,
                      args.face,
                      [n.strip() for n in args.face_stages.split(',')] if args.face_stages else None,
                      args.depth, args.ego_video, args.show,
                      plot_angles=args.angles, angle_joints=angle_joints,
                      rom=rom, rom_render=not args.no_rom_render,
                      enable_rppg=args.rppg, enable_hrv=args.hrv,
                      enable_respiration=args.respiration,
                      respiration_source=args.respiration_source,
                      rppg_method=args.rppg_method,
                      rppg_window_sec=args.rppg_window_sec)
