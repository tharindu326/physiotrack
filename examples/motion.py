from physiotrack import Pose, Video, Models, Detection, Tracker
from physiotrack.trackers import Config
from physiotrack.pose.config import COCO_WHOLEBODY_NAMES, HUMAN26M_NAMES
from physiotrack.signals.motion.utils import extract_keypoint_sequence_3d, extract_keypoint_sequence_2d, add_body_centroid, add_head_centroid, resample_dataframe_by_interpolation, add_pelvic_centroid, extract_keypoints_sequence
from physiotrack.signals.motion.features import get_relative_coordinates, compute_all_motion_features, get_keypoint_features, select_feature_data
from physiotrack.signals.normalize import min_max_normalize
from physiotrack.signals.filters import band_pass_filter
from physiotrack.pose.pose3D import Pose3D
from physiotrack.pose.canonicalizer import PoseCanonicalizer
from physiotrack.signals.evaluate import calculate_pearson_correlation, calculate_dtw_distance, normalized_cross_correlation, phase_synchrony, compute_rmse, compute_plv
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import json
import numpy as np

def run_motion_pipeline():
    """Run the motion analysis pipeline"""
    
    # Initialize models
    print("Initializing models...")
    pose_estimator = Pose.Custom(
        model=Models.Pose.ViTPose.WholeBody.b_WHOLEBODY, 
        render_box_detections=False, 
        render_labels=True, 
        overlay_keypoints=True, 
        verbose=False, 
        device=0
    )
    
    detector = Detection.Person(
        model=Models.Detection.YOLO.PERSON.m_person, 
        render_box_detections=False, 
        render_labels=False, 
        verbose=False, 
        device=0
    )
    
    TrackerConfig = Config()
    TrackerConfig.tracker_type = 'ocsort'
    TrackerConfig.debug_mode = True
    TrackerConfig.classes = [0]
    TrackerConfig.enable_student_tracking = True
    tracker = Tracker(config=TrackerConfig)
    
    # Setup paths
    input_video = 'CVSSP3D/s1/rom1/TC_S1_rom1_cam1.mp4'
    output_directory = 'output/CVSSP3D_s1_rom1_cam1'
    input_path = Path(input_video)
    video_name = input_path.stem
    
    # Process 2D poses
    print("Processing 2D poses...")
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
    
    # Process 3D poses
    print("Processing 3D poses...")
    pose3D = Pose3D(
        model=Models.Pose3D.MotionBERT.MB_ft_h36m_global_lite,
        config=None,
        device='cuda',
        clip_len=243,
        pixel=False,
        render_video=True,
        save_npy=True,
        testloader_params=None
    )
    
    detection_data_3D, results_3d = pose3D.estimate(
        json_path=json_output_path,
        vid_path=input_video,
        out_path=output_directory,
        canonical_view=Models.Pose3D.Canonicalizer.View.FRONT,
        canonical_model=Models.Pose3D.Canonicalizer.Models._3DPCNetS2
    )

    # input_video = 'CVSSP3D/s1/rom1/TC_S1_rom1_cam1.mp4'
    # output_directory = 'output/CVSSP3D_s1_rom1_cam1_DDH'
    # input_path = Path(input_video)
    # video_name = input_path.stem
    
    # # Process 2D poses
    # print("Processing 2D poses...")
    # video_processor = Video(
    #     video_path=input_video,
    #     pose_estimator=pose_estimator,
    #     detector=detector,
    #     tracker=tracker,
    #     required_fps=None,
    #     frame_resize=None,
    #     frame_rotate=False,
    #     output_path=output_directory,
    #     verbose=True
    # )
    
    # # video_output_path = Path(output_directory) / f"{video_name}_poses.mp4"
    # # json_output_path = Path(output_directory) / f"{video_name}_result.json"
    # # detection_data_2D = video_processor.run(video_output_path, json_output_path)
    # json_output_path = 'output/CVSSP3D_s1_rom1_cam1/TC_S1_rom1_cam1_result.json'
    # detection_data_2D =None
    # sampling_freq = video_processor.video_fps
    
    # # Process 3D poses
    # print("Processing 3D poses...")
    # pose3D = Pose3D(
    #     model=Models.Pose3D.DDH.best,
    #     config=None,
    #     device='cuda',
    #     clip_len=243,
    #     pixel=False,
    #     render_video=True,
    #     save_npy=True,
    #     testloader_params=None,
    #     num_proposals=10,
    #     sampling_timesteps=5
    # )
    
    # detection_data_3D, results_3d = pose3D.estimate(
    #     json_path=json_output_path,
    #     vid_path='BV_S17_cut1.mp4',
    #     out_path=output_directory,
    #     canonical_view=Models.Pose3D.Canonicalizer.View.FRONT,
    #     canonical_model=Models.Pose3D.Canonicalizer.Models._3DPCNetS2,
    #     batch_size=1  # Small batch size for DDH to avoid OOM
    # )
    
    return detection_data_2D, detection_data_3D, results_3d, pose_estimator, sampling_freq, output_directory, video_name


if __name__ == "__main__":
    # This block only runs when the script is executed directly,
    # not when it's imported by parallel workers
    detection_data_2D, detection_data_3D, results_3d, pose_estimator, sampling_freq, output_directory, video_name = run_motion_pipeline()

    # file_path = 'CVSSP3D/s1/acting1/poses_3d_cvssp3d_s1_acting1.json'
    # with open(file_path, 'r', encoding='utf-8') as file:
    #     detection_data_3D = json.load(file)

    detection_data = add_body_centroid(detection_data_3D, pose_estimator.archetecture)
    detection_data = add_head_centroid(detection_data, pose_estimator.archetecture)
    detection_data = add_pelvic_centroid(detection_data, pose_estimator.archetecture)

    # keypoint extraction
    keypoint_id_2d = int(COCO_WHOLEBODY_NAMES['left_wrist'])
    keypoint_id_3d = int(HUMAN26M_NAMES['left_wrist'])
    feature_type = 'coordinates'  # 'coordinates', 'velocity', 'acceleration', or 'angles'

    # keypoint_df_2d = extract_keypoint_sequence_2d(detection_data, keypoint_id, original_fps=video_processor.video_fps)
    # keypoint_df_3d = extract_keypoint_sequence_3d(detection_data, keypoint_id, original_fps=video_processor.video_fps)

    keypoints_df = extract_keypoints_sequence(detection_data, candidate_key_points=list(range(17)) + [135])
    # rel_coords = get_relative_coordinates(keypoints_df, reference_point_id=135) # pelvic
    motion_df = compute_all_motion_features(keypoints_df)
    keypoint_df_2d, keypoint_df_3d = get_keypoint_features(motion_df, keypoint_id_2d, keypoint_id_3d)
    selected_features = select_feature_data(keypoint_id_2d, keypoint_id_3d, feature_type)


    # 2D Processing - use selected features
    keypoint_df_2d = keypoint_df_2d.dropna(subset=selected_features['2d_features'])
    keypoint_df_2d = resample_dataframe_by_interpolation(keypoint_df_2d, input_fs=sampling_freq, output_fs=30, columns_to_resample=None)
    keypoint_df_2d['norm_x'] = min_max_normalize(keypoint_df_2d[selected_features['2d_features'][0]], feature_range=(0, 1))
    keypoint_df_2d['norm_y'] = min_max_normalize(keypoint_df_2d[selected_features['2d_features'][1]], feature_range=(0, 1))

    x = keypoint_df_2d['norm_x'].to_numpy()
    y = keypoint_df_2d['norm_y'].to_numpy()
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

    # 3D Processing - use selected features  
    keypoint_df_3d = keypoint_df_3d.dropna(subset=selected_features['3d_features'])
    keypoint_df_3d = resample_dataframe_by_interpolation(keypoint_df_3d, input_fs=sampling_freq, output_fs=30, columns_to_resample=None)
    keypoint_df_3d['norm_x'] = min_max_normalize(keypoint_df_3d[selected_features['3d_features'][0]], feature_range=(0, 1))
    keypoint_df_3d['norm_y'] = min_max_normalize(keypoint_df_3d[selected_features['3d_features'][1]], feature_range=(0, 1))
    keypoint_df_3d['norm_z'] = min_max_normalize(keypoint_df_3d[selected_features['3d_features'][2]], feature_range=(0, 1))

    x_3d = keypoint_df_3d['norm_x'].to_numpy()
    y_3d = keypoint_df_3d['norm_y'].to_numpy()
    z_3d = keypoint_df_3d['norm_z'].to_numpy()
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

    print(f"Length of estimated signals: {len(x)}")
    print(f"Length of gt signals: {len(x_3d)}")

    pearson_x = calculate_pearson_correlation(x_3d, x)
    dtw_x = calculate_dtw_distance(x_3d, x)
    ncc_x = normalized_cross_correlation(x_3d, x)
    rmse_x = compute_rmse(x_3d, x)
    phase_sync = phase_synchrony(x_3d, x)
    plv_value = compute_plv(x_3d, x)

    print("\n")
    print(f"Pearson Correlation (x-axis signals): {pearson_x}")
    print(f"DTW Distance (x-axis signals): {dtw_x}")
    print(f"NCC Distance (x-axis signals): {ncc_x}")
    print(f"RMSE x-axis: {rmse_x:.3f}")
    print(f"Phase Synchronization x-axis: {phase_sync:.3f}")
    print(f"Phase Locking Value (PLV) x-axis: {plv_value:.3f}")
    print("\n")
    print("\n")
