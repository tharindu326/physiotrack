import pandas as pd
import os
import numpy as np
from physiotrack.pose.config import COCO_WHOLEBODY, HUMAN26M
import sys
import json
import cv2
from tqdm import tqdm


def read_biosignals(input_file):
    pd.set_option("display.precision", 15)
    readed = False
    biosignals = None
    if os.path.isfile(input_file):
        readed = True
        biosignals = pd.read_csv(input_file, index_col=False, dtype='str')
        # Remove columns where all values are NaN
        biosignals.dropna(axis=1, how='all', inplace=True)
        # Convert only numeric columns to float
        for col in biosignals.columns:
            # Skip conversion if the column cannot be converted to float
            try:
                biosignals[col] = biosignals[col].astype('float64')
            except ValueError:
                pass  # Skip non-numeric columns
        # biosignals = biosignals.astype('float64')
    return biosignals, readed


def extract_keypoint_sequence_2d(data, keypoint_id, original_fps):
    """
    Extract the sequence of a given 2D keypoint (by keypoint_id) across all frames
    """
    sequence_data = []
    for frame_info in data:
        frame_number = frame_info.get("frame_id")
        timestamp = frame_info.get("timestamp")
        for detection in frame_info.get("detections", []):
            detection_id = detection.get("id", None)
            for kp in detection.get("keypoints", []):
                if kp.get("id") == keypoint_id:
                    sequence_data.append({
                        "time": timestamp,
                        "frame": frame_number,
                        "detection_id": detection_id,
                        "keypoint_name": COCO_WHOLEBODY[str(keypoint_id)],
                        "keypoint_id": keypoint_id,
                        "y": kp.get("y"),
                        "x": kp.get("x"),
                        "confidence": kp.get("confidence")
                    })
    df = pd.DataFrame(sequence_data, columns=["time", "frame", "detection_id", "keypoint_id", "x", "y", "confidence"])
    return df

def extract_keypoint_sequence_3d(data, keypoint_id, original_fps):
    """
    Extract the sequence of a given 3D keypoint (by keypoint_id) across all frames
    """
    sequence_data = []
    for frame_info in data:
        frame_number = frame_info.get("frame_id")
        timestamp = frame_info.get("timestamp")
        for detection in frame_info.get("detections", []):
            detection_id = detection.get("id", None)
            for kp in detection.get("keypoints3D", []):
                if kp.get("id") == keypoint_id:
                    sequence_data.append({
                        "time": timestamp,
                        "frame": frame_number,
                        "detection_id": detection_id,
                        "keypoint_name": HUMAN26M[int(keypoint_id)],
                        "keypoint_id": keypoint_id,
                        "x": kp.get("x"),
                        "y": kp.get("y"),
                        "z": kp.get("z")
                    })
    df = pd.DataFrame(sequence_data, columns=["time", "frame", "detection_id", "keypoint_id", "x", "y", "z"])
    return df

def extract_keypoints_sequence(data, candidate_key_points=list(range(17))):
    """
    Extract the sequence of keypoints (2D and 3D) across all frames
    """
    sequence_data = []
    columns = ['time', 'frame', 'detection_id']
    processed_key_points_2d = []
    processed_key_points_3d = []
        
    for frame_info in tqdm(data, desc="Extracting key points", unit="frame"):
        frame_number = frame_info.get("frame_id")
        timestamp = frame_info.get("timestamp")
        detections = frame_info.get("detections", [])
        
        for detection in detections:
            detection_id = detection.get("id", None)
            frame_data = {
                "time": timestamp,
                "frame": frame_number,
                "detection_id": detection_id
            }
            
            # Process 2D keypoints
            for kp in detection.get("keypoints", []):
                keypoint_id = kp.get("id")
                if keypoint_id in candidate_key_points:
                    frame_data[f"{keypoint_id}_x"] = kp.get("x")
                    frame_data[f"{keypoint_id}_y"] = kp.get("y")
                    frame_data[f"{keypoint_id}_confidence"] = kp.get("confidence")
                    if keypoint_id not in processed_key_points_2d:
                        columns.extend([f"{keypoint_id}_x", f"{keypoint_id}_y", f"{keypoint_id}_confidence"])
                        processed_key_points_2d.append(keypoint_id)
            
            # Process 3D keypoints
            for kp3d in detection.get("keypoints3D", []):
                keypoint_id = kp3d.get("id")
                if keypoint_id in candidate_key_points:
                    frame_data[f"3d_{keypoint_id}_x"] = kp3d.get("x")
                    frame_data[f"3d_{keypoint_id}_y"] = kp3d.get("y")
                    frame_data[f"3d_{keypoint_id}_z"] = kp3d.get("z")
                    if keypoint_id not in processed_key_points_3d:
                        columns.extend([f"3d_{keypoint_id}_x", f"3d_{keypoint_id}_y", f"3d_{keypoint_id}_z"])
                        processed_key_points_3d.append(keypoint_id)
            
            sequence_data.append(frame_data)
    
    df = pd.DataFrame(sequence_data, columns=columns)
    return df

def add_head_centroid(data, pose_architecture):
    """
    Add head centroid keypoint to both 2D and 3D data
    """
    for frame_info in data:
        for detection in frame_info.get("detections", []):
            # 2D head centroid
            x_coords_2d = []
            y_coords_2d = []
            confidences_2d = []
            
            # 3D head centroid
            x_coords_3d = []
            y_coords_3d = []
            z_coords_3d = []
            
            # Process 2D keypoints
            for kp in detection.get("keypoints", []):
                if pose_architecture == 'WHOLEBODY':
                    if 23 <= kp.get("id") <= 90:  # Face keypoints
                        x_coords_2d.append(kp.get("x"))
                        y_coords_2d.append(kp.get("y"))
                        confidences_2d.append(kp.get("confidence"))
                elif pose_architecture == 'COCO':
                    if 0 <= kp.get("id") <= 4:  # Face keypoints
                        x_coords_2d.append(kp.get("x"))
                        y_coords_2d.append(kp.get("y"))
                        confidences_2d.append(kp.get("confidence"))
                else:
                    raise ValueError(f"[ERROR] pose_architecture: {pose_architecture} is unknown. It should be either WHOLEBODY or COCO")
            
            # Add 2D head centroid if coordinates exist
            if x_coords_2d and y_coords_2d:
                centroid_x_2d = np.mean(x_coords_2d)
                centroid_y_2d = np.mean(y_coords_2d)
                centroid_confidence_2d = np.mean(confidences_2d)
                detection["keypoints"].append({
                    "id": 133,
                    "x": centroid_x_2d,
                    "y": centroid_y_2d,
                    "confidence": centroid_confidence_2d
                })
            
            # Process 3D keypoints if they exist
            if "keypoints3D" in detection:
                for kp3d in detection.get("keypoints3D", []):
                    if pose_architecture == 'WHOLEBODY':
                        if 23 <= kp3d.get("id") <= 90:  # Face keypoints
                            x_coords_3d.append(kp3d.get("x"))
                            y_coords_3d.append(kp3d.get("y"))
                            z_coords_3d.append(kp3d.get("z"))
                    elif pose_architecture == 'COCO':
                        if 0 <= kp3d.get("id") <= 4:  # Face keypoints
                            x_coords_3d.append(kp3d.get("x"))
                            y_coords_3d.append(kp3d.get("y"))
                            z_coords_3d.append(kp3d.get("z"))
                
                # Add 3D head centroid if coordinates exist
                if x_coords_3d and y_coords_3d and z_coords_3d:
                    centroid_x_3d = np.mean(x_coords_3d)
                    centroid_y_3d = np.mean(y_coords_3d)
                    centroid_z_3d = np.mean(z_coords_3d)
                    
                    if "keypoints3D" not in detection:
                        detection["keypoints3D"] = []
                    
                    detection["keypoints3D"].append({
                        "id": 133,
                        "x": centroid_x_3d,
                        "y": centroid_y_3d,
                        "z": centroid_z_3d
                    })
    
    return data


def add_body_centroid(data, pose_architecture):
    """
    Add body centroid keypoint to both 2D and 3D data
    """
    for frame_info in data:
        for detection in frame_info.get("detections", []):
            # 2D body centroid
            x_coords_2d = []
            y_coords_2d = []
            confidences_2d = []
            
            # 3D body centroid
            x_coords_3d = []
            y_coords_3d = []
            z_coords_3d = []
            
            # Process 2D keypoints
            for kp in detection.get("keypoints", []):
                if 0 <= kp.get("id") <= 16:  # COCO body keypoints
                    x_coords_2d.append(kp.get("x"))
                    y_coords_2d.append(kp.get("y"))
                    confidences_2d.append(kp.get("confidence"))
            
            # Add 2D body centroid if coordinates exist
            if x_coords_2d and y_coords_2d:
                centroid_x_2d = np.mean(x_coords_2d)
                centroid_y_2d = np.mean(y_coords_2d)
                centroid_confidence_2d = np.mean(confidences_2d)
                detection["keypoints"].append({
                    "id": 134,
                    "x": centroid_x_2d,
                    "y": centroid_y_2d,
                    "confidence": centroid_confidence_2d
                })
            
            # Process 3D keypoints if they exist
            if "keypoints3D" in detection:
                for kp3d in detection.get("keypoints3D", []):
                    if 0 <= kp3d.get("id") <= 16:  # COCO body keypoints
                        x_coords_3d.append(kp3d.get("x"))
                        y_coords_3d.append(kp3d.get("y"))
                        z_coords_3d.append(kp3d.get("z"))
                
                # Add 3D body centroid if coordinates exist
                if x_coords_3d and y_coords_3d and z_coords_3d:
                    centroid_x_3d = np.mean(x_coords_3d)
                    centroid_y_3d = np.mean(y_coords_3d)
                    centroid_z_3d = np.mean(z_coords_3d)
                    
                    if "keypoints3D" not in detection:
                        detection["keypoints3D"] = []
                    
                    detection["keypoints3D"].append({
                        "id": 134,
                        "x": centroid_x_3d,
                        "y": centroid_y_3d,
                        "z": centroid_z_3d
                    })
    
    return data


def add_pelvic_centroid(data, pose_architecture):
    """
    Add pelvic centroid keypoint to both 2D and 3D data.
    Calculated as the midpoint between left and right hip keypoints.
    """
    for frame_info in data:
        for detection in frame_info.get("detections", []):
            # 2D pelvic centroid
            left_hip_2d = None
            right_hip_2d = None
            
            # 3D pelvic centroid
            left_hip_3d = None
            right_hip_3d = None
            
            # Find 2D hip keypoints
            for kp in detection.get("keypoints", []):
                if kp.get("id") == 11:  # Left hip
                    left_hip_2d = kp
                elif kp.get("id") == 12:  # Right hip
                    right_hip_2d = kp
            
            # Add 2D pelvic centroid if both hips are found
            if left_hip_2d and right_hip_2d:
                pelvic_x_2d = (left_hip_2d.get("x") + right_hip_2d.get("x")) / 2
                pelvic_y_2d = (left_hip_2d.get("y") + right_hip_2d.get("y")) / 2
                pelvic_confidence_2d = (left_hip_2d.get("confidence") + right_hip_2d.get("confidence")) / 2
                
                detection["keypoints"].append({
                    "id": 135,
                    "x": pelvic_x_2d,
                    "y": pelvic_y_2d,
                    "confidence": pelvic_confidence_2d
                })
            
            # Process 3D keypoints if they exist
            if "keypoints3D" in detection:
                # Find 3D hip keypoints
                for kp3d in detection.get("keypoints3D", []):
                    if kp3d.get("id") == 11:  # Left hip
                        left_hip_3d = kp3d
                    elif kp3d.get("id") == 12:  # Right hip
                        right_hip_3d = kp3d
                
                # Add 3D pelvic centroid if both hips are found
                if left_hip_3d and right_hip_3d:
                    pelvic_x_3d = (left_hip_3d.get("x") + right_hip_3d.get("x")) / 2
                    pelvic_y_3d = (left_hip_3d.get("y") + right_hip_3d.get("y")) / 2
                    pelvic_z_3d = (left_hip_3d.get("z") + right_hip_3d.get("z")) / 2
                    
                    if "keypoints3D" not in detection:
                        detection["keypoints3D"] = []
                    
                    detection["keypoints3D"].append({
                        "id": 135,
                        "x": pelvic_x_3d,
                        "y": pelvic_y_3d,
                        "z": pelvic_z_3d
                    })
    
    return data


def resample_to_match_reference(input_signal, reference_signal):
    """
    Resamples the input signal using linear interpolation to match
    the length of the reference signal.

    Args:
        input_signal (numpy.ndarray): The signal to be resampled.
        reference_signal (numpy.ndarray): The reference signal whose length is the target.

    Returns:
        numpy.ndarray: Resampled input signal with the same length as the reference signal.
    """
    target_length = len(reference_signal)  # Desired length
    current_length = len(input_signal)  # Original length

    if current_length == target_length:
        return input_signal  # No resampling needed

    # Generate new sample positions (evenly spaced)
    resampled_signal = np.interp(
        np.linspace(0.0, 1.0, target_length, endpoint=False),  # New positions
        np.linspace(0.0, 1.0, current_length, endpoint=False),  # Original positions
        input_signal  # Data values
    )

    return resampled_signal


def resample_by_interpolation(signal, input_fs, output_fs):
    scale = output_fs / input_fs
    # calculate new length of sample
    n = round(len(signal) * scale)

    # use linear interpolation
    # endpoint keyword means than linspace doesn't go all the way to 1.0
    # If it did, there are some off-by-one errors
    # e.g. scale=2.0, [1,2,3] should go to [1,1.5,2,2.5,3,3]
    # but with endpoint=True, we get [1,1.4,1.8,2.2,2.6,3]
    # Both are OK, but since resampling will often involve
    # exact ratios (i.e. for 44100 to 22050 or vice versa)
    # using endpoint=False gets less noise in the resampled sound
    resampled_signal = np.interp(
        np.linspace(0.0, 1.0, n, endpoint=False),  # where to interpret
        np.linspace(0.0, 1.0, len(signal), endpoint=False),  # known positions
        signal,  # known data points
    )
    # print(len(resampled_signal))
    return resampled_signal


def resample_dataframe_by_interpolation(df, input_fs, output_fs, columns_to_resample=None):
    """
    Resample DataFrame columns using linear interpolation.
    """
    scale = output_fs / input_fs
    n = round(len(df) * scale)
    
    if columns_to_resample is None:
        columns_to_resample = df.select_dtypes(include=[np.number]).columns.tolist()
    
    resampled_data = {}
    
    for col in columns_to_resample:
        if col in df.columns:
            signal = df[col].values
            resampled_signal = np.interp(
                np.linspace(0.0, 1.0, n, endpoint=False),
                np.linspace(0.0, 1.0, len(signal), endpoint=False),
                signal
            )
            resampled_data[col] = resampled_signal
    
    resampled_df = pd.DataFrame(resampled_data)
    if 'time' in df.columns:
        resampled_df['time'] = np.linspace(0, len(df) / input_fs, n, endpoint=False)
    return resampled_df