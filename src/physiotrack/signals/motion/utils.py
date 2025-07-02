import pandas as pd
import os
import numpy as np
from physiotrack.pose.config import COCO_WHOLEBODY
import sys
import json
import cv2


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


def extract_key_point_sequence(data, keypoint_id, original_fps):
    """
    Extract the sequence of a given keypoint (by keypoint_id) across all frames
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
    df = pd.DataFrame(sequence_data, columns=["time", "frame", "detection_id", "keypoint_name", "x", "y", "confidence"])
    # df["time"] = df["frame_id"] / original_fps
    return df


def add_head_centroid(data):
    for frame_info in data.get("frame_data", []):
        for detection in frame_info.get("detections", []):
            x_coords = []
            y_coords = []
            confidences = []
            for kp in detection.get("keypoints", []):
                if 23 <= kp.get("id") <= 90:  # Use face keypoints
                    x_coords.append(kp.get("y"))
                    y_coords.append(kp.get("x"))
                    confidences.append(kp.get("confidence"))
            if x_coords and y_coords:
                centroid_x = np.mean(x_coords)
                centroid_y = np.mean(y_coords)
                centroid_confidence = np.mean(confidences)
                detection["keypoints"].append({
                    "id": 133,
                    "x": centroid_y,
                    "y": centroid_x,
                    "confidence": centroid_confidence
                })
    return data


def add_body_centroid(data):
    for frame_info in data.get("frame_data", []):
        for detection in frame_info.get("detections", []):
            x_coords = []
            y_coords = []
            confidences = []
            for kp in detection.get("keypoints", []):
                if 0 <= kp.get("id") <= 132:  # Use face keypoints
                    x_coords.append(kp.get("y"))
                    y_coords.append(kp.get("x"))
                    confidences.append(kp.get("confidence"))
            if x_coords and y_coords:
                centroid_x = np.mean(x_coords)
                centroid_y = np.mean(y_coords)
                centroid_confidence = np.mean(confidences)
                detection["keypoints"].append({
                    "id": 134,
                    "x": centroid_y,
                    "y": centroid_x,
                    "confidence": centroid_confidence
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