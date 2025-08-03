import numpy as np
import pandas as pd


def compute_velocity(rel_df):
    """
    Computes the velocity for each keypoint coordinate (2D and 3D) in the relative DataFrame.
    Velocity is computed as: V_t = (P_t - P_{t-1}) / dt
    where dt is dynamically computed using the 'time' column.
    Args:
        rel_df (pd.DataFrame): DataFrame with relative keypoint coordinates.
    Returns:
        pd.DataFrame: A new DataFrame with velocity columns.
    """
    velocity_df = pd.DataFrame(index=rel_df.index)
    
    if "time" not in rel_df.columns:
        raise ValueError("The 'time' column is required to compute velocity but is missing.")
    
    dt = rel_df["time"].diff()  # Compute per-frame time difference
    
    # Find 2D keypoint columns
    keypoint_2d_columns = [col for col in rel_df.columns 
                          if (col.endswith('_x') or col.endswith('_y')) 
                          and not col.startswith('3d_')
                          and col != 'time']
    
    # Find 3D keypoint columns
    keypoint_3d_columns = [col for col in rel_df.columns 
                          if col.startswith('3d_') 
                          and (col.endswith('_x') or col.endswith('_y') or col.endswith('_z'))
                          and col != 'time']
    
    # Compute 2D velocities using V_t = (P_t - P_{t-1}) / dt
    for col in keypoint_2d_columns:
        velocity_df[f"vel_2d_{col}"] = rel_df[col].diff() / dt
    
    # Compute 3D velocities using V_t = (P_t - P_{t-1}) / dt
    for col in keypoint_3d_columns:
        # Remove '3d_' prefix from column name for cleaner naming
        clean_col = col.replace('3d_', '')
        velocity_df[f"vel_3d_{clean_col}"] = rel_df[col].diff() / dt
    
    return velocity_df


def compute_acceleration(rel_df):
    """
    Computes the acceleration for each keypoint coordinate (2D and 3D) in the relative DataFrame.
    Acceleration is computed as: A_t = (P_t - 2 * P_{t-1} + P_{t-2}) / (dt^2)
    where dt is dynamically computed using the 'time' column.
    
    Args:
        rel_df (pd.DataFrame): DataFrame with relative keypoint coordinates.
    Returns:
        pd.DataFrame: A new DataFrame with acceleration columns.
    """
    acceleration_df = pd.DataFrame(index=rel_df.index)
    
    if "time" not in rel_df.columns:
        raise ValueError("The 'time' column is required to compute acceleration but is missing.")
    
    dt = rel_df["time"].diff()  # Compute per-frame time difference
    dt_squared = dt ** 2        # Compute dt^2
    
    # Find 2D keypoint columns
    keypoint_2d_columns = [col for col in rel_df.columns 
                          if (col.endswith('_x') or col.endswith('_y')) 
                          and not col.startswith('3d_')
                          and col != 'time']
    
    # Find 3D keypoint columns
    keypoint_3d_columns = [col for col in rel_df.columns 
                          if col.startswith('3d_') 
                          and (col.endswith('_x') or col.endswith('_y') or col.endswith('_z'))
                          and col != 'time']
    
    # Compute 2D accelerations using A_t = (P_t - 2 * P_{t-1} + P_{t-2}) / (dt^2)
    for col in keypoint_2d_columns:
        # Manual calculation: P_t - 2*P_{t-1} + P_{t-2}
        P_t = rel_df[col]
        P_t_minus_1 = rel_df[col].shift(1)
        P_t_minus_2 = rel_df[col].shift(2)
        acceleration_df[f"acc_2d_{col}"] = (P_t - 2 * P_t_minus_1 + P_t_minus_2) / dt_squared
    
    # Compute 3D accelerations using A_t = (P_t - 2 * P_{t-1} + P_{t-2}) / (dt^2)
    for col in keypoint_3d_columns:
        # Manual calculation: P_t - 2*P_{t-1} + P_{t-2}
        P_t = rel_df[col]
        P_t_minus_1 = rel_df[col].shift(1)
        P_t_minus_2 = rel_df[col].shift(2)
        # Remove '3d_' prefix from column name for cleaner naming
        clean_col = col.replace('3d_', '')
        acceleration_df[f"acc_3d_{clean_col}"] = (P_t - 2 * P_t_minus_1 + P_t_minus_2) / dt_squared
    
    return acceleration_df


def compute_joint_angle_2d(A, B, C):
    """
    Computes the joint angle at point B given three 2D keypoints A, B, and C using the cosine rule.
    Args:
        A, B, C: Array-like or tuple with two elements (x, y) representing the coordinates.
    Returns:
        float: The angle in radians at point B.
    """
    A = np.array(A)
    B = np.array(B)
    C = np.array(C)
    BA = A - B
    BC = C - B
    dot_product = np.dot(BA, BC)
    norm_BA = np.linalg.norm(BA)
    norm_BC = np.linalg.norm(BC)
    
    # Avoid division by zero
    if norm_BA == 0 or norm_BC == 0:
        return np.nan
    
    cosine_angle = dot_product / (norm_BA * norm_BC)
    cosine_angle = np.clip(cosine_angle, -1, 1)  # Clip for numerical stability
    angle = np.arccos(cosine_angle)
    return angle


def compute_joint_angle_3d(A, B, C):
    """
    Computes the joint angle at point B given three 3D keypoints A, B, and C using the cosine rule.
    Args:
        A, B, C: Array-like or tuple with three elements (x, y, z) representing the coordinates.
    Returns:
        float: The angle in radians at point B.
    """
    A = np.array(A)
    B = np.array(B)
    C = np.array(C)
    BA = A - B
    BC = C - B
    dot_product = np.dot(BA, BC)
    norm_BA = np.linalg.norm(BA)
    norm_BC = np.linalg.norm(BC)
    
    # Avoid division by zero
    if norm_BA == 0 or norm_BC == 0:
        return np.nan
    
    cosine_angle = dot_product / (norm_BA * norm_BC)
    cosine_angle = np.clip(cosine_angle, -1, 1)  # Clip for numerical stability
    angle = np.arccos(cosine_angle)
    return angle


def compute_all_joint_angles(rel_df):
    """
    Computes joint angles for both 2D and 3D keypoints and adds them as new columns to the DataFrame.
    Joint triplets used:
        - leftShoulder: (left_elbow, left_shoulder, left_hip) -> (7, 5, 11)
        - rightShoulder: (right_elbow, right_shoulder, right_hip) -> (8, 6, 12)
        - leftElbow: (left_shoulder, left_elbow, left_wrist) -> (5, 7, 9)
        - rightElbow: (right_shoulder, right_elbow, right_wrist) -> (6, 8, 10)
        - leftHip: (left_shoulder, left_hip, left_knee) -> (5, 11, 13)
        - rightHip: (right_shoulder, right_hip, right_knee) -> (6, 12, 14)
        - leftKnee: (left_hip, left_knee, left_ankle) -> (11, 13, 15)
        - rightKnee: (right_hip, right_knee, right_ankle) -> (12, 14, 16)
    Args:
        rel_df (pd.DataFrame): DataFrame with relative keypoint coordinates.
    """
    joint_triplets = {
        "leftShoulder": (7, 5, 11),   # left_elbow, left_shoulder, left_hip
        "rightShoulder": (8, 6, 12),  # right_elbow, right_shoulder, right_hip
        "leftElbow": (5, 7, 9),       # left_shoulder, left_elbow, left_wrist
        "rightElbow": (6, 8, 10),     # right_shoulder, right_elbow, right_wrist
        "leftHip": (5, 11, 13),       # left_shoulder, left_hip, left_knee
        "rightHip": (6, 12, 14),      # right_shoulder, right_hip, right_knee
        "leftKnee": (11, 13, 15),     # left_hip, left_knee, left_ankle
        "rightKnee": (12, 14, 16)     # right_hip, right_knee, right_ankle
    }
    
    df_angles = pd.DataFrame(index=rel_df.index)
    
    has_2d_data = any(col.endswith('_x') and not col.startswith('3d_') for col in rel_df.columns)
    has_3d_data = any(col.startswith('3d_') and col.endswith('_x') for col in rel_df.columns)
    
    for joint, triplet in joint_triplets.items():
        A_joint, B_joint, C_joint = triplet
        
        # Compute 2D joint angles if 2D data exists
        if has_2d_data:
            col_name_2d = f"ang_2d_{joint}"
            
            def compute_angle_row_2d(row):
                try:
                    A_coord = (row[f"{A_joint}_x"], row[f"{A_joint}_y"])
                    B_coord = (row[f"{B_joint}_x"], row[f"{B_joint}_y"])
                    C_coord = (row[f"{C_joint}_x"], row[f"{C_joint}_y"])
                except KeyError:
                    return np.nan
                
                # Check for NaN values
                if any(pd.isna(x) for x in (A_coord + B_coord + C_coord)):
                    return np.nan
                
                return compute_joint_angle_2d(A_coord, B_coord, C_coord)
            
            df_angles[col_name_2d] = rel_df.apply(compute_angle_row_2d, axis=1)
        
        # Compute 3D joint angles if 3D data exists
        if has_3d_data:
            col_name_3d = f"ang_3d_{joint}"
            
            def compute_angle_row_3d(row):
                try:
                    A_coord = (row[f"3d_{A_joint}_x"], row[f"3d_{A_joint}_y"], row[f"3d_{A_joint}_z"])
                    B_coord = (row[f"3d_{B_joint}_x"], row[f"3d_{B_joint}_y"], row[f"3d_{B_joint}_z"])
                    C_coord = (row[f"3d_{C_joint}_x"], row[f"3d_{C_joint}_y"], row[f"3d_{C_joint}_z"])
                except KeyError:
                    return np.nan
                
                # Check for NaN values
                if any(pd.isna(x) for x in (A_coord + B_coord + C_coord)):
                    return np.nan
                
                return compute_joint_angle_3d(A_coord, B_coord, C_coord)
            
            df_angles[col_name_3d] = rel_df.apply(compute_angle_row_3d, axis=1)
    
    return df_angles


def compute_all_motion_features(rel_df):
    """
    Computes all motion features (velocity, acceleration, and joint angles) for both 2D and 3D data.
    """
    # velocity
    velocity_df = compute_velocity(rel_df)
    # acceleration
    acceleration_df = compute_acceleration(rel_df)
    # joint angles
    angles_df = compute_all_joint_angles(rel_df)
    motion_features_df = pd.concat([rel_df, velocity_df, acceleration_df, angles_df], axis=1)
    return motion_features_df


def get_relative_coordinates(df, reference_point_id=135):
    """
    Get relative keypoint coordinates by subtracting the selected reference point.
    Args:
        df: DataFrame with keypoint data
        reference_point_id: ID of the reference keypoint (default: 135 for pelvis)
    """
    # Define reference point column names for 2D and 3D
    ref_x_2d = f"{reference_point_id}_x"
    ref_y_2d = f"{reference_point_id}_y"
    ref_x_3d = f"3d_{reference_point_id}_x"
    ref_y_3d = f"3d_{reference_point_id}_y"
    ref_z_3d = f"3d_{reference_point_id}_z"
    
    # Check if reference point exists in the data
    has_2d_ref = ref_x_2d in df.columns and ref_y_2d in df.columns
    has_3d_ref = ref_x_3d in df.columns and ref_y_3d in df.columns and ref_z_3d in df.columns
    
    if not has_2d_ref and not has_3d_ref:
        raise ValueError(f"Reference point ID {reference_point_id} not found in the data.")
    
    relative_data = pd.DataFrame()
    
    # Get reference coordinates
    if has_2d_ref:
        center_x_2d = df[ref_x_2d]
        center_y_2d = df[ref_y_2d]
    
    if has_3d_ref:
        center_x_3d = df[ref_x_3d]
        center_y_3d = df[ref_y_3d]
        center_z_3d = df[ref_z_3d]
    
    # Identify all keypoint columns (2D and 3D) excluding reference point
    reference_cols = [ref_x_2d, ref_y_2d, ref_x_3d, ref_y_3d, ref_z_3d]
    
    # Process 2D keypoints
    if has_2d_ref:
        keypoint_2d_cols = [col for col in df.columns 
                           if (col.endswith('_x') or col.endswith('_y')) 
                           and not col.startswith('3d_')
                           and col not in reference_cols]
        
        for col in keypoint_2d_cols:
            if col.endswith('_x'):
                relative_data[col] = df[col] - center_x_2d
            elif col.endswith('_y'):
                relative_data[col] = df[col] - center_y_2d
    
    # Process 3D keypoints
    if has_3d_ref:
        keypoint_3d_cols = [col for col in df.columns 
                           if col.startswith('3d_') 
                           and (col.endswith('_x') or col.endswith('_y') or col.endswith('_z'))
                           and col not in reference_cols]
        
        for col in keypoint_3d_cols:
            if col.endswith('_x'):
                relative_data[col] = df[col] - center_x_3d
            elif col.endswith('_y'):
                relative_data[col] = df[col] - center_y_3d
            elif col.endswith('_z'):
                relative_data[col] = df[col] - center_z_3d
    
    # Copy non-coordinate columns (like frame, time, etc.)
    non_coord_cols = [col for col in df.columns 
                      if not (col.endswith('_x') or col.endswith('_y') or col.endswith('_z') or col.endswith('_confidence'))
                      or col in ['frame', 'time']]
    
    for col in non_coord_cols:
        if col in df.columns:
            relative_data[col] = df[col]
    
    return relative_data


def get_keypoint_features(motion_df, keypoint_id_2d, keypoint_id_3d=None):
    """
    Get features for a specific keypoint, returning separate 2D and 3D DataFrames
    Args:
        motion_df: DataFrame with all motion features
        keypoint_id_2d: 2D keypoint ID
        keypoint_id_3d: 3D keypoint ID (optional, defaults to same as 2D)
    Returns:
        tuple: (features_2d_df, features_3d_df)
    """
    if keypoint_id_3d is None:
        keypoint_id_3d = keypoint_id_2d
    
    base_cols = ['time', 'frame', 'detection_id']
    
    # 2D features
    cols_2d = base_cols.copy()
    for col in motion_df.columns:
        # 2D coordinates, velocity, acceleration
        if col in [f'{keypoint_id_2d}_x', f'{keypoint_id_2d}_y'] or \
           col in [f'vel_2d_{keypoint_id_2d}_x', f'vel_2d_{keypoint_id_2d}_y'] or \
           col in [f'acc_2d_{keypoint_id_2d}_x', f'acc_2d_{keypoint_id_2d}_y'] or \
           col.startswith('ang_2d_'):
            cols_2d.append(col)
    
    # 3D features
    cols_3d = base_cols.copy()
    for col in motion_df.columns:
        # 3D coordinates, velocity, acceleration
        if col in [f'3d_{keypoint_id_3d}_x', f'3d_{keypoint_id_3d}_y', f'3d_{keypoint_id_3d}_z'] or \
           col in [f'vel_3d_{keypoint_id_3d}_x', f'vel_3d_{keypoint_id_3d}_y', f'vel_3d_{keypoint_id_3d}_z'] or \
           col in [f'acc_3d_{keypoint_id_3d}_x', f'acc_3d_{keypoint_id_3d}_y', f'acc_3d_{keypoint_id_3d}_z'] or \
           col.startswith('ang_3d_'):
            cols_3d.append(col)
    
    existing_cols_2d = [col for col in cols_2d if col in motion_df.columns]
    existing_cols_3d = [col for col in cols_3d if col in motion_df.columns]
    
    features_2d = motion_df[existing_cols_2d] if existing_cols_2d else pd.DataFrame()
    features_3d = motion_df[existing_cols_3d] if existing_cols_3d else pd.DataFrame()
    
    return features_2d, features_3d


def select_feature_data(keypoint_id_2d, keypoint_id_3d, feature_type='coordinates'):
    """
    Select feature data based on feature type
    Returns:
        dict with selected feature column names for 2D and 3D
    """
    
    if feature_type == 'coordinates':
        return {
            '2d_features': [f'{keypoint_id_2d}_x', f'{keypoint_id_2d}_y'],
            '3d_features': [f'3d_{keypoint_id_3d}_x', f'3d_{keypoint_id_3d}_y', f'3d_{keypoint_id_3d}_z']
        }
    elif feature_type == 'velocity':
        return {
            '2d_features': [f'vel_2d_{keypoint_id_2d}_x', f'vel_2d_{keypoint_id_2d}_y'],
            '3d_features': [f'vel_3d_{keypoint_id_3d}_x', f'vel_3d_{keypoint_id_3d}_y', f'vel_3d_{keypoint_id_3d}_z']
        }
    elif feature_type == 'acceleration':
        return {
            '2d_features': [f'acc_2d_{keypoint_id_2d}_x', f'acc_2d_{keypoint_id_2d}_y'],
            '3d_features': [f'acc_3d_{keypoint_id_3d}_x', f'acc_3d_{keypoint_id_3d}_y', f'acc_3d_{keypoint_id_3d}_z']
        }
    elif feature_type == 'angles':
        return {
            '2d_features': ['ang_2d_leftElbow', 'ang_2d_rightElbow'],
            '3d_features': ['ang_3d_leftElbow', 'ang_3d_rightElbow']
        }