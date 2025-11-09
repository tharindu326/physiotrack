"""
Example script demonstrating face orientation estimation with visualization.
"""

import cv2
from physiotrack import Face, FaceOrientation, Models
from physiotrack.face import draw_axis, plot_pose_cube

# Load image
image_path = 'kinect_s1_v1_frame1.png'  # Change this to your image path
img = cv2.imread(image_path)

# Initialize face detector and face orientation estimator
face_detector = Face(device=0)
face_orientation = FaceOrientation(device=0, render_pose=False)

# Detect faces
det_results, _ = face_detector.detect(img)
bboxes = det_results[0].boxes.data.cpu().numpy()[:, :4]

# Estimate face orientation
output_img, pose_results = face_orientation.predict(img, bboxes)

# Visualize pose
vis_img = img.copy()
for detection in pose_results['detections']:
    pose = detection['pose']
    bbox = detection['bbox']
    
    x1, y1, x2, y2 = bbox
    face_center_x = int((x1 + x2) / 2)
    face_center_y = int((y1 + y2) / 2)
    
    # Draw bounding box
    # cv2.rectangle(vis_img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
    
    face_width = x2 - x1
    face_height = y2 - y1
    
    # Draw direction axes
    axis_size = max(face_width, face_height) * 0.6  # Scale axes to 60% of larger face dimension
    vis_img = draw_axis(
        vis_img,
        yaw=pose['yaw'],
        pitch=pose['pitch'],
        roll=pose['roll'],
        tdx=face_center_x,
        tdy=face_center_y,
        size=axis_size
    )
    
    # Draw pose cube (alternative visualization)
    # cube_size = max(face_width, face_height) * 0.6  # Scale cube to 60% of larger face dimension
    # vis_img = plot_pose_cube(
    #     vis_img,
    #     yaw=pose['yaw'],
    #     pitch=pose['pitch'],
    #     roll=pose['roll'],
    #     tdx=face_center_x,
    #     tdy=face_center_y,
    #     size=cube_size
    # )

# Save output
cv2.imwrite('face_orientation_output.png', vis_img)

