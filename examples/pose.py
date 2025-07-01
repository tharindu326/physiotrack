from physiotrack import Pose, Models
import cv2 

image = cv2.imread('frame_1.png')
detector = Pose.Person(render_box_detections=False, render_labels=True, overlay_keypoints=True)
frame, results = detector.estimate(image)

print(f"Number of poses detected: {len(results)}")
print(f"Pose architecture: {results.pose_archetecture}")

print("\n=== Pose Objects ===")
for i, pose in enumerate(results):
    print(f"\nPose {i}:")
    print(f"  ID: {pose.id}")
    print(f"  Bounding Box: {pose.box}")
    
    # Print some key keypoints
    print("  Key Keypoints:")
    key_points = ["nose", "left_shoulder", "right_shoulder", "left_wrist", "right_wrist"]
    for kp_name in key_points:
        kp = pose.keypoints.name(kp_name)
        if kp:
            print(f"    {kp_name}: x={kp.x:.1f}, y={kp.y:.1f}, confidence={kp.confidence:.3f}")
    
    print("  All visible keypoints (confidence > 0.5):")
    for kp_id in range(133):
        kp = pose.keypoints.id(kp_id)
        if kp and kp.confidence > 0.5:
            print(f"    {kp.name} (ID:{kp.id}): x={kp.x:.1f}, y={kp.y:.1f}, conf={kp.confidence:.3f}")

cv2.imwrite('out.png', frame)