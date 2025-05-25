from physiotrack import Pose, Models
import cv2 

image = cv2.imread('frame_1.png')
detector = Pose.VRStudent(render_box_detections=False, render_labels=True, overlay_keypoints=True)
frame, results = detector.estimate(image)
cv2.imwrite('out.png', frame)
