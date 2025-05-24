from physiotrack import Pose, Models
import cv2 

image = cv2.imread('frame_1.png')
# Default Person Detector
detector = Pose.Person(model=Models.Pose.YOLO.COCO.M11, render_box_detections=True, render_labels=True)
results, frame = detector.detect(image)
cv2.imwrite('out.png', frame)
