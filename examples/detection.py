from physiotrack import Detection, Models
import cv2 

image = cv2.imread('frame_1.png')
# Default Person Detector
detector = Detection.Person(model=Models.Detection.YOLO.PERSON.m_person, render_box_detections=True, render_labels=True)
results, frame = detector.detect(image)
cv2.imwrite('out.png', frame)
