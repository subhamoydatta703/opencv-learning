import cv2
import numpy as np

# Create Sharingan image ONCE
sharingan = np.zeros((300, 300, 3), dtype=np.uint8)

cv2.circle(sharingan, (150, 150), 100, (0,0,255), -1)

detector = cv2.FaceDetectorYN.create("face_detection_yunet_2026may.onnx", "", (320, 320))

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    
    if not ret:
        break
    height, width = frame.shape[:2]
    
    detector.setInputSize((width, height))
        
    _, faces = detector.detect(frame)

    if faces is None:
            continue
    for face in faces:
        x,y,w,h = face[:4]
        left_eye_x= int(face[6])
        left_eye_y= int(face[7])

        sharingan_size = int(w * 0.20)
        resized_sharingan = cv2.resize( sharingan, (sharingan_size, sharingan_size))

        top_left_x = left_eye_x - sharingan_size //2
        top_left_y = left_eye_y - sharingan_size //2

        cv2.circle(frame, (left_eye_x, left_eye_y), 5, (0,0,255), -1)
    cv2.imshow("sharingan", frame)

    if cv2.waitKey(1)==27:
         break
cap.release()
cv2.destroyAllWindows()
