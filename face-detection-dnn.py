import cv2

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
        x, y, w, h = face[:4]

        
            

        cv2.rectangle(
            frame,
            (int(x), int(y)),
            (int(x + w), int(y + h)),
            (0, 255, 0),
            2
        )
    cv2.imshow("Face Detection", frame)

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()