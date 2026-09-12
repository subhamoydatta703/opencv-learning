import cv2

# Load the pre-trained Haar Cascade face detector
face_cascade = cv2.CascadeClassifier(
    "haarcascade_frontalface_default.xml"
)

# Open the webcam
cap = cv2.VideoCapture(0)

while True:

    # Capture one frame from the webcam
    ret, frame = cap.read()

    # Stop if the frame was not captured successfully
    if not ret:
        break

    # Convert the frame from BGR to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the grayscale frame
    faces = face_cascade.detectMultiScale( gray, scaleFactor=1.1, minNeighbors=5)

    # Draw a rectangle around every detected face
    for (x, y, w, h) in faces:

        cv2.rectangle( frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Display the frame
    cv2.imshow("Face Detection", frame)

    # Press ESC to stop
    if cv2.waitKey(1) == 27:
        break

# Release the webcam
cap.release()

# Close OpenCV windows
cv2.destroyAllWindows()