# import cv2

# print("cv2 location:", cv2.__file__)
# print("cv2 version:", cv2.__version__)
# print("Has CascadeClassifier:", hasattr(cv2, "CascadeClassifier"))

import cv2

print(cv2.__version__)
print("-----------------")
print(hasattr(cv2, "FaceDetectorYN"))
print("-----------------")


print(cv2.__version__)
print(cv2.FaceDetectorYN.setInputSize.__doc__)