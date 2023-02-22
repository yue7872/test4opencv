import cv2
import mediapipe as mp
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose
import numpy as np

# For webcam input:
cap = cv2.VideoCapture('./1.mp4')
c = 1
resultNum = 0
with mp_pose.Pose(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as pose:
  while cap.isOpened():
    success, image = cap.read()
    if not success:
      print("Ignoring empty camera frame.")
      # If loading a video, use 'break' instead of 'continue'.
      break
    # To improve performance, optionally mark the image as not writeable to
    # pass by reference.
    image.flags.writeable = False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = pose.process(image)

    # Draw the pose annotation on the image.
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    mp_drawing.draw_landmarks(
        image,
        results.pose_landmarks,
        mp_pose.POSE_CONNECTIONS,
        landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
    # Flip the image horizontally for a selfie-view display.
    cv2.imshow('MediaPipe Pose', cv2.flip(image, 1))
    timeF = 10
    if(c%timeF == 0):
      # # 点编号
      # for i,lm in enumerate(results.pose_landmarks.landmark):
      #   xPos = int(image.shape[1] * lm.x)
      #   yPos = int(image.shape[0] * lm.y)
      #   cv2.putText(image, str(i), (xPos+5, yPos+5), 0, 0.4, (0, 0, 0))
      cv2.imwrite('/Users/blairyue/Desktop/yyz/test4opencv/pose-recognition/video-result/'+ str(resultNum) +'.png', image)
      resultNum = resultNum + 1
      print('save image: ', resultNum)
    c = c + 1
    if cv2.waitKey(5) & 0xFF == 27:
      break
cap.release()
