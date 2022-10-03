import cv2
import mediapipe as mp
import numpy as np
from compare import runAllImageSimilaryFun
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

# For webcam input:
cap = cv2.VideoCapture(0)
fingers = []
with mp_hands.Hands(
    model_complexity=0,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
    max_num_hands=1,
    ) as hands:
  while cap.isOpened():
    success, image = cap.read()
    if not success:
      print("Ignoring empty camera frame.")
      # If loading a video, use 'break' instead of 'continue'.
      continue

    # To improve performance, optionally mark the image as not writeable to
    # pass by reference.
    image.flags.writeable = False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image)

    # Draw the hand annotations on the image.
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    canvas = [[300,100],[1000,600]]
    earse = [[0,0],[100,100]]
    cv2.rectangle(image, (canvas[0][0], canvas[0][1]), (canvas[1][0], canvas[1][1]), (0, 255, 0), 2)
    cv2.rectangle(image, (earse[0][0], earse[0][1]), (earse[1][0], earse[1][1]), (0, 255, 255), 2)
    if results.multi_hand_landmarks:
      for hand_landmarks in results.multi_hand_landmarks:
        for i,lm in enumerate(hand_landmarks.landmark):
          xPos = int(image.shape[1] * lm.x)
          yPos = int(image.shape[0] * lm.y)
          if i == 8:
            # 手指中心圆圈
            cv2.circle(image, (xPos, yPos), 15, (255, 0, 255), cv2.FILLED)
            if xPos > canvas[0][0] and xPos < canvas[1][0] and yPos > canvas[0][1] and yPos < canvas[1][1]:
              # 在框内则添加轨迹点
              fingers.append([xPos,yPos])
            if xPos > earse[0][0] and xPos < earse[1][0] and yPos > earse[0][1] and yPos < earse[1][1]:
              # 移动至清除框内则清除轨迹点
              fingers = []
        # mp_drawing.draw_landmarks(
        #     image,
        #     hand_landmarks,
        #     mp_hands.HAND_CONNECTIONS,
        #     mp_drawing_styles.get_default_hand_landmarks_style(),
        #     mp_drawing_styles.get_default_hand_connections_style())
    # 按c清除
    if cv2.waitKey(1) & 0xFF == ord('c'):
      fingers = []
    # 绘制轨迹
    if len(fingers) >= 2:
      pts = np.array([fingers], np.int32)
      cv2.polylines(image, [pts], False, (0, 255, 255), 10)
      img = np.zeros((750,1280), np.uint8)
      img.fill(255)
      cv2.polylines(img, [pts], False, (0, 255, 255), 10)
      img = cv2.flip(img,1)
      cv2.imwrite('mydraw.png', img)
    image = cv2.flip(image,1)
    cv2.imshow('MediaPipe Hands', image)
    # 按w比较
    if cv2.waitKey(1) & 0xFF == ord('w'):
      mydraw="./mydraw.png"
      p1="./fire.png"
      p2="./v.png"
      p3="./circleq.png"

      runAllImageSimilaryFun(mydraw,p1)
      runAllImageSimilaryFun(mydraw,p2)
      runAllImageSimilaryFun(mydraw,p3)
    # 按q退出
    if cv2.waitKey(1) & 0xFF == ord('q'):
      break
cap.release()
