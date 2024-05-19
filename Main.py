import mediapipe as mp
import cv2 as cv
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import numpy as np
import pygame
from StolenGoogleDrawfunc import draw_landmarks_on_image



if __name__ == '__main__':   
  # pygame setup
  pygame.init()
  screen = pygame.display.set_mode((1280, 720))
  clock = pygame.time.Clock()
  #model initialization
  #put in specifications regarding the tuning of the model
  model_path = "D:\\Toilet\\HandTracking\\hand_landmarker.task"

  BaseOptions = mp.tasks.BaseOptions
  HandLandmarker = mp.tasks.vision.HandLandmarker
  HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
  VisionRunningMode = mp.tasks.vision.RunningMode

  options = HandLandmarkerOptions(
      base_options=BaseOptions(model_asset_path=model_path),
      running_mode=VisionRunningMode.IMAGE,
      num_hands = 2    
  )

  detector = vision.HandLandmarker.create_from_options(options)


  cap = cv.VideoCapture(0)

  while True:
      screen.fill("white")
      index_finger_coords = [0,0]
      ret,frame = cap.read()
      mp_frame = cv.cvtColor(frame,cv.COLOR_BGR2RGB)
      final_frame = mp.Image(image_format = mp.ImageFormat.SRGB, data = mp_frame)
      #cv.imshow('frame',frame)
      hand_landmark_result = detector.detect(final_frame)
      annotated_image = draw_landmarks_on_image(mp_frame, hand_landmark_result)
      cv.imshow("new frame",cv.cvtColor(annotated_image, cv.COLOR_RGB2BGR))
      landmark_list = hand_landmark_result.hand_landmarks
      if(len(landmark_list)> 0):
        for idx in range(len(landmark_list)):
          landmarks = landmark_list[idx]
        index_finger_coords[0] = (landmarks[8].x * 500) + 640
        index_finger_coords[1] = (landmarks[8].y * 500) + 360
        print(index_finger_coords)
        pygame.draw.circle(screen,"red",(index_finger_coords[0], index_finger_coords[1]),20,0)  
      
      #waitkey bs
      k = cv.waitKey(5) & 0xFF
      if k == 27:
          break
      for event in pygame.event.get():
        if event.type == pygame.QUIT:
            break
      
      pygame.display.flip()
      clock.tick(60)


  cv.destroyAllWindows()
  pygame.quit()
