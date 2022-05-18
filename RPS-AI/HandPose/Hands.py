import mediapipe as mp
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
import time
from id_distance import calc_all_distance
from cv2 import cv2
import hand_detection_module
from data_generation import num_hand
import pickle

model_name = 'hand_model.sav'

def rps(num):
  if num == 0: return 'PAPER'
  elif num == 1: return 'ROCK'
  else: return 'SCISSOR'

font = cv2.FONT_HERSHEY_PLAIN
hands = hand_detection_module.HandDetector(max_hands = num_hand)
model = pickle.load(open(model_name,'rb'))
model2 = pickle.load(open(model_name,'rb'))
cap = cv2.VideoCapture(0)
#prevTime = 0
with mp_hands.Hands(
    static_image_mode = False,
    min_detection_confidence=0.5,       
    min_tracking_confidence=0.5) as asdHands:
  while cap.isOpened():
    success, frame = cap.read()
    if not success:
      print("Ignoring empty camera frame.")
      continue
  
    image, my_list = hands.find_hand_landmarks(cv2.flip(frame, 1),
                                               draw_landmarks=False)  
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = asdHands.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    if my_list:
      height, width, _ = image.shape
      all_distance = calc_all_distance(height,width, my_list)
      pred = rps(model.predict([all_distance])[0])
      pos = (int(my_list[12][0]*height), int(my_list[12][1]*width))
      pos2 = (int(my_list[12][0]*height), int(my_list[12][1]*width))
      image = cv2.putText(image,pred,pos,font,2,(0,0,0),2)
    
    if results.multi_hand_landmarks:
      #for hand_landmarks in results.multi_hand_landmarks:
      for hand_no, hand_landmarks in enumerate(results.multi_hand_landmarks):
        mp_drawing.draw_landmarks(
            image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
          
    #currTime = time.time()
    #fps = 1 / (currTime - prevTime)
    #prevTime = currTime
    #cv2.putText(image, f'FPS: {int(fps)}', (20, 70), cv2.FONT_HERSHEY_PLAIN, 3, (0, 196, 255), 2)
    cv2.imshow('MediaPipe Hands', image)
    
    if cv2.waitKey(1) == ord('x'):
      break
cap.release()
