import mediapipe as mp
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
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

class mpHands:
    import mediapipe as mp
    def __init__(self,maxHands=2,tol1=.5,tol2=.5):
        self.hands=self.mp.solutions.hands.Hands(False,maxHands,tol1,tol2)
    def Marks(self,frame):
        myHands=[]
        handsType=[]
        frameRGB=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        results=self.hands.process(frameRGB)
        if results.multi_hand_landmarks != None:
            for hand in results.multi_handedness:
                handType=hand.classification[0].label
                handsType.append(handType)
            for handLandMarks in results.multi_hand_landmarks:
                myHand=[]
                for landMark in handLandMarks.landmark:
                    myHand.append((int(landMark.x*width),int(landMark.y*height)))
                myHands.append(myHand)
        return myHands,handsType

font = cv2.FONT_HERSHEY_PLAIN
hands = hand_detection_module.HandDetector(max_hands = num_hand)
model = pickle.load(open(model_name,'rb'))
model2 = pickle.load(open(model_name,'rb'))
cap = cv2.VideoCapture(0)
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

    for hand in my_list:
      height, width, _ = image.shape
      all_distance = calc_all_distance(height,width, hand)
      pred = rps(model.predict([all_distance])[0])
      pos = (int(hand[12][0]*height), int(hand[12][1]*width))
      pos2 = (int(hand[12][0]*height), int(hand[12][1]*width))
      image = cv2.putText(image,pred,pos,font,2,(0,0,0),2)
    if cv2.waitKey(1) == ord('x'):
      break
    cv2.imshow('MediaPipe Hands', image)
cap.release()
