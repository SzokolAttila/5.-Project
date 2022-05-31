from ast import arg
from asyncore import read
from concurrent.futures import thread
from operator import truediv
from pickletools import read_bytes1
from tracemalloc import start
from matplotlib.pyplot import margins
import mediapipe as mp
from numpy import size
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
from id_distance import calc_all_distance
from cv2 import cv2
import hand_detection_module
from data_generation import num_hand
import pickle
import threading
from tkinter import *

model_name = 'hand_model.sav'

def showWinner():
  winner = Toplevel()
  winner.title("Winner")
  winnerText = Label(winner, text=defineWinner(p1points, p2points), font=("Times New Roman", 50)).pack()

def points():
  counter = Toplevel()
  p1Label = Label(counter, text=f"P1:{p1points}", font=("Times New Roman", 50)).grid(row=0, column=0)
  p2Label = Label(counter, text=f"P2: {p2points}", font=("Times New Roman", 50)).grid(row=0, column=2)
  btn = Button(counter, text="Show the winner", command=showWinner, justify="center", font=("Times New Roman", 20)).grid(row=1, column=1)
  counter.title("The Ultimate Rock-Paper-Scissors")

def main():
  global p1points, p2points
  p1points = 0
  p2points = 0
  font = cv2.FONT_HERSHEY_PLAIN
  hands = hand_detection_module.HandDetector(max_hands = num_hand)
  model = pickle.load(open(model_name,'rb'))
  model2 = pickle.load(open(model_name,'rb'))
  cap = cv2.VideoCapture(0)
  p1pred = ""
  p2pred = ""

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
        if my_list.index(hand) == 0:
          player = "P1:"
          p1pred = pred
        else:
          player = "P2:"
          p2pred = pred
        image = cv2.putText(image,player + pred,pos,font,2,(0,0,0),2)

      if cv2.waitKey(1) == ord('c'):
        global win
        win = ''
        if p1pred == "SCISSORS":
          if p2pred == "PAPER":
            p1points += 1
            win = "p1"
          elif p2pred == "ROCK":
            p2points += 1
            win = "p2"
        elif p1pred == "ROCK":
          if p2pred == "PAPER":
            p2points += 1
            win = "p2"
          elif p2pred == "SCISSORS":
            p1points += 1
            win = "p1"
        else:
          if p2pred == "ROCK":
            p1points += 1
            win = "p1"
          elif p2pred == "SCISSORS":
            p2points += 1
            win = "p2"
        print(f"done, point goes to {win}")
      if cv2.waitKey(1) == ord('x'):
          break
      cv2.imshow('RPS', image)
  cap.release()

def rps(num):
  if num == 0: return 'PAPER'
  elif num == 1: return 'ROCK'
  else: return 'SCISSORS'

def defineWinner(p1, p2):
  if p1 > p2:
    winner = "The winner is: P1"
  elif p1 < p2:
    winner = "The winner is: P2"
  else:
    winner = "It's a tie"
  return  winner

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

program = threading.Thread(target=main)
program.start()
root = Tk()
root.geometry("200x100")
btn = Button(root, text="Show points", command=points, justify="center", font=("Times New Roman", 20), pady=10).place(relx=0.5, rely=0.5, anchor=CENTER)
root.title("The Ultimate Rock-Paper-Scissors")
root.mainloop()


