import numpy as np
import cv2
import os
import PIL
from PIL import ImageTk
import PIL.Image
import speech_recognition as sr
import pyttsx3
from itertools import count
from tkinter import *
try:
    import tkinter as tk
except:
    import tkinter as tk
# from newpro import new
from matplotlib import pyplot as plt
import time
import mediapipe as mp
from PIL import Image, ImageDraw, ImageFont
from PIL import ImageFont, ImageDraw, Image
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import TensorBoard,EarlyStopping
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
import tensorflow as tf
from keras.models import load_model
mp_holistic = mp.solutions.holistic # Holistic model
mp_drawing = mp.solutions.drawing_utils # Drawing utilities
image_x, image_y = 64,64
def check_sim(i,file_map):
       for item in file_map:
              for word in file_map[item]:
                     if(i==word):
                            return 1,item
                     
       return -1,""

op_dest=r"C:/Users/being/OneDrive/Desktop/two-way-sign-language-translator/filtered_data/"
dirListing = os.listdir(op_dest)
editFiles = []
for item in dirListing:
       if ".webp" in item:
              editFiles.append(item)

file_map={} 
for i in editFiles:
       tmp=i.replace(".webp","")
       tmp=tmp.split()
       file_map[i]=tmp

def func(a):
       all_frames=[]
       final= PIL.Image.new('RGB', (380, 260))
       words=a.split()
       for i in words:
              flag,sim=check_sim(i,file_map)
              if(flag!=-1):
                     im = PIL.Image.open(op_dest+sim)
                     im.info.pop('background', None)
                     frameCnt = im.n_frames
                     for frame_cnt in range(frameCnt):
                            im.seek(frame_cnt)
                            im.save("tmp.png")
                            img = cv2.imread("tmp.png")
                            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                            img = cv2.resize(img, (380,260))
                            im_arr = PIL.Image.fromarray(img)
                            all_frames.append(im_arr)
       final.save("out.gif", save_all=True, append_images=all_frames, duration=100, loop=0)
       return all_frames      

img_counter = 0
img_text=''
class Tk_Manage(tk.Tk):
       def __init__(self, *args, **kwargs):     
              tk.Tk.__init__(self, *args, **kwargs)
              container = tk.Frame(self)
              container.pack(side="top", fill="both", expand = True)
              container.grid_rowconfigure(0, weight=1)
              container.grid_columnconfigure(0, weight=1)
              self.frames = {}
              for F in (StartPage, TtoS, StoT):
                     frame = F(container, self)
                     self.frames[F] = frame
                     frame.grid(row=0, column=0, sticky="nsew")
              self.show_frame(StartPage)
                     
       def show_frame(self, cont):
              frame = self.frames[cont]
              frame.tkraise()

        
class StartPage(tk.Frame):

       def __init__(self, parent, controller):
              tk.Frame.__init__(self,parent)
              label = tk.Label(self, text="Two Way Sign Langage Translator", font=("Verdana", 15))
              label.pack(pady=20,padx=20)
              button = tk.Button(self, text="Text to Sign",command=lambda: controller.show_frame(TtoS))
              button.pack()
              button2 = tk.Button(self, text="Sign to Text",command=lambda: controller.show_frame(StoT))
              button2.pack(padx=10, pady=10)
              load = PIL.Image.open("TextToSign.jpg")
              load = load.resize((620, 450))
              render = ImageTk.PhotoImage(load)
              img = Label(self, image=render)
              img.image = render
              img.place(x=100, y=200) 
              


class TtoS(tk.Frame):
       def __init__(self, parent, controller):
              cnt=0
              gif_frames=[]
              inputtxt=None
              tk.Frame.__init__(self, parent)
              label = tk.Label(self, text="Text to Sign", font=("Verdana", 15))
              label.pack(pady=10,padx=10)
              gif_box = tk.Label(self)
              
              button1 = tk.Button(self, text="Back to Home",command=lambda: controller.show_frame(StartPage))
              button1.pack(pady=10,padx=10)
              def gif_stream():
                     global cnt
                     global gif_frames
                     if(cnt==len(gif_frames)):
                            return
                     img = gif_frames[cnt]
                     cnt+=1
                     imgtk = ImageTk.PhotoImage(image=img)
                     gif_box.imgtk = imgtk
                     gif_box.configure(image=imgtk)
                     gif_box.after(50, gif_stream)
              def Take_input():
                     INPUT = inputtxt.get("1.0", "end-1c")
                     global gif_frames
                     gif_frames=func(INPUT)
                     global cnt
                     cnt=0
                     gif_stream()
                     gif_box.place(x=400,y=160)
              
              l = tk.Label(self,text = "Enter Text:")
              inputtxt = tk.Text(self, height = 4,width = 25)
              Display = tk.Button(self, height = 2,width = 20,text ="Convert",command = lambda:Take_input())
              l.place(x=50, y=160)
              # l1.place(x=115, y=230)
              inputtxt.place(x=50, y=250)
              Display.pack()

class StoT(tk.Frame):
       
       def __init__(self, parent, controller):
              tk.Frame.__init__(self, parent)
              label = tk.Label(self, text="Sign to Text", font=("Verdana", 12))
              label.pack(pady=10,padx=10)
              button1 = tk.Button(self, text="Back to Home",command=lambda: controller.show_frame(StartPage))
              button1.pack(pady=10,padx=10)
              def start_video():
                     video_frame = tk.Label(self)
                     cam = cv2.VideoCapture(0)
                     
                     global img_counter
                     img_counter = 0
                     global img_text
                     img_text = ''
                     def video_stream():
                            mp_holistic = mp.solutions.holistic # Holistic model
                            mp_drawing = mp.solutions.drawing_utils # Drawing utilities
                            def mediapipe_detection(image, model):
                                   image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # COLOR CONVERSION BGR 2 RGB as model can only detect in RGB
                                   image.flags.writeable = False                  # Image is no longer writeable
                                   results = model.process(image)                 # Use Model to make prediction
                                   image.flags.writeable = True                   # Image is now writeable 
                                   image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) # COLOR COVERSION RGB 2 BGR
                                   return image, results
                            def draw_landmarks(image, results): # draw landmarks for each image/frame
                                   #mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_CONTOURS) # Draw face connections
                                   mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS) # Draw pose connections
                                   mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS) # Draw left hand connections
                                   mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS) # Draw right hand connections
                            def draw_styled_landmarks(image, results): # draw landmarks for each image/frame, fix colour of landmark drawn
                            # Draw face connections
                                   mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_CONTOURS, 
                                                        mp_drawing.DrawingSpec(color=(80,110,10), thickness=1, circle_radius=1), 
                                                        mp_drawing.DrawingSpec(color=(80,256,121), thickness=1, circle_radius=1)
                                                        ) 
                            # Draw pose connections
                                   mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                                                        mp_drawing.DrawingSpec(color=(80,22,10), thickness=2, circle_radius=4), 
                                                        mp_drawing.DrawingSpec(color=(80,44,121), thickness=2, circle_radius=2)
                                                        ) 
                            # Draw left hand connections
                                   mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                                                        mp_drawing.DrawingSpec(color=(121,22,76), thickness=2, circle_radius=4), 
                                                        mp_drawing.DrawingSpec(color=(121,44,250), thickness=2, circle_radius=2)
                                                        ) 
                            # Draw right hand connections  
                                   mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                                                        mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=4), 
                                                        mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
                                                        ) 
                            def extract_keypoints(results):
                                   pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
                                   lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
                                   rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
                                   #     return np.concatenate([pose, face, lh, rh]) # concatenate all the keypoints that are flattened
                                   return np.concatenate([pose, lh, rh])

                            actions = np.array(['அணில்','அறுவடை செய்','அறுவை சிகிச்சை','ஆந்தை','ஆப்பிள்','ஆரஞ்சுப்பழம்','ஆற்றல் நிறைந்த','இஞ்சி','இதயம்','இரத்த அழுத்தம்','உடல்','ஊழல் செய்தல்','எலி','ஒட்டகச்சிவிங்கி','ஒட்டகம்','கங்காரு','கண்','கரடி','கலங்கரை விளக்கம்','கழுகு','கழுதை','கழுத்து','காகம்','காண்டாமிருகம்','கிளி','குடியரசுத் தலைவர்','குதிரை','குரங்கு','சாத்துக்குடி','சிட்டுக்குருவி','செம்மறியாடு','சேவல்','தர்பூசணி','தலை','தேங்காய்','தேசீய கீதம்','நரி','பசு','பறவை','பீன்ஸ்','புதிதாக கண்டுபிடி','புரிந்து கொள்ளுதல்','பூசணிக்காய்','பூனை','போக்குவரத்து விளக்கு','மயில்','மான்','மீன்','முகம்','முதலை','முருங்கைக்காய்','மூக்கு','யானை','வரிக்குதிரை','வாத்து','வாய்','வாழைப்பழம்','வான்கோழி','வெள்ளரிக்காய்','வெள்ளாடு','உடல் நலம்','உதவி செய்தல்','உயிர் வாழ்தல்','காலம் தவறாத','காவல் நிலையம்','சேமித்து வை','தனியார் அஞ்சல் சேவை','துறைமுகம்','பிரார்த்தனை செய்','பேருந்து நிலையம்','விமான நிலையம்'])
                            # actions = np.array(['அணில்','அறுவடை செய்','அறுவை சிகிச்சை','ஆந்தை'])
                            no_sequences = 40
                            sequence_length = 30 
                            label_map = {label:num for num, label in enumerate(actions)} #create label map dictionary

                            model = Sequential()
                            model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(30,258))) #each video has input shape of 30 frames of 1662 keypoints: X.shape
                            model.add(LSTM(128, return_sequences=False, activation='relu'))
                            #Dense layer with relu
                            model.add(Dense(64, activation='relu'))
                            model.add(Dense(32, activation='relu'))
                            model.add(Dense(actions.shape[0], activation='softmax'))
                            # Build LSTM Model Architecture Layers using Keras high-level # experiment 4
                            model = Sequential()
                            model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(30,258))) #each video has input shape of 30 frames of 1662 keypoints: X.shape
                            model.add(LSTM(128, return_sequences=True, activation='relu'))
                            model.add(LSTM(64, return_sequences=False, activation='relu')) #next layer is a dense layer so we do not return sequences here
                            model.add(Dense(64, activation='relu'))
                            model.add(Dense(32, activation='relu'))
                            model.add(Dense(actions.shape[0], activation='softmax'))
                            # Build LSTM Model Architecture Layers using Keras high-level API # experiment 5
                            model = Sequential()
                            model.add(LSTM(128, activation='relu', input_shape=(30,258))) #each video has input shape of 30 frames of 1662 keypoints: X.shape
                            model.add(Dropout(0.2))
                            #Dense layer with relu
                            model.add(Dense(64, activation='relu'))
                            model.add(Dense(actions.shape[0], activation='softmax'))
                            model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])
                            model.load_weights('action.h5',by_name=True)
                            font_file =r"C:/Users/being/Downloads/latha.ttf"
                            font_size = 30
                            font = ImageFont.truetype(font_file, size=font_size)
                            # 1. New detection variables
                            sequence = []
                            sentence = []
                            predictions = []
                            threshold = 0.5
                            # use computer webcam and make keypoint detections
                            cap = cv2.VideoCapture(0)
                            # Set mediapipe model 
                            with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
                                   while cap.isOpened():
                                          ret, image = cap.read()
                                          if ret == True:    
                                                 image, results = mediapipe_detection(image, holistic)
                                                 draw_styled_landmarks(image, results)
                                                 keypoints = extract_keypoints(results)
                                                 sequence.append(keypoints)
                                                 sequence = sequence[-30:]
                                          if len(sequence) == 30:
                                                 res = model.predict(np.expand_dims(sequence, axis=0))[0]
                                                 if res[np.argmax(res)] > threshold: 
                                                        predicted_word = actions[np.argmax(res)]
                                                 if len(sentence) == 0 or predicted_word != sentence[-1]:
                                                        sentence.append(predicted_word)
                                                        sentence = [predicted_word]

                                                 img_pil = Image.fromarray(image)
                                                 draw = ImageDraw.Draw(img_pil)
                                                 # Convert Tamil sentence to Unicode and display on image
                                                 draw.text((3, 30), ' '.join(sentence), font=font, fill=(0, 0, 255, 255), outline=None)
                                                 image = np.array(img_pil)
                                          cv2.imshow('OpenCV Feed', image)
                                          if cv2.waitKey(1) & 0xFF == ord('q'):
                                                 break
                                   # Release video capture and close OpenCV window
                                   cap.release()
                                   cv2.destroyAllWindows()
                     video_stream()
                     video_frame.pack()
              start_vid = tk.Button(self,height = 2,width = 20, text="Start Video",command=lambda: start_video())
              start_vid.pack()
app = Tk_Manage()
app.geometry("800x750")
app.mainloop()