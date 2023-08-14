import cv2
import numpy as np
import os
from matplotlib import pyplot as plt
import time
import mediapipe as mp
from PIL import Image, ImageDraw, ImageFont
import os
from PIL import ImageFont, ImageDraw, Image
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import TensorBoard,EarlyStopping
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.utils import plot_model
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

import tensorflow as tf
import pandas as pd
# # Use Holistic Models for detections
mp_holistic = mp.solutions.holistic # Holistic model
mp_drawing = mp.solutions.drawing_utils # Drawing utilities


# Make keypoint detection, model can only detect in RGB
def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # COLOR CONVERSION BGR 2 RGB as model can only detect in RGB
    image.flags.writeable = False                  # Image is no longer writeable
    results = model.process(image)                 # Use Model to make prediction
    image.flags.writeable = True                   # Image is now writeable 
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) # COLOR COVERSION RGB 2 BGR
    return image, results

def draw_landmarks(image, results): # draw landmarks for each image/frame
    mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_CONTOURS) # Draw face connections
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

# use computer webcam and make keypoint detections
cap = cv2.VideoCapture(0)

# Set mediapipe model configurations
min_detection_confidence = 0.5
min_tracking_confidence= 0.5

with mp_holistic.Holistic(min_detection_confidence=min_detection_confidence, min_tracking_confidence=min_tracking_confidence) as holistic:
    while cap.isOpened():
        # Read feed
        ret, frame = cap.read()

        # Make detections by calling our function
        image, results = mediapipe_detection(frame, holistic) #mediapipe_detection(image, model) 
        #print(results)
        #print(results.face_landmarks)
        
        # Draw landmarks
        draw_styled_landmarks(image, results)
        
        # Show to screen
        cv2.imshow('OpenCV Feed: Hold Q to Quit', image)

        # Break gracefully
        if cv2.waitKey(10) & 0xFF == ord('q'): #press q to quit
            break
    cap.release() #release webcam
    cv2.destroyAllWindows()

#show last frame with keypoints drawn using draw styled landmarks
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

# call helper function to draw landmarks
draw_landmarks(frame, results)
# plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
           
#Show length of landmarks x,y,z spatial coordinates for right hand pose
len(results.left_hand_landmarks.landmark)           
pose = []
for res in results.pose_landmarks.landmark:
    test = np.array([res.x, res.y, res.z, res.visibility])
    pose.append(test)


# define extract keypoint function and convert to numpy array to be saved
def extract_keypoints(results):
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
    face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468*3)
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    # return np.concatenate([pose, face, lh,rh]) # concatenate all the keypoints that are flattened
    return np.concatenate([pose, lh, rh])

result_test = extract_keypoints(results) 
extract_keypoints(results).shape  


# save results as numpy array
# np.save('results', result_test)

#load numpy array
np.load('results.npy')

DATA_PATH = os.path.join('test')
actions = np.array(['அணில்','அறுவடை செய்','அறுவை சிகிச்சை','ஆந்தை','ஆப்பிள்','ஆரஞ்சுப்பழம்','இஞ்சி','இதயம்','இரத்த அழுத்தம்','உடல்','ஊழல் செய்தல்','எலி','ஒட்டகச்சிவிங்கி','ஒட்டகம்','கங்காரு','கண்','கரடி','கலங்கரை விளக்கம்','கழுகு','கழுதை','கழுத்து','காகம்','காண்டாமிருகம்','கிளி','குடியரசுத் தலைவர்','குதிரை','குரங்கு','சாத்துக்குடி','சிட்டுக்குருவி','செம்மறியாடு','சேவல்','தர்பூசணி','தலை','தேங்காய்','தேசீய கீதம்','நரி','பசு','பறவை','பீன்ஸ்','புதிதாக கண்டுபிடி','புரிந்து கொள்ளுதல்','பூசணிக்காய்','பூனை','போக்குவரத்து விளக்கு','மயில்','மான்','மீன்','முகம்','முதலை','முருங்கைக்காய்','மூக்கு','யானை','வரிக்குதிரை','வாத்து','வாய்','வாழைப்பழம்','வான்கோழி','வெள்ளரிக்காய்','வெள்ளாடு'])

# 60 videos worth of data for each action
no_sequences = 40

# Videos are going to be 30 frames in length (30 frames of data for each action)
sequence_length = 30



# # model.load_weights(r"C:\Users\being\OneDrive\Desktop\two-way-sign-language-translator\Epoch-60-Loss-0.23.h5")
# model.load_weights(r"C:\Users\being\OneDrive\Desktop\two-way-sign-language-translator\Epoch-60-Loss-0.23.h5", by_name=True)

#create label map dictionary
# label_map = {label:num for num, label in enumerate(actions)} 
# # print(label_map)

# #sequences represent x data, labels represent y data/the action classes.
# sequences, labels = [], []
# #Loop through the action classes you want to detect
# for action in actions:
#     #loop through each sequence
#     for sequence in np.array(os.listdir(os.path.join(DATA_PATH, action))).astype(int):
#         window = []
#         for frame_num in range(sequence_length):
#             res = np.load(os.path.join(DATA_PATH, action, str(sequence), "{}.npy".format(frame_num)))
#             window.append(res)
#         sequences.append(window)
#         labels.append(label_map[action])

# # X = Training Data that contains spatial coordinates x,y,z of landmarks
# X = np.array(sequences)

# # y = categorical labels
# y = to_categorical(labels).astype(int) #one-hot-encoding to catergorical variable

# # print('X Shape:',X.shape)
# # print('y Shape:',y.shape)

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20,stratify=y)
# test_count_label = tf.reduce_sum(y_test, axis=0)
# train_count_label = tf.reduce_sum(y_train, axis=0)

# Build LSTM Model Architecture Layers using Keras high-level # experiment 4
model = Sequential()
model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(30,258))) #each video has input shape of 30 frames of 1662 keypoints: X.shape
model.add(LSTM(128, return_sequences=True, activation='relu'))
model.add(LSTM(64, return_sequences=False, activation='relu')) #next layer is a dense layer so we do not return sequences here
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(actions.shape[0], activation='softmax'))

#Compile defines the loss function, the optimizer and the metrics. 
model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])

# model.load_weights(r"C:\Users\being\OneDrive\Desktop\two-way-sign-language-translator\Epoch-60-Loss-0.23.h5")
model.load_weights(r"C:\Users\being\OneDrive\Desktop\two-way-sign-language-translator\Epoch-60-Loss-0.23.h5", by_name=True)


# Specify Tamil font file path
font_file =r"C:\Users\being\Downloads\latha.ttf"
font_size = 22
font = ImageFont.truetype(font_file, size=font_size)

# font_data = np.asarray(font.getmask("Hello"), dtype=np.uint8)
# font_img = cv2.imdecode(font_data, cv2.IMREAD_GRAYSCALE)
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

    cap.release()
    cv2.destroyAllWindows()
# import cv2
# import numpy as np
# import os
# from matplotlib import pyplot as plt
# import time
# import mediapipe as mp
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import LSTM, Dense, Dropout
# from tensorflow.keras.callbacks import TensorBoard

# # Use Holistic Models for detections
# mp_holistic = mp.solutions.holistic # Holistic model
# mp_drawing = mp.solutions.drawing_utils # Drawing utilities

# # Make keypoint detection, model can only detect in RGB
# def mediapipe_detection(image, model):
#     image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # COLOR CONVERSION BGR 2 RGB as model can only detect in RGB
#     image.flags.writeable = False                  # Image is no longer writeable
#     results = model.process(image)                 # Use Model to make prediction
#     image.flags.writeable = True                   # Image is now writeable 
#     image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) # COLOR COVERSION RGB 2 BGR
#     return image, results

# def draw_landmarks(image, results): # draw landmarks for each image/frame
#     #mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_CONTOURS) # Draw face connections
#     mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS) # Draw pose connections
#     mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS) # Draw left hand connections
#     mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS) # Draw right hand connections
    
    
# def draw_styled_landmarks(image, results): # draw landmarks for each image/frame, fix colour of landmark drawn
#     # Draw face connections
#     mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_CONTOURS, 
#                              mp_drawing.DrawingSpec(color=(80,110,10), thickness=1, circle_radius=1), 
#                              mp_drawing.DrawingSpec(color=(80,256,121), thickness=1, circle_radius=1)
#                              ) 
#     # Draw pose connections
#     mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
#                              mp_drawing.DrawingSpec(color=(80,22,10), thickness=2, circle_radius=4), 
#                              mp_drawing.DrawingSpec(color=(80,44,121), thickness=2, circle_radius=2)
#                              ) 
#     # Draw left hand connections
#     mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
#                              mp_drawing.DrawingSpec(color=(121,22,76), thickness=2, circle_radius=4), 
#                              mp_drawing.DrawingSpec(color=(121,44,250), thickness=2, circle_radius=2)
#                              ) 
#     # Draw right hand connections  
#     mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
#                              mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=4), 
#                              mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
#                              ) 
    
# # define extract keypoint function
# def extract_keypoints(results):
#     pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
#     face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468*3)
#     lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
#     rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
# #     return np.concatenate([pose, face, lh, rh]) # concatenate all the keypoints that are flattened
#     return np.concatenate([pose, lh, rh])

# # extract_keypoints(results).shape

# #load numpy array
# np.load('results.npy')
# # extract_keypoints(results).shape
# # Path for exported data, numpy arrays
# #DATA_PATH = os.path.join('MP_Data') 

# # Actions that we try to detect
# actions = np.array(['அணில்','அறுவடை செய்','அறுவை சிகிச்சை','ஆந்தை','ஆப்பிள்','ஆரஞ்சுப்பழம்','இஞ்சி','இதயம்','இரத்த அழுத்தம்','உடல்','ஊழல் செய்தல்','எலி','ஒட்டகச்சிவிங்கி','ஒட்டகம்','கங்காரு','கண்','கரடி','கலங்கரை விளக்கம்','கழுகு','கழுதை','கழுத்து','காகம்','காண்டாமிருகம்','கிளி','குடியரசுத் தலைவர்','குதிரை','குரங்கு','சாத்துக்குடி','சிட்டுக்குருவி','செம்மறியாடு','சேவல்','தர்பூசணி','தலை','தேங்காய்','தேசீய கீதம்','நரி','பசு','பறவை','பீன்ஸ்','புதிதாக கண்டுபிடி','புரிந்து கொள்ளுதல்','பூசணிக்காய்','பூனை','போக்குவரத்து விளக்கு','மயில்','மான்','மீன்','முகம்','முதலை','முருங்கைக்காய்','மூக்கு','யானை','வரிக்குதிரை','வாத்து','வாய்','வாழைப்பழம்','வான்கோழி','வெள்ளரிக்காய்','வெள்ளாடு'])

# # Thirty videos worth of data for each action
# no_sequences = 40

# # Videos are going to be 30 frames in length (30 frames of data for each action)
# sequence_length = 30

# label_map = {label:num for num, label in enumerate(actions)} #create label map dictionary

# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import LSTM, Dense
# from tensorflow.keras.callbacks import TensorBoard


# # Build LSTM Model Architecture Layers using Keras high-level # experiment 4
# model = Sequential()
# model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(30,258))) #each video has input shape of 30 frames of 1662 keypoints: X.shape
# model.add(LSTM(128, return_sequences=True, activation='relu'))
# model.add(LSTM(64, return_sequences=False, activation='relu')) #next layer is a dense layer so we do not return sequences here
# model.add(Dense(64, activation='relu'))
# model.add(Dense(32, activation='relu'))
# model.add(Dense(actions.shape[0], activation='softmax'))


# # Build LSTM Model Architecture Layers using Keras high-level # experiment 3
# model = Sequential()
# model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(30,258))) #each video has input shape of 30 frames of 1662 keypoints: X.shape
# model.add(LSTM(128, return_sequences=False, activation='relu'))
# #Dense layer with relu
# model.add(Dense(64, activation='relu'))
# model.add(Dense(32, activation='relu'))
# model.add(Dense(actions.shape[0], activation='softmax'))

# # Build LSTM Model Architecture Layers using Keras high-level API # experiment 5
# model = Sequential()
# model.add(LSTM(128, activation='relu', input_shape=(30,258))) #each video has input shape of 30 frames of 1662 keypoints: X.shape
# model.add(Dropout(0.2))
# #Dense layer with relu
# model.add(Dense(64, activation='relu'))
# model.add(Dense(actions.shape[0], activation='softmax'))



# model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])
# # use categorical cross entropy as we are building a multiclass 
# model.load_weights(r"C:\Users\being\OneDrive\Desktop\two-way-sign-language-translator\Epoch-60-Loss-0.23.h5", by_name=True)
# #model.load_weights('tutorial_weights_action.h5')

# colors = [(245,221,173), (245,185,265), (146,235,193),(204,152,295),(255,217,179),(0,0,179)]
# def prob_viz(res, actions, input_frame, colors):
#     output_frame = input_frame.copy()
#     for num, prob in enumerate(res):
#         cv2.rectangle(output_frame, (0,60+num*40), (int(prob*100), 90+num*40), colors[num], -1) #change length of bar depending on probability
#         cv2.putText(output_frame, actions[num]+' '+str(round(prob*100,2))+'%', (0, 85+num*40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)
        
#     return output_frame
    
# # Specify Tamil font file path
# font_file =r"C:\Users\being\Downloads\latha.ttf"
# font_size = 22
# font = ImageFont.truetype(font_file, size=font_size)

# # 1. New detection variables
# sequence = []
# sentence = []
# predictions = []
# threshold = 0.5

# # use computer webcam and make keypoint detections
# cap = cv2.VideoCapture(0)

# # Set mediapipe model 
# with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
#     while cap.isOpened():
#         ret, image = cap.read()
#         if ret == True:    
#             image, results = mediapipe_detection(image, holistic)
#             draw_styled_landmarks(image, results)

#             keypoints = extract_keypoints(results)
#             sequence.append(keypoints)
#             sequence = sequence[-30:]

#             if len(sequence) == 30:
#                 res = model.predict(np.expand_dims(sequence, axis=0))[0]
#                 if res[np.argmax(res)] > threshold: 
#                     predicted_word = actions[np.argmax(res)]
#                     if len(sentence) == 0 or predicted_word != sentence[-1]:
#                         sentence.append(predicted_word)
#                         sentence = [predicted_word]
                
#                 img_pil = Image.fromarray(image)
#                 draw = ImageDraw.Draw(img_pil)
#                 # Convert Tamil sentence to Unicode and display on image
#                 draw.text((3, 30), ' '.join(sentence), font=font, fill=(0, 0, 255, 255), outline=None)

#                 image = np.array(img_pil)

#             cv2.imshow('OpenCV Feed', image)
#             if cv2.waitKey(1) & 0xFF == ord('q'):
#                 break

#     cap.release()
#     cv2.destroyAllWindows()    


# import cv2
# import numpy as np
# import os
# from matplotlib import pyplot as plt
# import time
# import mediapipe as mp


# mp_holistic = mp.solutions.holistic # Holistic model
# mp_drawing = mp.solutions.drawing_utils # Drawing utilities
# def mediapipe_detection(image, model):
#     image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # COLOR CONVERSION BGR 2 RGB
#     image.flags.writeable = False                  # Image is no longer writeable
#     results = model.process(image)                 # Make prediction
#     image.flags.writeable = True                   # Image is now writeable 
#     image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) # COLOR COVERSION RGB 2 BGR
#     return image, results

# def draw_landmarks(image, results):
#     mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_CONTOURS) # Draw face connections
#     mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS) # Draw pose connections
#     mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS) # Draw left hand connections
#     mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS) # Draw right hand connections


# def draw_styled_landmarks(image, results):
#     # Draw face connections
#     mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_CONTOURS, 
#                              mp_drawing.DrawingSpec(color=(80,110,10), thickness=1, circle_radius=1), 
#                              mp_drawing.DrawingSpec(color=(80,256,121), thickness=1, circle_radius=1)
#                              ) 
#     # Draw pose connections
#     mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
#                              mp_drawing.DrawingSpec(color=(80,22,10), thickness=2, circle_radius=4), 
#                              mp_drawing.DrawingSpec(color=(80,44,121), thickness=2, circle_radius=2)
#                              ) 
#     # Draw left hand connections
#     mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
#                              mp_drawing.DrawingSpec(color=(121,22,76), thickness=2, circle_radius=4), 
#                              mp_drawing.DrawingSpec(color=(121,44,250), thickness=2, circle_radius=2)
#                              ) 
#     # Draw right hand connections  
#     mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
#                              mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=4), 
#                              mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
#                              ) 

# # cap = cv2.VideoCapture(0)
# # # Set mediapipe model 
# # with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
# #     while cap.isOpened():

# #         # Read feed
# #         ret, frame = cap.read()

# #         # Make detections
# #         image, results = mediapipe_detection(frame, holistic)
# #         print(results)
        
# #         # Draw landmarks
# #         draw_styled_landmarks(image, results)

# #         # Show to screen
# #         cv2.imshow('OpenCV Feed', image)

# #         # Break gracefully
# #         if cv2.waitKey(10) & 0xFF == ord('q'):
# #             break
# #     cap.release()
# #     cv2.destroyAllWindows()

# # use computer webcam and make keypoint detections
# cap = cv2.VideoCapture(0)

# # Set mediapipe model configurations
# min_detection_confidence = 0.5
# min_tracking_confidence= 0.5

# with mp_holistic.Holistic(min_detection_confidence=min_detection_confidence, min_tracking_confidence=min_tracking_confidence) as holistic:
#     while cap.isOpened():
#         # Read feed
#         ret, frame = cap.read()

#         # Make detections by calling our function
#         image, results = mediapipe_detection(frame, holistic) #mediapipe_detection(image, model) 
#         #print(results)
#         #print(results.face_landmarks)
        
#         # Draw landmarks
#         draw_styled_landmarks(image, results)
        
#         # Show to screen
#         cv2.imshow('OpenCV Feed: Hold Q to Quit', image)

#         # Break gracefully
#         if cv2.waitKey(10) & 0xFF == ord('q'): #press q to quit
#             break
#     cap.release() #release webcam
#     cv2.destroyAllWindows()



# len(results.left_hand_landmarks.landmark)

# pose = []
# for res in results.pose_landmarks.landmark:
#     test = np.array([res.x, res.y, res.z, res.visibility])
#     pose.append(test)

# pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(132)
# face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(1404)
# lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
# rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)    

# def extract_keypoints(results):
#     pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
#     face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468*3)
#     lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
#     rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
#     return np.concatenate([pose, face, lh, rh])
# result_test = extract_keypoints(results)

# np.save('0', result_test)
# np.load('0.npy')
# # Path for exported data, numpy arrays
# DATA_PATH = os.path.join('test') 

# # Actions that we try to detect
# # actions = np.array(['hello', 'thanks', 'iloveyou'])
# actions = np.array(['அணில்','அறுவடை செய்','அறுவை சிகிச்சை','ஆந்தை','ஆப்பிள்','ஆரஞ்சுப்பழம்','இஞ்சி','இதயம்','இரத்த அழுத்தம்','உடல்','ஊழல் செய்தல்','எலி','ஒட்டகச்சிவிங்கி','ஒட்டகம்','கங்காரு','கண்','கரடி','கலங்கரை விளக்கம்','கழுகு','கழுதை','கழுத்து','காகம்','காண்டாமிருகம்','கிளி','குடியரசுத் தலைவர்','குதிரை','குரங்கு','சாத்துக்குடி','சிட்டுக்குருவி','செம்மறியாடு','சேவல்','தர்பூசணி','தலை','தேங்காய்','தேசீய கீதம்','நரி','பசு','பறவை','பீன்ஸ்','புதிதாக கண்டுபிடி','புரிந்து கொள்ளுதல்','பூசணிக்காய்','பூனை','போக்குவரத்து விளக்கு','மயில்','மான்','மீன்','முகம்','முதலை','முருங்கைக்காய்','மூக்கு','யானை','வரிக்குதிரை','வாத்து','வாய்','வாழைப்பழம்','வான்கோழி','வெள்ளரிக்காய்','வெள்ளாடு'])


# # Thirty videos worth of data
# no_sequences = 40

# # Videos are going to be 30 frames in length
# sequence_length = 30

# # Folder start
# # start_folder = 30

# #  loop through the actions we are detecting and make folders to store keypoints as numpy arrays
# for action in actions: 
#     #dirmax = np.max(np.array(os.listdir(os.path.join(DATA_PATH, action))).astype(int))
#     # loop through the sequences that we are collecting
#     for sequence in range(1,no_sequences+1):
#         try: #if the directory do not exist, we create a new directory to store the frames
#             #os.makedirs(os.path.join(DATA_PATH, action, str(dirmax+sequence)))
#             os.makedirs(os.path.join(DATA_PATH, action, str(sequence)))
#         except:
#             pass

# from sklearn.model_selection import train_test_split
# from tensorflow.keras.utils import to_categorical
# #create label map dictionary
# label_map = {label:num for num, label in enumerate(actions)} 
# # print(label_map)

# #sequences represent x data, labels represent y data/the action classes.
# sequences, labels = [], []
# #Loop through the action classes you want to detect
# for action in actions:
#     #loop through each sequence
#     for sequence in np.array(os.listdir(os.path.join(DATA_PATH, action))).astype(int):
#         window = []
#         for frame_num in range(sequence_length):
#             res = np.load(os.path.join(DATA_PATH, action, str(sequence), "{}.npy".format(frame_num)))
#             window.append(res)
#         sequences.append(window)
#         labels.append(label_map[action])

# X = np.array(sequences)
# y = to_categorical(labels).astype(int)
# # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05)

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20,stratify=y)

# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import LSTM, Dense
# from tensorflow.keras.callbacks import TensorBoard

# log_dir = os.path.join('Logs')
# tb_callback = TensorBoard(log_dir=log_dir)

# # model = Sequential()
# # model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(30,1662)))
# # model.add(LSTM(128, return_sequences=True, activation='relu'))
# # model.add(LSTM(64, return_sequences=False, activation='relu'))
# # model.add(Dense(64, activation='relu'))
# # model.add(Dense(32, activation='relu'))
# # model.add(Dense(actions.shape[0], activation='softmax'))

# # Build LSTM Model Architecture Layers using Keras high-level # experiment 3
# model = Sequential()
# model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(30,258))) #each video has input shape of 30 frames of 1662 keypoints: X.shape
# model.add(LSTM(128, return_sequences=False, activation='relu'))
# #Dense layer with relu
# model.add(Dense(64, activation='relu'))
# model.add(Dense(32, activation='relu'))
# model.add(Dense(actions.shape[0], activation='softmax'))

# # Build LSTM Model Architecture Layers using Keras high-level # experiment 4
# model = Sequential()
# model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(30,258))) #each video has input shape of 30 frames of 1662 keypoints: X.shape
# model.add(LSTM(128, return_sequences=True, activation='relu'))
# model.add(LSTM(64, return_sequences=False, activation='relu')) #next layer is a dense layer so we do not return sequences here
# model.add(Dense(64, activation='relu'))
# model.add(Dense(32, activation='relu'))
# model.add(Dense(actions.shape[0], activation='softmax'))

# # Build LSTM Model Architecture Layers using Keras high-level API # experiment 5
# model = Sequential()
# model.add(LSTM(128, activation='relu', input_shape=(30,258))) #each video has input shape of 30 frames of 1662 keypoints: X.shape
# model.add(Dropout(0.2))
# #Dense layer with relu
# model.add(Dense(64, activation='relu'))
# model.add(Dense(actions.shape[0], activation='softmax'))

# model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])


# model.fit(X_train, y_train, epochs=2000, callbacks=[tb_callback])



# model.save('action.h5')


# model.load_weights('action.h5')

# import cv2
# import numpy as np
# from PIL import ImageFont, ImageDraw, Image

# # Specify Tamil font file path
# font_file =r"C:\Users\being\Downloads\latha.ttf"
# font_size = 22
# font = ImageFont.truetype(font_file, size=font_size)

# # 1. New detection variables
# sequence = []
# sentence = []
# predictions = []
# threshold = 0.5

# # use computer webcam and make keypoint detections
# cap = cv2.VideoCapture(0)

# # Set mediapipe model 
# with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
#     while cap.isOpened():
#         ret, image = cap.read()
#         if ret == True:    
#             image, results = mediapipe_detection(image, holistic)
#             draw_styled_landmarks(image, results)

#             keypoints = extract_keypoints(results)
#             sequence.append(keypoints)
#             sequence = sequence[-30:]

#             if len(sequence) == 30:
#                 res = model.predict(np.expand_dims(sequence, axis=0))[0]
#                 if res[np.argmax(res)] > threshold: 
#                     predicted_word = actions[np.argmax(res)]
#                     if len(sentence) == 0 or predicted_word != sentence[-1]:
#                         sentence.append(predicted_word)
#                         sentence = [predicted_word]
                
#                 img_pil = Image.fromarray(image)
#                 draw = ImageDraw.Draw(img_pil)
#                 # Convert Tamil sentence to Unicode and display on image
#                 draw.text((3, 30), ' '.join(sentence), font=font, fill=(0, 0, 255, 255), outline=None)

#                 image = np.array(img_pil)

#             cv2.imshow('OpenCV Feed', image)
#             if cv2.waitKey(1) & 0xFF == ord('q'):
#                 break

#     cap.release()
#     cv2.destroyAllWindows()


