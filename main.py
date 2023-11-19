import cv2
import pygame
import numpy as np
import os
import streamlit as st
import pickle
import matplotlib.pyplot as pltno
import time
import mediapipe as mp
import glob
import tensorflow as tf
from scipy import stats
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.metrics import multilabel_confusion_matrix, accuracy_score
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences



# Path for Data
DATA_PATH = os.path.join('MP_Data')

# Actions we're detecting
actions = np.array(['dancing', 'not dancing'])
num_classes = len(actions)
# Thirty vidoes worth of data
no_sequences = 15

# Vidoes are going to be 30 frames in length
sequence_length = 60

# Log Creation
log_dir = os.path.join('Logs')
tb_callback = TensorBoard(log_dir=log_dir)

# Define model
model = 'action.h5'

not_dancing_time = 0
not_dancing_threshold = 100


# Initialize Pygame mixer
pygame.mixer.init()

# Function to play an MP3 file
def play_mp3(file_path):
    pygame.mixer.music.load(file_path)
    pygame.mixer.music.play()

# # Function to handle the playing of songs in a separate thread
# def handle_songs(folder_path, play_next_event, song_files):
#     current_song = 0
#     play_mp3(os.path.join(folder_path, song_files[current_song][2]))

#     while True:
#         play_next_event.wait()  # Wait for the signal to play the next song
#         play_next_event.clear()

#         # Move to next song
#         current_song = (current_song + 1) % len(song_files)
#         play_mp3(os.path.join(folder_path, song_files[current_song]))

file = open("songs_data.pkl", 'rb')
song_data = pickle.load(file)
file.close()

# st.set_page_config(page_title='AFK DJ',  layout='wide')
# st.markdown("""
# <style>
# 	[data-testid="stHeader"] {
# 		background-image: linear-gradient(90deg, rgb(57, 163, 161), rgb(170, 115, 199));
# 	}
# </style>""",
# unsafe_allow_html=True)
# st.title("AFK DJ")
# st.write("Currently Playing " + song_data[5][0])
# key1 = 0
# logtxtbox = st.empty()
# logtxt = song_data[5][0]
# logtxtbox.text_area("Currently Playing ",logtxt, height = 50, key=key1)
# st.markdown('''
#     <a href="https://docs.streamlit.io">
#         <img src=song_data[5][3] />
#     </a>''',
#     unsafe_allow_html=True
# )

current_song = [5]

def perform_action(folder_path, action, song_files, current_song):
    name = song_files[current_song[0]][2] + ".mp3"
    global not_dancing_time
    if action == "not dancing":
        not_dancing_time += 1
    
        

    else:
        not_dancing_time = 0

    if not_dancing_time >= not_dancing_threshold:
        current_song[0] = (current_song[0] + 1) % len(song_files)
        name = song_files[current_song[0]][2] + ".mp3"
        play_mp3(os.path.join(folder_path, name))
        not_dancing_time=0
    
    


        

# Create MP_Data folder structure
for action in actions: 
    for sequence in range(no_sequences):
        sequence_path = os.path.join(DATA_PATH, action, str(sequence))
        if not os.path.exists(sequence_path):
            try: 
                os.makedirs(sequence_path)
            except:
                pass


mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose=mp.solutions.pose

# Label map
label_map = {label:num for num, label in enumerate(actions)}



def create_model(input_shape, num_classes):
    model = Sequential()
    model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=input_shape))
    model.add(LSTM(128, return_sequences=True, activation='relu'))
    model.add(LSTM(64, return_sequences=False, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))
    return model


def compile_model(model, optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy']):
    # Model Compiling
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    
    return model



def collect_data(actions, no_sequences, sequence_length):
    for action in actions: 
        for sequence in range(no_sequences):
            sequence_path = os.path.join(DATA_PATH, action, str(sequence))
            
            # Set mediapipe model 
            with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
                for frame_num in range(sequence_length):
                    # Read feed
                    ret, frame = cap.read()

                    # Make detections
                    image, results = mediapipe_detection(frame, pose)
                    
                    # Draw landmarks
                    draw_landmarks(image, results)
                    
                    # Apply wait logic
                    if frame_num == 0: 
                        cv2.putText(image, 'STARTING COLLECTION', (120,200), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255, 0), 4, cv2.LINE_AA)
                        cv2.putText(image, 'Collecting frames for {} Video Number {}'.format(action, sequence), (15,12), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                        # Show to screen
                        cv2.imshow('OpenCV Feed', image)
                        cv2.waitKey(2000)
                    else: 
                        cv2.putText(image, 'Collecting frames for {} Video Number {}'.format(action, sequence), (15,12), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                        # Show to screen
                        cv2.imshow('OpenCV Feed', image)

                    # Export keypoints
                    keypoints = extract_keypoints(results)
                    npy_path = os.path.join(sequence_path, str(frame_num))
                    np.save(npy_path, keypoints)

                    # Break gracefully
                    if cv2.waitKey(10) & 0xFF == ord('q'):
                        break

    cap.release()
    cv2.destroyAllWindows()

def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # Color Conversion BGR 2 RGB
    image.flags.writeable = False                  # Image is not longer writeable
    results = model.process(image)                 # Make Prediction
    image.flags.writeable = True                   # Image is now writeable
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) # Color Conversion RGB 2 BGR
    return image, results

# Draw landmarks
def draw_landmarks(image, results):
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

def draw_styled_landmarks(image, results):
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                  mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4),
                                  mp_drawing.DrawingSpec(color=(121, 44, 250), thickness=2, circle_radius=2))

# Extract important points
def extract_keypoints(results):
    if results.pose_landmarks:
        # Extract pose landmarks
        return np.array([[res.x, res.y, res.z] for res in results.pose_landmarks.landmark]).flatten()
    else:
        return np.zeros(33 * 3)  # Adjust the number based on the pose landmarks
    

# Function for pre-processing data
def preprocess_data(actions, no_sequences, sequence_length, label_map):
    sequences, labels = [], []
    for action in actions:
        for sequence in range(no_sequences):
            window = []
            for frame_num in range(sequence_length):
                res = np.load(os.path.join(DATA_PATH, action, str(sequence), "{}.npy".format(frame_num)))
                window.append(res)
            sequences.append(window)
            labels.append(label_map[action])





    x = np.array(sequences)
    y = to_categorical(labels).astype(int)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.05)
    
    return x_train, x_test, y_train, y_test


colors = [(245,117,16), (117,245,16), (16,117,245)]
def prob_viz(res, actions, input_frame, colors):
    output_frame = input_frame.copy()
    for num, prob in enumerate(res):
        cv2.rectangle(output_frame, (0,60+num*40), (int(prob*100), 90+num*40), colors[num], -1)
        cv2.putText(output_frame, actions[num], (0, 85+num*40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)
        
    return output_frame




cap = cv2.VideoCapture(0)



# Ask user if they want to collect new data
collect_data_decision = input("Would you like to collect new data? (yes/no): ")
if collect_data_decision.lower() == "yes":
    collect_data(actions, no_sequences, sequence_length)


# Pre-process code
x_train, x_test, y_train, y_test = preprocess_data(actions, no_sequences, sequence_length, label_map)

# Save or load model
print("Would you like to: 1. Train a new model 2. Load an existing model")
choice = input("Enter choice (1 or 2): ")

if choice == '1':
    
    
    
    # Create, compile and train the model
    model = create_model((60, 99), num_classes)
    model = compile_model(model)

    model.fit(x_train, y_train, epochs=1000, callbacks=[tb_callback])

    # After training, save the model
    model.save('action.h5')

elif choice == '2':
    # Load the model
    model = tf.keras.models.load_model('action.h5')
else:
    print("Invalid choice. Please enter either 1 or 2.")

# Predictions
yhat = model.predict(x_test)
ytrue = np.argmax(y_test, axis=1).tolist()
yhat = np.argmax(yhat, axis=1).tolist()




# 1. New detection variables
sequence = []
sentence = []
predictions = []
threshold = 0.5

play_mp3(os.path.join("songs", song_data[current_song[0]][2] + ".mp3"))

cap = cv2.VideoCapture(0)
# Set mediapipe model 
with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while cap.isOpened():

        # Read feed
        ret, frame = cap.read()

        # Make detections
        image, results = mediapipe_detection(frame, pose)
        print(results)
        
        # Draw landmarks
        draw_styled_landmarks(image, results)
        
        # 2. Prediction logic
        keypoints = extract_keypoints(results)
        sequence.append(keypoints)
        sequence = sequence[-30:]
        
        if len(sequence) == 30:
            res = model.predict(np.expand_dims(sequence, axis=0))[0]
            action = actions[np.argmax(res)]
            perform_action("songs", action, song_data, current_song)
            # logtxt = song_data[current_song[0]][0]
            # key1 = key1 + 1
            # logtxtbox.text_area("Currently Playing: ", logtxt, height=50, key=key1)
            # st.markdown("![Alt Text]" + song_data[current_song[0]][3])
            print(action)
            predictions.append(np.argmax(res))
            
            
        #3. Viz logic
            if np.unique(predictions[-10:])[0]==np.argmax(res): 
                if res[np.argmax(res)] > threshold: 
                    
                    if len(sentence) > 0: 
                        if actions[np.argmax(res)] != sentence[-1]:
                            sentence.append(actions[np.argmax(res)])
                    else:
                        sentence.append(actions[np.argmax(res)])

            if len(sentence) > 5: 
                sentence = sentence[-5:]

            # Viz probabilities
            image = prob_viz(res, actions, image, colors)
            
        cv2.rectangle(image, (0,0), (640, 40), (245, 117, 16), -1)
        cv2.putText(image, ' '.join(sentence), (3,30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        
        # Show to screen
        cv2.imshow('OpenCV Feed', image)

        # Break gracefully
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()