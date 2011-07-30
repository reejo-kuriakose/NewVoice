import pyrebase
import numpy as np
import cv2
import mediapipe as mp
import joblib
import time
import webbrowser
import time
import os, sys
import pyttsx3
from random import random
from english_words import english_words_lower_alpha_set
from logging import FileHandler,WARNING
from flask import Flask,redirect, session,Response, render_template, request , url_for, jsonify

config = {"apiKey": "AIzaSyAP0Es6cmLbuzvjQmbb5yC6HiAb160DJpo",
    "authDomain": "newvoice-7ff3b.firebaseapp.com",
    "projectId": "newvoice-7ff3b",
    "storageBucket": "newvoice-7ff3b.appspot.com",
    "messagingSenderId": "112761611506",
    "appId": "1:112761611506:web:c78abd429f18ef6c559c6b",
    "databaseURL": ""
}

firebase = pyrebase.initialize_app(config)
auth = firebase.auth()

app = Flask(__name__)

app.secret_key = 'secret'

global easy,free,freestyle,text,st_time,easy_word_user
txt="Translated Text goes here"
easy=0
free=1

clf = joblib.load("model.joblib")

def text_to_speech(text):
    engine = pyttsx3.init()
    engine.setProperty('rate', 125)
    engine.setProperty('volume',0.8)
    voices = engine.getProperty('voices')
    engine.setProperty('voice','Female')
    engine.say(text)
    engine.runAndWait()


def camera_max():
    '''Returns int value of available camera devices connected to the host device'''
    camera = -1
    
    while True:
        if (cv2.VideoCapture(camera).grab()):
            camera = camera + 1
        else:
            cv2.destroyAllWindows()
            return(max(0,int(camera-1)))
        
cam_max = camera_max()

cap = cv2.VideoCapture(cam_max , cv2.CAP_DSHOW)

letters = ['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z']
words = [i for i in sorted(list(english_words_lower_alpha_set)) if 'z' not in i and len(i) > 3 and len(i) <= 10]
start_time = time.time()
curr_time = 0
easy_word_user = ''
eraser = 0
easy_word = words[int(random()*len(words))].upper()
easy_word_index = 0
location = 0
letter_help = 0


def easy_mode(frame):
    global cap, easy_word_user, easy_word, easy_word_index, curr_time, location, letter_help
    
    def mediapipe_detection(image, model):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # COLOR CONVERSION BGR 2 RGB
        results = model.process(image)                 # Make prediction
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) # COLOR CONVERSION RGB 2 BGR
        return image, results

    def get_landmark_dist_test(results, x, y):
        hand_array = []
        wrist_pos = results.multi_hand_landmarks[0].landmark[0]
        for result in results.multi_hand_landmarks[0].landmark:
            hand_array.append((result.x-wrist_pos.x) * (width/x))
            hand_array.append((result.y-wrist_pos.y) * (height/y))
        return(hand_array[2:])


    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Set mediapipe model
    mp_hands = mp.solutions.hands # Hands model
    with mp_hands.Hands(min_detection_confidence=0.6, min_tracking_confidence=0.6, max_num_hands=1) as hands:
        while cap.isOpened():


            try:
                cv2.putText(frame, easy_word, (int(width*0.05), int(height*0.95)), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 100, 100), 2, cv2.LINE_4)
                cv2.putText(frame, easy_word_user, (int(width*0.05), int(height*0.95)), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 2, cv2.LINE_4)
            except Exception as e:
                print(e)

            # Make detections
            image, results = mediapipe_detection(frame, hands)

            letter_help = cv2.resize(cv2.imread('easy_mode_letters/{}.png'.format(easy_word[easy_word_index].lower())), (0,0), fx=0.2, fy=0.2)

            #Find bounding box of hand
            if results.multi_hand_landmarks:
                x = [None,None]
                y=[None,None]
                for result in results.multi_hand_landmarks[0].landmark:
                    if x[0] is None or result.x < x[0]: x[0] = result.x
                    if x[1] is None or result.x > x[1]: x[1] = result.x

                    if y[0] is None or result.y < y[0]: y[0] = result.y
                    if y[1] is None or result.y > y[1]: y[1] = result.y


                if curr_time < round((time.time() - start_time)/3,1) and x[0] is not None:
                        curr_time = round((time.time() - start_time)/3,1)
                        try:
                            test_image = get_landmark_dist_test(results, x[1]-x[0], y[1]-y[0])
                            test_pred = np.argmax(clf.predict_proba(np.array([test_image])))
                            test_probs = clf.predict_proba(np.array([test_image]))[0]
                            print("Predicted:",letters[test_pred], ", pred prob:", max(test_probs), ", current index:", easy_word_index, ", current time:", curr_time)
                            if max(test_probs) >= 0.8 or (max(test_probs) >= 0.6 and letters[test_pred] in ['p','r','u','v','n','m','t']):
                                pred_letter = letters[test_pred].upper()
                                if easy_word_index < len(easy_word) and pred_letter == easy_word[easy_word_index] and (easy_word_index == 0 or easy_word[easy_word_index] != easy_word[easy_word_index - 1]):
                                    easy_word_user += pred_letter
                                    easy_word_index += 1
                                    location = results.multi_hand_landmarks[0].landmark[0].x
                                if easy_word_index < len(easy_word) and pred_letter == easy_word[easy_word_index] and easy_word_index > 0 and easy_word[easy_word_index] == easy_word[easy_word_index - 1] and abs(location - results.multi_hand_landmarks[0].landmark[0].x) > 0.1:
                                    easy_word_user += pred_letter
                                    easy_word_index += 1
                                    location = results.multi_hand_landmarks[0].landmark[0].x

                            if easy_word_user == easy_word:
                                time.sleep(0.5)
                                easy_word = words[int(random()*len(words))].upper()
                                easy_word_index = 0
                                easy_word_user = ''

                        except Exception as e:
                            print(e)

            # Show letter helper
            frame[5:5+letter_help.shape[0],width-5-letter_help.shape[1]:width-5] = letter_help

            return frame
            
    return frame

#freestyle
def free_mode(frame):
    global cap, easy_word_user, easy_word, easy_word_index, curr_time, location, letter_help
    
    def mediapipe_detection(image, model):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # COLOR CONVERSION BGR 2 RGB
        results = model.process(image)                 # Make prediction
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) # COLOR CONVERSION RGB 2 BGR
        return image, results

    def get_landmark_dist_test(results, x, y):
        hand_array = []
        wrist_pos = results.multi_hand_landmarks[0].landmark[0]
        for result in results.multi_hand_landmarks[0].landmark:
            hand_array.append((result.x-wrist_pos.x) * (width/x))
            hand_array.append((result.y-wrist_pos.y) * (height/y))
        return(hand_array[2:])


    #Main function
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Set mediapipe model
    mp_hands = mp.solutions.hands # Hands model
    with mp_hands.Hands(min_detection_confidence=0.6, min_tracking_confidence=0.6, max_num_hands=1) as hands:
        while cap.isOpened():


            # try:
            #     cv2.putText(frame, easy_word_user, (int(width*0.05), int(height*0.95)), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2, cv2.LINE_4)
            # except Exception as e:
            #     print(e)

            # Make detections
            image, results = mediapipe_detection(frame, hands)

            #Find bounding box of hand
            if results.multi_hand_landmarks:
                x = [None,None]
                y=[None,None]
                for result in results.multi_hand_landmarks[0].landmark:
                    if x[0] is None or result.x < x[0]: x[0] = result.x
                    if x[1] is None or result.x > x[1]: x[1] = result.x

                    if y[0] is None or result.y < y[0]: y[0] = result.y
                    if y[1] is None or result.y > y[1]: y[1] = result.y


                if curr_time < round((time.time() - start_time)/3,1) and x[0] is not None:
                        st_time=curr_time
                        curr_time = round((time.time() - start_time)/3,1)
                        try:
                            test_image = get_landmark_dist_test(results, x[1]-x[0], y[1]-y[0])
                            test_pred = np.argmax(clf.predict_proba(np.array([test_image])))
                            test_probs = clf.predict_proba(np.array([test_image]))[0]
                            print("Predicted:",letters[test_pred], ", pred prob:", max(test_probs), ", current index:", easy_word_index, ", current time:", curr_time)
                            if (curr_time-st_time)>1.5 and easy_word_index!=0:
                                easy_word_user +=' '
                                easy_word_index +=1
                            if (letters[test_pred] not in ['q','r'] and max(test_probs) >= 0.8) or (max(test_probs) >= 0.6 and letters[test_pred] in ['p','u','v','n','t','k']) or (letters[test_pred] in ['q','r'] and max(test_probs)>=0.9):
                                pred_letter = letters[test_pred].upper()
                                if (easy_word_index == 0):
                                    easy_word_user += pred_letter
                                    easy_word_index += 1
                                    location = results.multi_hand_landmarks[0].landmark[0].x
                                elif easy_word_user[easy_word_index-1] != pred_letter:
                                    easy_word_user += pred_letter
                                    easy_word_index += 1
                                    location = results.multi_hand_landmarks[0].landmark[0].x
                                elif easy_word_user[easy_word_index-1] == pred_letter and (abs(location - results.multi_hand_landmarks[0].landmark[0].x) > 0.15):
                                    easy_word_user += pred_letter
                                    easy_word_index += 1
                                    location = results.multi_hand_landmarks[0].landmark[0].x
                        except Exception as e:
                            print(e)
            return frame
            
    return frame


def sign_frame():  # generate frame by frame from camera
    global easy, cap
    while True:
        success, frame = cap.read() 
        if success:
            if(easy):                
                frame = easy_mode(frame)
            elif(free):                
                frame = free_mode(frame)
   
            try:
                ret, buffer = cv2.imencode('.jpg', frame)
                frame = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            except Exception as e:
                pass

@app.route('/requests',methods=['POST','GET'])
def mode():
    global easy,easy_word_index,easy_word_user,free
    easy_word_index=0
    easy_word_user=''
    if request.method == 'POST':
        if request.form.get('learn') == 'Learn':
            easy= 1
            free =  0
        elif  request.form.get('free') == 'Freestyle':
            free=1  
            easy = 0
        elif request.form.get('stop') == 'Stop':
            free=0
            easy=0
    elif request.method=='GET':
        return render_template('camera.html')
    return render_template('camera.html',text=easy_word_user)

@app.route('/speech',methods=['POST','GET'])
def speech():
    global easy_word_user
    if request.method == 'POST':
        if request.form.get('speech') == 'Speech':
            gender='Female'
            todo=request.form.get('speech')
            print(todo)
            text_to_speech(easy_word_user)
    return render_template('camera.html')

@app.route('/video_feed')
def video_feed():
    return Response(sign_frame(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/_stuff', methods = ['GET'])
def stuff():
    print(easy_word_user)
    return jsonify(result=easy_word_user)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/login', methods = ['GET', 'POST'])
def login():
    if ('user' in session):
        return redirect("/")
    else:
        if request.method == 'POST':
            email = request.form['mail']
            password = request.form['pass']
            try:
                auth.sign_in_with_email_and_password(email, password)
                user_info = auth.sign_in_with_email_and_password(email, password)
                account_info = auth.get_account_info(user_info['idToken'])
                session['user'] = email
                return render_template('camera.html',text=easy_word_user)
            except:
                unsuccessful = 'Please check your credentials'
                return render_template('login.html', umessage = unsuccessful)
        return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if ('user' in session):
        return redirect("/")
    else:
        if request.method =='POST':
            pass0 = request.form['pass0']
            pass1 = request.form['pass1']
            if pass0 == pass1:
                try:
                    email = request.form['email']
                    password = request.form['pass1']
                    auth.create_user_with_email_and_password(email,password)
                    session['user'] = email
                    return render_template('camera.html',text=easy_word_user)
                except:
                    existing_account = 'This email is already being used'
                    return render_template('register.html',exist_message=existing_account)
        return render_template('register.html')

@app.route('/camera')
def camera():
    if ('user' in session):
        return render_template('camera.html',text=easy_word_user)
    else:
        return redirect("/login")

@app.route('/logout')
def logout():
    session.pop('user')
    return redirect('/')

if __name__ == "__main__": 
    app.run(debug=True)

file_handler = FileHandler('errorlog.txt')
file_handler.setLevel(WARNING)