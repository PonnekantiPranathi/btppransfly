# flask run --host=0.0.0.0 --port=5000 - Start the server in local machine with this command
# Then in the browser go to 'http://localhost:5000/video_page'
from ultralytics import YOLO
import cv2
from flask import Flask, render_template, Response
from flask_socketio import SocketIO, emit
import openai
import base64
import time
import threading
import face_recognition


app = Flask(__name__)
socketio = SocketIO(app)
model = YOLO("yolov10x.pt")
detected_objects = []  # Stores detected objects here
task_ended = False  # Flag to check if the task is completed
current_angle = 0  
last_frame = None  # Global variable to store the last frame
detect_only_mode = threading.Event()  # Event for "detect only" mode
target_object = ""  # Store the target object for "detect only" mode


known_face_encodings = []
known_face_names = []


# Generate frames with YOLO detection
def generate_frames():
    global detected_objects, last_frame, target_object, known_face_encodings, known_face_names
    cap = cv2.VideoCapture(0)
    print("Starting video capture...")
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    cap.set(cv2.CAP_PROP_FPS,60)
    
    while True:
        success, frame = cap.read()
        if not success:
            print("Failed to capture video frame.")
            break

        results = model.predict(source=frame, show=False, verbose=False)
        annotated_frame = frame.copy()  # Start with a clean copy for annotations

        # Update detected_objects with YOLO predictions
        detected_objects = [model.names[int(cls)] for cls in results[0].boxes.cls]
        last_frame = annotated_frame  

        if detect_only_mode.is_set(): 
            print(f"'Detect only' mode active. Target object: {target_object}")
            target_found = False
            
            
            for box, cls in zip(results[0].boxes.xyxy, results[0].boxes.cls):
                obj_name = model.names[int(cls)]
                
                if obj_name == target_object:
                    target_found = True
                    print(f"Target object '{target_object}' found. Drawing bounding box.")
                    x1, y1, x2, y2 = map(int, box)  
                    # Draw bounding box only for the target object
                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(annotated_frame, target_object, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            if not target_found:
                print(f"Target object '{target_object}' not found in this frame.")
            
            # Encode and yield the frame whether the target object was found or not
            ret, buffer = cv2.imencode('.jpg', annotated_frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

        else:
            # Normal mode: Draw bounding boxes for all detected objects
            # print("Normal mode active. Drawing bounding boxes for all detected objects.")
            for box, cls, conf in zip(results[0].boxes.xyxy, results[0].boxes.cls, results[0].boxes.conf):
                obj_name = model.names[int(cls)]
                x1, y1, x2, y2 = map(int, box)

                if obj_name == "person":  # Process only confident "person" detections
                    cropped_face = frame[y1:y2, x1:x2]  # Crop potential face area
                    rgb_face = cv2.cvtColor(cropped_face, cv2.COLOR_BGR2RGB)
                    face_encodings = face_recognition.face_encodings(rgb_face)

                    if face_encodings:
                        face_encoding = face_encodings[0]
                        matches = face_recognition.compare_faces(known_face_encodings, face_encoding, tolerance=0.6)

                        iname = "Unknown"

                        if True in matches:
                            # Match found
                            match_index = matches.index(True)
                            iname = known_face_names[match_index]
                        else:
                            # Prompt user to enter the name for the unrecognized person
                            print("New face detected. Enter the person's name:")
                            name = input("Name: ").strip()

                            if name.lower() == "exit":
                                print("Skipping this face.")
                                
                            elif name and name not in known_face_names:
                                # Save the face encoding and name if not already known
                                known_face_encodings.append(face_encoding)
                                known_face_names.append(name)
                                iname = name
                            else:
                                print(f"The name '{name}' already exists or is invalid. Skipping.")

                        # Annotate frame
                        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(annotated_frame, iname, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    else:
                        # If no face is detected in the bounding box
                        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                        cv2.putText(annotated_frame, "person", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
                else:
                    # Annotate non-person objects
                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    cv2.putText(annotated_frame, obj_name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

            ret, buffer = cv2.imencode('.jpg', annotated_frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()

# Function to enable "detect only" mode
def enable_detect_only_mode(target):
    global target_object
    target_object = target
    detect_only_mode.set()  # Enable "detect only" mode
    print(f"'Detect only' mode enabled for target: {target_object}")
    
# Function to reset and return to normal mode
def reset_to_normal_mode():
    global target_object
    detect_only_mode.clear()  # Disable "detect only" mode
    target_object = ""  # Clear the target object
    print("'Detect only' mode disabled. Returning to normal mode.")

@app.route('/video_feed')
def video_stream():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/video_page')
def video_page():
    return render_template('index.html')  # Render the HTML template

# Function to encode the image frame
def encode_image(image):
    _, buffer = cv2.imencode('.jpg', image)
    return base64.b64encode(buffer).decode('utf-8')

# WebSocket event for processing questions from the client
@socketio.on('send_question')
def handle_question(data):
    global detected_objects, task_ended, current_angle, last_frame

    question = data['question']
    response_message = ""
    task_ended = False
    current_angle = 0

    
    reset_to_normal_mode()

    
    if "detect only" in question.lower():
        
        detect_only_index = question.lower().find("detect only")
        target = question[detect_only_index + len("detect only"):].strip()

        if target:  
            
            threading.Thread(target=enable_detect_only_mode, args=(target,)).start()
        else:
            print("No target object specified after 'detect only'.")

    while not task_ended and current_angle < 360:
        if detected_objects and last_frame is not None:
            # Encode the last frame in base64 format
            base64_image = encode_image(last_frame)
            
            # Generate the prompt for OpenAI based on detected objects and image
            gpt_prompt = (f"The user asked: '{question}'. The following objects are currently detected in the video feed: "
                          f"{', '.join(detected_objects)}. Based on this information, please add a 'yes' along with the gpt response if it is possible to provide an answer relevant to the user's question, "
                          "or 'no' along with the gpt response if it is not possible.")
            
            response_message = query_openai(gpt_prompt, base64_image)
            print(response_message)  
            
            if "yes" in response_message.lower():  # Positive sentiment suggests relevance
                print("Positive response detected. Ending task.")
                task_ended = True
                break
            elif "no" in response_message.lower():
                print(f"No positive relevance in response. Command: Rotate to {current_angle + 45} degrees")
        
        # Increment the rotation angle by 45 degrees and check again
        current_angle += 45
        socketio.sleep(10)  # Wait for some time for the next detection check

    # After completing 360 degrees or finding the object, finalize the task
    if current_angle >= 360:
        task_ended = True
        print("Completed a full rotation without detecting the relevant object.")
        response_message = "No relevant object found after full rotation."
        
    emit('receive_answer', {'response': f"{response_message} Task ended."})

def query_openai(gpt_prompt, base64_image):
    openai.api_key = "sk-gKbmIkCmTPptcsx474iGFZ3m4H4dQO8X7KiOXQjwreT3BlbkFJjqq0hb0UN029BlBDiKqKFA3nGUeY20KFuSB8D06eMA"  # Replace with your actual API key
    response = openai.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
              "role": "user",
              "content": [
                {
                  "type": "text",
                  "text": gpt_prompt,
                },
                {
                  "type": "image_url",
                  "image_url": {
                    "url":  f"data:image/jpeg;base64,{base64_image}"
                  },
                },
              ],
            }
        ]
    )
    return response.choices[0].message.content

if __name__ == '__main__':
    socketio.run(app, host='0.0.0.0', port=5000, debug=True)


