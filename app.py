from flask import Flask, render_template, request, jsonify, send_file, send_from_directory
import os
import time
from werkzeug.utils import secure_filename
import threading
import torch
from torchvision import transforms
import cv2
import numpy as np
import shutil
import threading  
from utils.datasets import letterbox
from utils.general import non_max_suppression_kpt
from utils.plots import output_to_keypoint, plot_skeleton_kpts
from moviepy.editor import ImageSequenceClip
from exercises import run_exercise
import subprocess
from flask_socketio import SocketIO
import base64
import csv
from flask import Response
import uuid 


app = Flask(__name__, static_folder='static')
app.config['UPLOAD_FOLDER'] = 'client/static/uploads'
app.config['OUTPUT_FOLDER'] = 'client/static'
app.config['VIDEO_FOLDER'] = 'client/static'

# socketio = SocketIO(app)
socketio = SocketIO(app, engineio_logger=True, max_http_buffer_size=2000000)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def load_model():
    model = torch.load('yolov7-w6-pose.pt', map_location=device)['model']
    # Put in inference mode
    model.float().eval()

    if torch.cuda.is_available():
        # half() turns predictions into float16 tensors
        # which significantly lowers inference time
        torch.cuda.empty_cache()
        print("GPU:", torch.cuda.get_device_name(device))
        model.half().to(device)
    else:
        print("No GPU available")

    
    return model

model = load_model()


def run_inference(image):
    # Resize and pad image
    image = letterbox(image, 960, stride=64, auto=True)[0]  # shape: (567, 960, 3)

    # Apply transforms
    image = transforms.ToTensor()(image)  # torch.Size([3, 567, 960])
    if torch.cuda.is_available():
        image = image.half().to(device)
    else:
        image = image.to(device)
    # Turn image into batch
    image = image.unsqueeze(0)  # torch.Size([1, 3, 567, 960])
    with torch.no_grad():
        output, _ = model(image)
    return output, image



def draw_keypoints(output, image):
    output = non_max_suppression_kpt(output,
                                     0.25,  # Confidence Threshold
                                     0.65,  # IoU Threshold
                                     nc=model.yaml['nc'],  # Number of Classes
                                     nkpt=model.yaml['nkpt'],  # Number of Keypoints
                                     kpt_label=True)
    with torch.no_grad():
        output = output_to_keypoint(output)
    nimg = image[0].permute(1, 2, 0) * 255
    nimg = nimg.cpu().numpy().astype(np.uint8)
    nimg = cv2.cvtColor(nimg, cv2.COLOR_RGB2BGR)
    for idx in range(output.shape[0]):
        plot_skeleton_kpts(nimg, output[idx, 7:].T, 3)

    return nimg


def process_video(video_file, exercise, webcam, draw_skeleton, recommendation):
    global processing_complete
    parity = str(uuid.uuid4())
    # parity = str(int(time.time() * 1000))
    # Define the output path for the processed video
    if video_file is not None:
        fname, fext = os.path.splitext(os.path.basename(video_file))
    else:
        fname, fext = f"{exercise}", ".mp4"
    output_filename = f"output_{fname}_{parity}_conv{fext}"
    output_path = os.path.join(app.config['UPLOAD_FOLDER'], output_filename)
    print("def pose estimation", video_file)
    

    if webcam == 'true':
        run_exercise(source='0', drawskeleton=draw_skeleton, recommendation = recommendation, parity=parity, exercise_name = exercise)
    else:
        run_exercise(source= video_file, drawskeleton=draw_skeleton, recommendation = recommendation, parity=parity, exercise_name = exercise ) 
    
        
    
    processing_complete = True
    # Return the output path of the processed video
    return output_path



def run_flask_server():
    
    socketio.run(app, debug=True, host='0.0.0.0')

# Set the initial value of the processing_complete flag
processing_complete = False
parity = ""

# -------------------------------------------------------------------------------
# Username : Password
# A dictionary to store valid username-password pairs (in a real application, this should be stored securely)
valid_credentials = {"admin": "admin"}

@app.route('/client/', methods=['GET', 'POST'])
def login():
    if request.method == 'GET':
        video_filename = 'loginbgvideo.mp4'
        return render_template('login.html', error=None, video_filename=video_filename)

    elif request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        if username in valid_credentials and password == valid_credentials[username]:
            # If the credentials are valid, serve another HTML page
            return render_template('index.html')
        else:
            # If the credentials are not valid, show an error message
            error_message = "Invalid username or password. Please try again."
            video_filename = 'loginbgvideo.mp4'
            return render_template('login.html', error=error_message, video_filename=video_filename)

@app.route('/client/static/<video_filename>', methods=['GET'])
def serve_login_video(video_filename):
    # Serve the static video file from the 'static/videos' folder
    return send_from_directory('client/static/videos', video_filename)

@app.route('/client/login.html')
def logout():
    video_filename = 'loginbgvideo.mp4'
    error_message = "You are logged out!"
    return render_template('login.html', error=error_message, video_filename=video_filename)

# -------------------------------------------------------------------------------


@app.route('/client/upload', methods=['POST'])
def upload():
    exercise = request.form['exercise']
    
    
    webcam = request.form['webcamstream']
    if request.form['drawSkeleton'] == 'yes':
        draw_skeleton = True
    else:
        draw_skeleton = False

    if request.form['recommend'] == 'yes':
        recommendation = True
    else:
        recommendation = False
    
    if webcam == 'false':
        file = request.files['video']

        if file:
            filename = secure_filename(file.filename)
            video_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(video_path)

            output_path = process_video(video_path, exercise, webcam, draw_skeleton, recommendation)

            print(output_path)
            output_filename = os.path.basename(output_path)
            output_url = f"/client/static/uploads/{output_filename}"

            return jsonify({'processed': True, 'video_url': output_url})
    else: 
        
        output_path = process_video(None, exercise, webcam, draw_skeleton, recommendation)
        output_filename = os.path.basename(output_path)
        output_url = f"/client/static/uploads/{output_filename}"

        return jsonify({'processed': True, 'video_url': output_url})

    return jsonify({'processed': False})

@app.route('/client/check_processing_status')
def check_processing_status():
    global processing_complete
    global output_filename

    if processing_complete:
        video_url = f"/client/static/uploads/{output_filename}"
        return jsonify({'status': 'complete', 'video_url': video_url})

    return jsonify({'status': 'incomplete'})

@app.route('/client/static/uploads/<filename>')
def serve_video(filename):
    video_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    print(video_path)
    return send_file(video_path, mimetype='video/mp4')

@app.route('/client/list_files/<directory>', methods=['GET'])
def list_file(path):
    result = subprocess.run("ls " + path, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    output = output = result.stdout.decode('utf-8')
    return output

def process_frame(image_data):
    # Decode base64 image data
    image_data = base64.b64decode(image_data.split(',')[1])

    # Convert the image data to a NumPy array
    nparr = np.frombuffer(image_data, np.uint8)

    # Decode the image using OpenCV
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # Perform your image processing here
    # For example, convert the frame to grayscale
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # # Resize the frame for display
    display_width = 320
    display_height = 240
    resized_frame = cv2.resize(gray_frame, (display_width, display_height))

    # Return the processed frame as base64-encoded image data
    _, encoded_frame = cv2.imencode('.jpg', resized_frame)
    return base64.b64encode(encoded_frame).decode('utf-8')

@socketio.on('video_feed')
def handle_video_feed(frameBuffer, exercise, draw_skeleton, recommendation):
    # processed_frame = process_frame(data)
    processed_frames = []
    for frame in frameBuffer:
        processed_frame = run_exercise(source='0', drawskeleton=draw_skeleton, recommendation = recommendation, parity=parity, exercise_name = exercise, img_data = frame)
        processed_frames.append(processed_frame)
        # You can send the processed frame back to the client or perform other actions as needed
    socketio.emit('processed_frame', processed_frames)

    
    

@app.route('/client/download_csv')
def download_csv():
    data = [
        {'Name': 'John', 'Age': 25, 'City': 'New York'},
        {'Name': 'Alice', 'Age': 30, 'City': 'San Francisco'},
        {'Name': 'Bob', 'Age': 22, 'City': 'Chicago'},
    ]

    # Specify the CSV file path
    csv_file_path = 'client/static/data.csv'

    return send_file(csv_file_path, as_attachment=True, download_name="example.csv")



    
if __name__ == '__main__':
    # Run the Flask server on a separate thread

    # Start the thread
    # flask_thread = threading.Thread(target=run_flask_server)
    # flask_thread.start()
    run_flask_server()


