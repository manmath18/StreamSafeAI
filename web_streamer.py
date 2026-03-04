#!/usr/bin/env python3
"""
Web Media Streamer
A Flask web app that allows uploading a video and streams it via MJPEG with runtime blurring/skipping.
Includes a timeline interface to move forward/backward and advanced kiss detection.
"""

from flask import Flask, request, render_template_string, Response, redirect, jsonify
import cv2
import os
import time
from werkzeug.utils import secure_filename
import numpy as np
import threading

# Import the pre-existing robust LiveNudeDetector
from live import LiveNudeDetector

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Central state for all active video streams to support timeline & playback
stream_states = {}
stream_states_lock = threading.Lock()

# Initialize detector exactly once when the server starts
print("Initializing the LiveNudeDetector...")
detector = LiveNudeDetector()
print("Detector successfully initialized. Web App Ready!")

HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>SafeVision Web Streamer</title>
    <style>
        body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; text-align: center; margin: 40px; background-color: #121212; color: #ffffff; }
        .container { background: #1e1e1e; padding: 30px; border-radius: 8px; box-shadow: 0 4px 15px rgba(0,0,0,0.5); max-width: 800px; margin: auto; }
        input[type="file"] { margin: 20px 0; padding: 10px; background: #2c2c2c; border-radius: 5px; color: white; width: 80%;}
        select, button { padding: 12px; font-size: 16px; margin: 10px; border-radius: 5px; border: none; }
        select { background-color: #2c2c2c; color: white; width: 80%; padding: 15px;}
        button { background-color: #00ADB5; color: white; cursor: pointer; font-weight: bold; width: 80%; padding: 15px; margin-top:20px;}
        button:hover { background-color: #007F86; }
        .cancel-btn { background-color: #e84545; width:auto; padding: 10px 20px;}
        .cancel-btn:hover { background-color: #c0392b; }
        img.video-stream { width: 100%; max-width: 800px; border: 3px solid #00ADB5; border-radius: 8px; margin-top: 20px; }
        h2 { color: #00ADB5; margin-bottom: 5px; font-size: 28px;}
        .subtitle { color: #aaaaaa; margin-bottom: 30px;}
        .form-group { margin-bottom: 20px;}
        
        .controls-container {
            display: flex;
            align-items: center;
            justify-content: space-between;
            margin-top: 20px;
            background: #2a2a2a;
            padding: 15px 25px;
            border-radius: 8px;
        }
        
        #timeline {
            flex-grow: 1;
            margin: 0 20px;
            cursor: pointer;
            height: 6px;
            -webkit-appearance: none;
            background: #555;
            border-radius: 4px;
        }
        
        #timeline::-webkit-slider-thumb {
            -webkit-appearance: none;
            height: 18px;
            width: 18px;
            border-radius: 50%;
            background: #00ADB5;
            cursor: pointer;
            box-shadow: 0 0 5px rgba(0,173,181, 0.5);
        }
        
        #play-pause-btn {
            width: 100px;
            margin: 0;
            padding: 10px;
            background-color: #393e46;
        }
        #play-pause-btn:hover { background-color: #505763; }
        
        #time-display {
            font-weight: bold;
            color: #00ADB5;
            min-width: 60px;
        }
        
    </style>
</head>
<body>
    <div class="container">
        <h2>SafeVision Web Streamer</h2>
        {% if not filename %}
            <p class="subtitle">Upload a video below. It will be streamed directly to your browser securely with timeline tracking.</p>
            <form action="/" method="post" enctype="multipart/form-data">
                <div class="form-group">
                    <input type="file" name="file" required accept="video/*">
                </div>
                <div class="form-group">
                    <select name="mode">
                        <option value="blur">Dynamic Blur Mode (Blurs Explicit Regions)</option>
                        <option value="skip">Full Frame Skip (Hides Entire Frame)</option>
                    </select>
                </div>
                <button type="submit">Upload & Stream</button>
            </form>
        {% else %}
            <h3 style="color:#FFF;">Streaming: <span style="color:#00ADB5;">{{ filename }}</span></h3>
            
            <img class="video-stream" src="/video_feed/{{ filename }}?mode={{ mode }}">
            
            <div class="controls-container">
                <button id="play-pause-btn" onclick="togglePause()">Pause</button>
                <input type="range" id="timeline" min="0" max="100" value="0" step="0.1" 
                       onchange="seekVideo(this.value)" oninput="pauseSync()">
                <span id="time-display">0.0%</span>
            </div>
            
            <form action="/" method="get">
                <button type="submit" class="cancel-btn" style="margin-top:30px;">Upload a Different Video</button>
            </form>
            
            <script>
                const filename = "{{ filename }}";
                let isDragging = false;
                
                function pauseSync() {
                    isDragging = true;
                    document.getElementById('time-display').innerText = parseFloat(document.getElementById('timeline').value).toFixed(1) + '%';
                }
                
                function seekVideo(val) {
                    fetch(`/seek/${filename}`, {
                        method: 'POST',
                        headers: {'Content-Type': 'application/json'},
                        body: JSON.stringify({ pct: parseFloat(val) })
                    }).then(() => {
                        // Unblock the timeline updates
                        isDragging = false; 
                    });
                }
                
                function togglePause() {
                    fetch(`/toggle_pause/${filename}`, { method: 'POST' })
                        .then(r => r.json())
                        .then(data => {
                            document.getElementById('play-pause-btn').innerText = data.paused ? "Play" : "Pause";
                        });
                }
                
                // Poll server for status update
                setInterval(() => {
                    if (isDragging) return; // don't jitter while dragging
                    fetch(`/status/${filename}`)
                        .then(r => r.json())
                        .then(data => {
                            if (!isDragging) {
                                document.getElementById('timeline').value = data.pct;
                                document.getElementById('time-display').innerText = data.pct.toFixed(1) + '%';
                                document.getElementById('play-pause-btn').innerText = data.paused ? "Play" : "Pause";
                            }
                        })
                        .catch(err => console.log('Stream ended or inactive'));
                }, 1000);
            </script>
        {% endif %}
    </div>
</body>
</html>
"""

def is_kiss_scene(detections):
    """
    Advanced Logic: Checks if two faces are close enough to represent a kiss scene.
    Filters "FACE_FEMALE" and "FACE_MALE" classes and computes distance thresholds.
    """
    face_boxes = []
    for d in detections:
        label = d["class"]
        score = d.get("score", 0.0)
        # Assuming model emits these face labels with reasonable confidence
        if label in ["FACE_FEMALE", "FACE_MALE"] and score > 0.4:
            face_boxes.append(d["box"])
            
    # At least two faces must be detected to verify face proximity
    if len(face_boxes) >= 2:
        for i in range(len(face_boxes)):
            for j in range(i+1, len(face_boxes)):
                x1, y1, w1, h1 = face_boxes[i]
                x2, y2, w2, h2 = face_boxes[j]
                
                # Center coords
                c1x, c1y = x1 + w1/2, y1 + h1/2
                c2x, c2y = x2 + w2/2, y2 + h2/2
                
                # Distance
                dist = np.sqrt((c1x - c2x)**2 + (c1y - c2y)**2)
                
                # Proximity Check (Less than average face width means practically overlaying/kissing)
                avg_width = (w1 + w2) / 2
                
                if dist < (avg_width * 1.0): 
                    return True
    return False

def generate_frames(video_path, filename, mode):
    cap = cv2.VideoCapture(video_path)
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps != fps or fps == 0:
        fps = 30.0
    frame_delay = max(0.01, 1.0 / fps)
    
    # Init or Reset Stream State tracking
    with stream_states_lock:
        stream_states[filename] = {"seek_pct": None, "current_pct": 0.0, "paused": False}
    
    while True:
        # 1. Fetch State Flags
        state = stream_states.get(filename, {})
        if not state:
            break
            
        target_seek = state.get("seek_pct")
        is_paused = state.get("paused", False)
        
        # 2. Check Playback Position Modification (Scrubbing / Seeking)
        if target_seek is not None:
            # Safely clamp
            target_pct = max(0.0, min(target_seek, 100.0))
            frame_idx = int((target_pct / 100.0) * total_frames)
            # Make sure it's valid
            if frame_idx >= total_frames: 
                frame_idx = total_frames - 1
                
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            
            with stream_states_lock:
                stream_states[filename]["seek_pct"] = None
        
        # 3. Handle pause natively by holding loop
        if is_paused:
            time.sleep(0.1)
            continue
            
        start_time = time.time()
        
        ret, frame = cap.read()
        if not ret:
            # Video finished, automatically restart
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue
            
        # Update current timeline tracker
        current_frame = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        if total_frames > 0:
            with stream_states_lock:
                stream_states[filename]["current_pct"] = (current_frame / total_frames) * 100.0
        
        # 4. Filter AI Detections
        detections = detector.detect_frame(frame)
        
        # Check standard severity violations
        has_sensitive_content = False
        for d in detections:
            label = d["class"]
            severity = d.get("severity", "MODERATE")
            score = d.get("score", 0.0)
            
            if detector.should_apply_blur(label):
                if severity in ['CRITICAL', 'HIGH']:
                    has_sensitive_content = True
                    break
                elif severity == 'MODERATE' and score >= 0.4:
                    has_sensitive_content = True
                    break
                elif severity == 'LOW' and score >= 0.6:
                    has_sensitive_content = True
                    break

        # Check Kiss Violation Explicit Rule
        is_kiss = is_kiss_scene(detections)

        # 5. Paint Frame
        # Whether user picks 'skip' mode, OR a kiss is detected (always full-skip kisses as per rules)
        if (mode == 'skip' and has_sensitive_content) or is_kiss:
            display_frame = np.zeros_like(frame)
            text = "KISS SCENE SKIPPED" if is_kiss else "SENSITIVE CONTENT SKIPPED"
            font = cv2.FONT_HERSHEY_SIMPLEX
            text_size = cv2.getTextSize(text, font, 1.0, 2)[0]
            text_x = (display_frame.shape[1] - text_size[0]) // 2
            text_y = (display_frame.shape[0] + text_size[1]) // 2
            cv2.putText(display_frame, text, (text_x, text_y), font, 1.0, (0, 0, 255), 2)
            cv2.putText(display_frame, f"Timeline Frame: {current_frame}", (10, 30), font, 0.7, (255, 255, 255), 2)
        else:
            display_frame = detector.apply_censoring(frame, detections)
            
        # Encode as JPEG
        ret, buffer = cv2.imencode('.jpg', display_frame)
        if ret:
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
                   
        # Maintain playback speed so it doesn't zip through the video too incredibly fast when buffering
        elapsed = time.time() - start_time
        sleep_time = frame_delay - elapsed
        if sleep_time > 0:
            time.sleep(sleep_time)

    cap.release()

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
            
        file = request.files['file']
        mode = request.form.get('mode', 'blur')
        
        if file.filename == '':
            return redirect(request.url)
            
        if file:
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            # Reset state for fresh stream
            stream_states[filename] = {"seek_pct": None, "current_pct": 0.0, "paused": False}
            
            return render_template_string(HTML_TEMPLATE, filename=filename, mode=mode)
            
    return render_template_string(HTML_TEMPLATE, filename=None, mode=None)

@app.route('/video_feed/<filename>')
def video_feed(filename):
    mode = request.args.get('mode', 'blur')
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(filename))
    if not os.path.exists(filepath):
        return "File not found", 404
        
    return Response(generate_frames(filepath, filename, mode), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/status/<filename>', methods=['GET'])
def get_status(filename):
    state = stream_states.get(filename, {})
    return jsonify({
        "pct": state.get("current_pct", 0.0),
        "paused": state.get("paused", False)
    })

@app.route('/seek/<filename>', methods=['POST'])
def seek_video(filename):
    data = request.json
    if "pct" in data and filename in stream_states:
        with stream_states_lock:
            stream_states[filename]["seek_pct"] = data["pct"]
    return jsonify({"success": True})

@app.route('/toggle_pause/<filename>', methods=['POST'])
def toggle_pause(filename):
    if filename in stream_states:
        with stream_states_lock:
            new_state = not stream_states[filename]["paused"]
            stream_states[filename]["paused"] = new_state
        return jsonify({"paused": new_state})
    return jsonify({"paused": False})

if __name__ == '__main__':
    # Run the Flask app openly on localhost
    print("=========================================================")
    print("Starting Web Media Streamer. Go to http://127.0.0.1:5000")
    print("Press CTRL+C in this terminal to stop the server.")
    print("=========================================================")
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
