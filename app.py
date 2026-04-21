#!/usr/bin/env python3
import os
import cv2
import time
import json
import threading
from flask import Flask, request, render_template, Response, jsonify, send_file
from werkzeug.utils import secure_filename
import numpy as np

from live import LiveNudeDetector
from video_trimmer import VideoTrimmer

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs('video_outputs', exist_ok=True)

stream_states = {}
stream_states_lock = threading.Lock()
analysis_cache = {}
trimmers = {}

print("Initializing LiveNudeDetector...")
detector = LiveNudeDetector()
print("LiveNudeDetector ready.")

def is_kiss_scene(detections):
    face_boxes = []
    for d in detections:
        if d["class"] in ["FACE_FEMALE", "FACE_MALE"] and d.get("score", 0.0) > 0.4:
            face_boxes.append(d["box"])
    if len(face_boxes) >= 2:
        for i in range(len(face_boxes)):
            for j in range(i+1, len(face_boxes)):
                x1, y1, w1, h1 = face_boxes[i]
                x2, y2, w2, h2 = face_boxes[j]
                c1x, c1y = x1 + w1/2, y1 + h1/2
                c2x, c2y = x2 + w2/2, y2 + h2/2
                dist = np.sqrt((c1x - c2x)**2 + (c1y - c2y)**2)
                avg_width = (w1 + w2) / 2
                if dist < (avg_width * 1.0): 
                    return True
    return False

def generate_frames(video_path, filename, mode):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_delay = max(0.01, 1.0 / fps)
    
    with stream_states_lock:
        stream_states[filename] = {"seek_pct": None, "current_pct": 0.0, "paused": False}
    
    safe_segments = None
    if filename in analysis_cache:
        safe_segments = [(s['start'], s['end']) for s in analysis_cache[filename] if s['type'] == 'safe']
        
    while True:
        state = stream_states.get(filename, {})
        if not state:
            break
            
        target_seek = state.get("seek_pct")
        is_paused = state.get("paused", False)
        
        if target_seek is not None:
            target_pct = max(0.0, min(target_seek, 100.0))
            frame_idx = int((target_pct / 100.0) * total_frames)
            if frame_idx >= total_frames: 
                frame_idx = total_frames - 1
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            with stream_states_lock:
                stream_states[filename]["seek_pct"] = None
        
        if is_paused:
            time.sleep(0.1)
            continue
            
        start_time = time.time()
        
        current_frame = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        current_time_sec = current_frame / fps
        
        # True seamless skip using pre-scanned segments
        if mode == 'skip' and safe_segments:
            is_in_safe_segment = False
            for s_start, s_end in safe_segments:
                if s_start <= current_time_sec <= s_end:
                    is_in_safe_segment = True
                    break
            
            if not is_in_safe_segment:
                next_safe_time = None
                for s_start, s_end in safe_segments:
                    if s_start > current_time_sec:
                        next_safe_time = s_start
                        break
                
                if next_safe_time is not None:
                    next_frame_idx = int(next_safe_time * fps)
                    cap.set(cv2.CAP_PROP_POS_FRAMES, next_frame_idx)
                    current_frame = next_frame_idx
                else:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, total_frames - 1)
                    continue
        
        ret, frame = cap.read()
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue
            
        if total_frames > 0:
            with stream_states_lock:
                stream_states[filename]["current_pct"] = (current_frame / total_frames) * 100.0
        
        # Real-time inference
        detections = detector.detect_frame(frame)
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

        is_kiss = is_kiss_scene(detections)

        # Apply fallback if skip mode is on but we don't have pre-scan
        if mode == 'skip' and (has_sensitive_content or is_kiss) and not safe_segments:
            display_frame = np.zeros_like(frame)
            text = "KISS SCENE SKIPPED" if is_kiss else "SENSITIVE CONTENT SKIPPED"
            font = cv2.FONT_HERSHEY_SIMPLEX
            text_size = cv2.getTextSize(text, font, 1.0, 2)[0]
            text_x = (display_frame.shape[1] - text_size[0]) // 2
            text_y = (display_frame.shape[0] + text_size[1]) // 2
            cv2.putText(display_frame, text, (text_x, text_y), font, 1.0, (0, 0, 255), 2)
        else:
            display_frame = detector.apply_censoring(frame, detections)
            
        ret, buffer = cv2.imencode('.jpg', display_frame)
        if ret:
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
                   
        elapsed = time.time() - start_time
        sleep_time = frame_delay - elapsed
        if sleep_time > 0:
            time.sleep(sleep_time)

    cap.release()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    if file:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        stream_states[filename] = {"seek_pct": None, "current_pct": 0.0, "paused": False}
        return jsonify({"success": True, "filename": filename})

@app.route('/video_feed/<filename>')
def video_feed(filename):
    mode = request.args.get('mode', 'blur')
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(filename))
    if not os.path.exists(filepath):
        return "File not found", 404
    return Response(generate_frames(filepath, filename, mode), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/status/<filename>')
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

@app.route('/analyze/<filename>', methods=['POST'])
def analyze(filename):
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(filename))
    if not os.path.exists(filepath):
        return jsonify({"error": "File not found"}), 404
    
    if filename not in trimmers:
        trimmers[filename] = VideoTrimmer(filepath, frame_skip=2)
    
    trimmer = trimmers[filename]
    
    def run_analysis():
        res = trimmer.analyze()
        analysis_cache[filename] = res["segments"]
        
    if trimmer.status == "idle":
        threading.Thread(target=run_analysis).start()
        
    return jsonify({"success": True, "message": "Analysis started"})

@app.route('/analyze_status/<filename>')
def analyze_status(filename):
    if filename not in trimmers:
        return jsonify({"error": "Not found"}), 404
    trimmer = trimmers[filename]
    prog = trimmer.get_progress()
    
    data = {
        "status": prog["status"],
        "progress": prog["progress"],
        "message": prog["message"]
    }
    if prog["status"] == "done":
        data["segments"] = analysis_cache.get(filename, [])
        
    return jsonify(data)

@app.route('/trim/<filename>', methods=['POST'])
def trim(filename):
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(filename))
    if not os.path.exists(filepath):
        return jsonify({"error": "File not found"}), 404
        
    if filename not in trimmers:
        return jsonify({"error": "Analyze first"}), 400
        
    trimmer = trimmers[filename]
    
    if trimmer.status == "done" and not trimmer.trim_output:
        def run_trim():
            try:
                trimmer.trim()
            except Exception as e:
                trimmer.status = "error"
                trimmer.status_message = str(e)
        threading.Thread(target=run_trim).start()
        
    return jsonify({"success": True, "message": "Trim started"})

@app.route('/download/<filename>')
def download(filename):
    if filename not in trimmers:
        return "Not found", 404
    trimmer = trimmers[filename]
    if trimmer.trim_output and os.path.exists(trimmer.trim_output):
        return send_file(trimmer.trim_output, as_attachment=True)
    return "File not ready", 404

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
