# SafeVision - Intelligent Video Content Moderation

SafeVision is a production-grade desktop application for intelligent real-time video content moderation and automatic unsafe scene skipping. Built with Python, PyQt6, VLC backend, and advanced AI detection.

## Features

- **Premium UI/UX**: Dark theme, glassmorphism, smooth transitions.
- **Video Playback Engine**: VLC backend with custom controls, smooth seeking, hardware acceleration.
- **Real-Time Frame Extraction**: Asynchronous multithreaded frame extraction without blocking playback.
- **AI Unsafe Detection Engine**: Ready for integration with YOLOv8 or EfficientNet-B3.
- **Temporal Intelligence Engine**: Uses sliding window buffer and hysteresis logic to prevent flickering false positives.
- **Intelligent Skip Controller**: State machine that skips only unsafe portions seamlessly with cooldown logic.

## Code Architecture

The application follows clean architecture principles:
- `/ui` - PyQt6 UI components (MainWindow, Sidebars, PlayerPanel)
- `/player` - VLC Media player backend integration
- `/detection` - AI Detection engine running asynchronously
- `/temporal_engine` - Smoothing and anti-flicker logic
- `/skip_controller` - Smart skip logic and state machine

## Setup Instructions

1. **Prerequisites**
   - Python 3.10+
   - VLC Media Player installed on your system (required for python-vlc)
   
2. **Install Dependencies**
   Activate your virtual environment and install the required packages:
   ```bash
   source venv/bin/activate
   pip install PyQt6 python-vlc opencv-python numpy scenedetect ultralytics
   ```

3. **Run the Application**
   Start the application using the new entry point:
   ```bash
   python run_safevision.py
   ```

## Note on AI Model
The current `ai_detector.py` file contains placeholder logic to simulate detection scores. To use the real YOLOv8 or EfficientNet models, uncomment the inference lines inside `/detection/ai_detector.py` and ensure you have the appropriate weights downloaded.
