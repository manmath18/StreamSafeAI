import time

class SkipController:
    """
    State machine for intelligent skip controller.
    Prevents repeated skips using a cooldown logic.
    """
    def __init__(self, cooldown_ms=3000):
        self.cooldown_ms = cooldown_ms
        self.last_skip_time = 0
        self.state = "MONITORING"
        
    def handle_unsafe(self, current_ms):
        current_sys_time = time.time() * 1000
        
        # Check cooldown
        if current_sys_time - self.last_skip_time < self.cooldown_ms:
            return None # Ignore, in cooldown
            
        self.state = "SEEKING"
        self.last_skip_time = current_sys_time
        
        # Here we would normally use PySceneDetect to find the next scene boundary.
        # Example using scenedetect:
        # scene_manager = SceneManager()
        # scene_manager.add_detector(ContentDetector())
        # scene_manager.detect_scenes(video_path)
        # -> find next scene cut after current_ms
        
        # Dummy behavior: return target to skip (e.g. +5 seconds)
        target_ms = current_ms + 5000
        
        self.state = "BUFFER_RESET"
        self.state = "MONITORING"
        
        return target_ms
