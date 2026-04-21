from collections import deque

class TemporalSmoother:
    """
    Implements temporal smoothing using a sliding window buffer,
    weighted average confidence, and consecutive unsafe threshold logic.
    Prevents flickering decisions.
    """
    def __init__(self, window_size=12, threshold=0.72):
        self.window_size = window_size
        self.threshold = threshold
        self.buffer = deque(maxlen=window_size)
        self.consecutive_unsafe = 0
        self.state_unsafe = False

    def process(self, current_score):
        self.buffer.append(current_score)
        avg_score = sum(self.buffer) / len(self.buffer)
        
        # Hysteresis switching and consecutive logic
        if avg_score > self.threshold:
            self.consecutive_unsafe += 1
            if self.consecutive_unsafe >= 5: # 5 consecutive risky frames
                self.state_unsafe = True
        else:
            self.consecutive_unsafe = 0
            if avg_score < 0.45: # Safe threshold
                self.state_unsafe = False
                
        return self.state_unsafe, avg_score
