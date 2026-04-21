import sys

# Optional try-except for VLC. If python-vlc is not installed, we fallback or raise error.
try:
    import vlc
except ImportError:
    vlc = None

class VLCBackend:
    def __init__(self):
        if vlc is None:
            print("Warning: python-vlc not installed. VLC backend will not work.")
            self.instance = None
            self.player = None
            return
            
        self.instance = vlc.Instance('--no-xlib', '--quiet')
        self.player = self.instance.media_player_new()

    def set_window(self, widget):
        if not self.player: return
        # Set the window id where to render VLC's video output
        if sys.platform.startswith('linux'):
            self.player.set_xwindow(int(widget.winId()))
        elif sys.platform == "win32":
            self.player.set_hwnd(int(widget.winId()))
        elif sys.platform == "darwin":
            self.player.set_nsobject(int(widget.winId()))

    def play_media(self, filepath):
        if not self.player: return
        media = self.instance.media_new(filepath)
        self.player.set_media(media)
        self.player.play()

    def play(self):
        if self.player: self.player.play()

    def pause(self):
        if self.player: self.player.pause()

    def stop(self):
        if self.player: self.player.stop()

    def is_playing(self):
        return self.player.is_playing() if self.player else False

    def get_time(self):
        return self.player.get_time() if self.player else 0

    def set_time(self, ms):
        if self.player: self.player.set_time(ms)

    def get_length(self):
        return self.player.get_length() if self.player else 0

    def get_position(self):
        return self.player.get_position() if self.player else 0

    def set_position(self, pos):
        if self.player: self.player.set_position(pos)
