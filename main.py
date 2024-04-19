from src.configurator import configuratorGUI
from src.videoprocessor import VideoProcessor

if __name__ == "__main__":
    VideoProcessor(configuratorGUI()).run()