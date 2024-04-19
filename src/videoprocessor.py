import json
import time
import cv2
import numpy as np
import torch
from ultralytics import YOLO
import pyvirtualcam

from src.configurator import VideoProcessingConfig, configuratorGUI
from src.processing import draw_black_polygons, experimental_apply_pixelation

class VideoProcessor:
    def __init__(self, config:VideoProcessingConfig) -> None:
        self.config:VideoProcessingConfig = config
        self.labels: dict = self.load_labels("./src/misc/coco-classes.json")
        self.width, self.height = self.get_camera_pixel_size(self.config.source_index)
        self.yolo_model: YOLO = self.load_yolo_model()
        self.setup_virtual_cam()

    def load_labels(self, filepath) -> dict:
        with open(filepath) as f:
            labels = json.load(f)
        return {int(key): value for key, value in labels.items()}

    def get_camera_pixel_size(self, camera_index) -> tuple:
        camera = cv2.VideoCapture(camera_index)
        width = int(camera.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
        camera.release()
        return width, height

    def load_yolo_model(self) -> YOLO:
        try:
            yolo_model = YOLO(self.config.yolo_model_path)
            yolo_model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
            yolo_model.compile()
            return yolo_model
        except Exception as e:
            print(f"Error loading YOLO model: {e}")
            return None

    def setup_virtual_cam(self) -> None:
        cv2.namedWindow("Virtual Camera", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Virtual Camera", 640, 480)

    def run(self) -> None:
        with pyvirtualcam.Camera(width=1920, height=1080, fps=30, fmt=pyvirtualcam.PixelFormat.BGR, device="cam1") as cam:
            with open("./src/misc/coco-classes.json") as f:
                labels = json.load(f)
            
            labels = {int(key): value for key, value in labels.items()}
            temp_labels = self.config.blackout_labels + ["person"]
            classes = [key for key, value in labels.items() if value in temp_labels]
            results = self.yolo_model.predict(retina_masks=False, source=self.config.source_index, stream=True,
                                        conf=self.config.model_confidence, verbose=False, half=True, imgsz=self.config.img_size, batch=5, vid_stride=4,
                                        classes=classes)
            for r in results:
                img = r.orig_img.copy()
                if img.shape[0] == 0 or img.shape[1] == 0:
                    continue
                combined_mask = {}
                for ci, c in enumerate(r):
                    label = c.names[int(c.boxes[0].cls.cpu().tolist()[0])]

                    if label not in combined_mask:
                        combined_mask[label] = {"combined": np.zeros_like(
                            img[:, :, 0])}

                    contour = c.masks.xy.pop().astype(np.int32).reshape(-1, 1, 2)
                    try:
                        cv2.drawContours(combined_mask[label]["combined"], [
                            contour], -1, (255, 255, 255), cv2.FILLED)
                    except Exception as e:
                        print(f"Error drawing contour: {e}")
                        pass
                if self.config.pixelate_state:
                    #img = apply_pixelation(img, combined_mask, width, height)
                    try:
                        img = experimental_apply_pixelation(img, combined_mask, self.config.pixelate_level)
                    except Exception as e:
                        print(f"Error applying pixelation: {e}")
                        continue
                img = draw_black_polygons(img, combined_mask, self.config.blackout_labels)


                cam.send(img)
                cam.sleep_until_next_frame()

                if self.width == 0 or self.height == 0:
                    self.width, self.height = 1920, 1080
                cv2.imshow("Virtual Camera", cv2.resize(
                    img, (int(self.width // 2), int(self.height // 2))))

                if cv2.waitKey(1) == ord('q'):
                    break
                if cv2.waitKey(1) == ord('c'):
                    self.config = configuratorGUI()
        cv2.destroyAllWindows()
        cam.close()