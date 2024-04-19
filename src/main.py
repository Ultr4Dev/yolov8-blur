import json
import time
import pyvirtualcam
import cv2
import numpy as np
import torch
from ultralytics import YOLO
import yaml

from configurator import VideoProcessingConfig, configuratorGUI
from processing import draw_black_polygons, experimental_apply_pixelation

from list_cameras import getCams

getCams()

getCams = None

# Initialize GUI window
cv2.namedWindow("Virtual Camera", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Virtual Camera", 640, 480)

def get_camera_pixel_size(camera:int) -> tuple[int, int]:
    # Get camera pixel size
    camera = cv2.VideoCapture(camera)
    
    width = int(camera.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT))

    camera.release()
    print(f"Camera pixel size: {width}x{height}")
    return width, height


# Image embedding function
def embed_image(img) -> np.ndarray:
    # TODO Implement image embedding for similiarity search
    raise NotImplementedError("Image embedding not implemented yet.")
    return img

# Main function

# Main function
def main(config:VideoProcessingConfig) -> None:
    print(config)
    with open("./src/misc/coco-classes.json") as f:
        labels = json.load(f)
    # Remap the keys for the labels to int not string
    labels = {int(key): value for key, value in labels.items()}
    width, height = get_camera_pixel_size(config.source_index)
    fps = 0
    try:
        yolo_model = YOLO(config.yolo_model_path)
    except FileNotFoundError:
        print("Error: YOLO model file not found.")
        return
    except Exception as e:
        print(f"Error loading YOLO model: {e}")
        return
    # Convert
    if "person" in config.blackout_labels:
        # Remove person from blackout labels
        config.blackout_labels.remove("person")
    temp_labels = config.blackout_labels + ["person"]
    classes = [key for key, value in labels.items() if value in temp_labels]
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    yolo_model.to(device)
    yolo_model.compile()
    print(f"Using {'CUDA' if use_cuda else 'CPU'} for inference.")
    
    frame_counter = 0
    start_time = time.time()
    
    with pyvirtualcam.Camera(width=1920, height=1080, fps=30, fmt=pyvirtualcam.PixelFormat.BGR, device="cam1") as cam:
        results = yolo_model.predict(retina_masks=False, source=config.source_index, stream=True,
                                     conf=config.model_confidence, verbose=False, half=True, imgsz=config.img_size, batch=5, vid_stride=4,
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
            if config.pixelate_state:
                #img = apply_pixelation(img, combined_mask, width, height)
                try:
                    img = experimental_apply_pixelation(img, combined_mask, config.pixelate_level)
                except Exception as e:
                    print(f"Error applying pixelation: {e}")
                    continue
            img = draw_black_polygons(img, combined_mask, config.blackout_labels)

            if frame_counter % 1 == 0:
                cam.send(img)
                cam.sleep_until_next_frame()
            frame_counter += 1
            if frame_counter % 5 == 0:
                if width == 0 or height == 0:
                    width, height = 1920, 1080
                cv2.imshow("Virtual Camera", cv2.resize(
                    img, (int(width // 2), int(height // 2))))

            if frame_counter % 30 == 0:
                end_time = time.time()
                fps = frame_counter / (end_time - start_time)
                print(f"FPS: {fps:.2f}")
                frame_counter = 0
                start_time = time.time()
            #cam.sleep_until_next_frame()
            if cv2.waitKey(1) == ord('q'):
                break
            if cv2.waitKey(1) == ord('c'):
                config = configuratorGUI()
    cv2.destroyAllWindows()
    cam.close()

if __name__ == "__main__":
    main(configuratorGUI())