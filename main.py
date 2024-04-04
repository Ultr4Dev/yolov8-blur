import time
import pyvirtualcam
import cv2
import numpy as np
import torch
from ultralytics import YOLO

"""
CONFIGURABLE PARAMETERS:
"""
# Configurable parameters
pixelate_state = True  # Whether to apply pixelation
blackout_labels = ["tv", "laptop", "cell phone"]  # Labels to blackout
yolo_model_path = "models/yolov9c-seg.pt"  # YOLO model path with segmentation will download if not found
source_index = 5  # Index of the video source (0 for default camera)
img_size = 480  # Input image size for YOLO model
    
"""
END OF CONFIGURABLE PARAMETERS
"""
# Function to convert class names to class IDs

height, width = (0, 0)
labels = ["person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
          "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
          "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
          "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
          "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
          "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
          "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard",
          "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
          "scissors", "teddy bear", "hair drier", "toothbrush"]
class_name_to_id = {name: idx for idx, name in enumerate(labels)}


def class_name_list_to_class_id_list(class_names: list[str]):

    return [class_name_to_id[class_name] for class_name in class_names]

def apply_pixelation(img, combined_mask, width, height):
    """
    Apply pixelation to the areas specified by the combined mask.

    Args:
        img (numpy.ndarray): Input image.
        combined_mask (dict): Combined mask for specified labels.
        width (int): Width of the image.
        height (int): Height of the image.

    Returns:
        numpy.ndarray: Image with pixelation applied.
    """

    if "person" in combined_mask and np.any(combined_mask["person"]["combined"] > 0):
        person_mask = combined_mask["person"]["combined"]

        # Vectorized resizing for efficiency
        isolated_small = cv2.resize(
            img, (width // 50, height // 50), interpolation=cv2.INTER_NEAREST)
        isolated_large = cv2.resize(
            isolated_small, (width, height), interpolation=cv2.INTER_NEAREST)

        # Pixel modification using direct mask indexing
        img[person_mask > 0] = isolated_large[person_mask > 0]

    return img

def draw_black_polygons(img, combined_mask, blackout_labels):
    """
    Draw black polygons on the image for specified labels.

    Args:
        img (numpy.ndarray): Input image.
        combined_mask (dict): Combined mask for specified labels.
        blackout_labels (list): Labels to blackout.

    Returns:
        numpy.ndarray: Image with black polygons drawn.
    """
    if not blackout_labels:
        return img
    # Draw black polygons for all blackout labels at once
    for label in blackout_labels:
        try:
            if np.sum(combined_mask[label]["combined"]) > 0:
                img[combined_mask[label]["combined"] > 0] = 0

        except KeyError:  # Handle cases where a label may not be detected
            continue

    return img

# Image embedding function

def embed_image(img):
    # TODO Implement image embedding for similiarity search
    return img

# Main function


def main():
    fps = 0
    # Initialize YOLO model
    try:
        yolo_model = YOLO(yolo_model_path)
    except FileNotFoundError:
        print("Error: YOLO model file not found.")
        return
    except Exception as e:
        print(f"Error loading YOLO model: {e}")
        return

    # Error handling for CUDA availability
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    yolo_model.to(device)
    print(f"Using {'CUDA' if use_cuda else 'CPU'} for inference.")
    
    frame_counter = 0
    start_time = time.time()
    # Initialize virtual camera
    with pyvirtualcam.Camera(width=1920, height=1080, fps=30, fmt=pyvirtualcam.PixelFormat.BGR) as cam:
        global height, width

        # Predict on video stream
        results = yolo_model.predict(retina_masks=False, iou=0.5, source=source_index, stream=True,
                                     conf=0.50, verbose=False, half=True, imgsz=img_size, batch=10, vid_stride=5,
                                     classes=class_name_list_to_class_id_list(["person", "tv", "laptop"]))
        for r in results:
            if width == 0 and height == 0:
                height, width = r.orig_img.shape[:2]
            img = r.orig_img.copy()

            # Initialize a combined mask for all persons
            combined_mask = {}
            for ci, c in enumerate(r):
                label = c.names[int(c.boxes[0].cls.cpu().tolist()[0])]

                if label not in combined_mask:
                    combined_mask[label] = {"combined": np.zeros_like(
                        img[:, :, 0])}

                contour = c.masks.xy.pop().astype(np.int32).reshape(-1, 1, 2)
                cv2.drawContours(combined_mask[label]["combined"], [
                    contour], -1, (255, 255, 255), cv2.FILLED)

            # Apply pixelation if enabled
            if pixelate_state:
                img = apply_pixelation(img, combined_mask, width, height)

            # Draw black polygons for specified labels
            img = draw_black_polygons(
                img, combined_mask, blackout_labels)

            if frame_counter % 2 == 0:
            # half resolution of the original image
                cv2.imshow("Virtual Camera", cv2.resize(
                    img, (int(width / 3), int(height / 3))))

            cam.send(img)
            cam.sleep_until_next_frame()
            frame_counter += 1

            # Calculate FPS every second (or every N frames, e.g., 30)
            if frame_counter % 60 == 0:
                end_time = time.time()
                fps = frame_counter / (end_time - start_time)
                print(f"FPS: {fps:.2f}")
                # Reset the frame counter and start time
                frame_counter = 0
                start_time = time.time()
            if cv2.waitKey(1) == ord('q'):
                break
    cv2.destroyAllWindows()
    cam.close()

if __name__ == "__main__":
    main()