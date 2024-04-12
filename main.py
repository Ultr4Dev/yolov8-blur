import time
import pyvirtualcam
import cv2
import numpy as np
import torch
from ultralytics import YOLO

# Configurable parameters
pixelate_state = True  # Whether to apply pixelation to people
pixelate_level = 50  # Pixelation level (higher values for more pixelation)
blackout_labels = ["tv", "laptop", "cell phone"]  # Labels to blackout
yolo_model_path = "models/yolov9c-seg.pt"  # YOLO model path with segmentation
source_index = 5  # Index of the video source (0 for default camera)
img_size = 640  # Input image size for YOLO model
model_confidence = 0.20  # Confidence threshold for YOLO model

# Initialize GUI window
cv2.namedWindow("Virtual Camera", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Virtual Camera", 640, 480)

# Create trackbars for configurable parameters
cv2.createTrackbar("Pixelate", "Virtual Camera", 1, 1, lambda x: None)
# Write trackbar initial values
cv2.setTrackbarPos("Pixelate", "Virtual Camera", int(pixelate_state))
# Pixelation level trackbar
cv2.createTrackbar("Pixelate level", "Virtual Camera", 1, 100, lambda x: None)
cv2.setTrackbarPos("Pixelate level", "Virtual Camera", pixelate_level)


# Function to convert class names to class IDs
labels = ["person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
          "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
          "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
          "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
          "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
          "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
          "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard",
          "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
          "scissors", "teddy bear", "hair drier", "toothbrush"]

    


height, width = (0, 0)
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
        pixelate_level = cv2.getTrackbarPos("Pixelate level", "Virtual Camera")
        if pixelate_level == 0:
            pixelate_level = 1
        # Vectorized resizing for efficiency
        isolated_small = cv2.resize(
            img, (width // pixelate_level, height // pixelate_level), interpolation=cv2.INTER_NEAREST)
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


# Main function
def main():
    fps = 0
    try:
        yolo_model = YOLO(yolo_model_path)
    except FileNotFoundError:
        print("Error: YOLO model file not found.")
        return
    except Exception as e:
        print(f"Error loading YOLO model: {e}")
        return

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    yolo_model.to(device)
    print(f"Using {'CUDA' if use_cuda else 'CPU'} for inference.")
    
    frame_counter = 0
    start_time = time.time()
    
    with pyvirtualcam.Camera(width=1920, height=1080, fps=30, fmt=pyvirtualcam.PixelFormat.BGR) as cam:
        global height, width
        results = yolo_model.predict(retina_masks=False, iou=0.5, source=source_index, stream=True,
                                     conf=model_confidence, verbose=False, half=True, imgsz=img_size, batch=10, vid_stride=5,
                                     classes=class_name_list_to_class_id_list(["person"] + blackout_labels))
        for r in results:
            if width == 0 and height == 0:
                height, width = r.orig_img.shape[:2]
            img = r.orig_img.copy()

            combined_mask = {}
            for ci, c in enumerate(r):
                label = c.names[int(c.boxes[0].cls.cpu().tolist()[0])]

                if label not in combined_mask:
                    combined_mask[label] = {"combined": np.zeros_like(
                        img[:, :, 0])}

                contour = c.masks.xy.pop().astype(np.int32).reshape(-1, 1, 2)
                cv2.drawContours(combined_mask[label]["combined"], [
                    contour], -1, (255, 255, 255), cv2.FILLED)

            if cv2.getTrackbarPos("Pixelate", "Virtual Camera") == 1:
                img = apply_pixelation(img, combined_mask, width, height)

            img = draw_black_polygons(img, combined_mask, blackout_labels)

            if frame_counter % 1 == 0:
                cam.send(img)
            frame_counter += 1
            if frame_counter % 5 == 0:
                cv2.imshow("Virtual Camera", cv2.resize(
                    img, (int(width / 4), int(height / 4))))

            if frame_counter % 60 == 0:
                end_time = time.time()
                fps = frame_counter / (end_time - start_time)
                print(f"FPS: {fps:.2f}")
                frame_counter = 0
                start_time = time.time()
            #cam.sleep_until_next_frame()
            if cv2.waitKey(1) == ord('q'):
                break
    cv2.destroyAllWindows()
    cam.close()

if __name__ == "__main__":
    main()