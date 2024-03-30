import pyvirtualcam
import cv2
import numpy as np
from ultralytics import YOLO
from pathlib import Path
import time

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

# Function to apply pixelation with optimized 'person' mask check


def apply_pixelation(img, combined_mask, width, height):

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

# Function to draw black polygons with combined iterations


def draw_black_polygons(img, combined_mask, blackout_labels):
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

# Main function


def main():
    # Initialize variables
    pixelate_state = True
    blackout_labels = ["tv", "laptop", "cell phone"]
    label_state = False
    # Initialize YOLO model
    yolo_model = YOLO("models/yolov8x-seg.pt")
    yolo_model.cuda()

    # Initialize virtual camera
    with pyvirtualcam.Camera(width=1920, height=1080, fps=30, fmt=pyvirtualcam.PixelFormat.BGR) as cam:
        start_time = time.time()
        frame_count = 0
        global height, width

        # Predict on video stream
        results = yolo_model.predict(retina_masks=False, iou=0.9, source=5, stream=True,
                                     conf=0.25, verbose=False,
                                     classes=class_name_list_to_class_id_list(["person", "tv", "laptop"]))
        for r in results:
            if width == 0 and height == 0:
                height, width = r.orig_img.shape[:2]
            frame_count += 1
            img = np.copy(r.orig_img)

            # Initialize a combined mask for all persons
            combined_mask = {}
            for ci, c in enumerate(r):
                label = c.names[int(c.boxes[0].cls.cpu().tolist()[0])]

                if label not in combined_mask:
                    combined_mask[label] = {"combined": np.zeros_like(
                        img[:, :, 0]), "individual": {}}

                if ci not in combined_mask[label]["individual"]:
                    combined_mask[label]["individual"][ci] = np.zeros_like(
                        img[:, :, 0])

                contour = c.masks.xy.pop().astype(np.int32).reshape(-1, 1, 2)
                cv2.drawContours(combined_mask[label]["combined"], [
                    contour], -1, (255, 255, 255), cv2.FILLED)
                cv2.drawContours(combined_mask[label]["individual"][ci], [
                    contour], -1, (255, 255, 255), cv2.FILLED)

            # Apply pixelation if enabled
            if pixelate_state:
                img = apply_pixelation(img, combined_mask, width, height)

            # Draw black polygons for specified labels
            img = draw_black_polygons(
                img, combined_mask, blackout_labels)

            # Calculate FPS
            elapsed_time = time.time() - start_time
            fps = frame_count / elapsed_time

            # Add FPS counter to top left corner
            cv2.putText(img, f"FPS: {fps:.2f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # half resolution of the original image
            cv2.imshow("Virtual Camera", cv2.resize(
                img, (int(img.shape[1] / 2), int(img.shape[0] / 2))))

            cam.send(img)
            cam.sleep_until_next_frame()
            cv2.waitKey(1)
    cam.close()


if __name__ == "__main__":
    main()
