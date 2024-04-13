import time
import pyvirtualcam
import cv2
import numpy as np
import torch
from ultralytics import YOLO
import yaml

# Configurable parameters
pixelate_state = True  # Whether to apply pixelation to people
#pixelate_level = 50  # Pixelation level (higher values for more pixelation)
blackout_labels = ["tv", "laptop", "cell phone"]  # Labels to blackout (must be in coco-classes.yaml)
yolo_model_path = "models/yolov9c-seg.pt"  # YOLO model path with segmentation
source_index = 5  # Index of the video source (0 for default camera)
img_size: int = int(640/2)  # Input image size for YOLO model
model_confidence = 0.20  # Confidence threshold for YOLO model

# Initialize GUI window
cv2.namedWindow("Virtual Camera", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Virtual Camera", 640, 480)

# Create trackbars for configurable parameters
cv2.createTrackbar("Pixelate", "Virtual Camera", 1, 1, lambda x: None)
# Write trackbar initial values
cv2.setTrackbarPos("Pixelate", "Virtual Camera", int(pixelate_state))



# Function to convert class names to class IDs
# load from coco-classes.yaml
with open("coco-classes.yaml", "r") as file:
    """
    format of coco-classes.yaml:
    names:
        id(int): name(str)
        
    """
    labels = yaml.load(file, Loader=yaml.FullLoader)["names"]
    print(labels)

def get_camera_pixel_size(camera:int):
    # Get camera pixel size
    camera = cv2.VideoCapture(camera)
    width = int(camera.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
    camera.release()
    
    return width, height

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
        
        # Cut out the person from the original image
        img[person_mask > 0] = 0
        
        # Pixel modification using direct mask indexing
        img[person_mask > 0] = isolated_large[person_mask > 0]

    return img

def experimental_apply_pixelation(img, combined_mask):
    if "person" in combined_mask and img is not None and img.shape[0] > 0 and img.shape[1] > 0:
        person_mask = combined_mask["person"]["combined"]
        if np.any(person_mask > 0):
            pixelate_level = 85

            # Prepare a blank canvas to pixelate on
            pixelated_img = img.copy()

            # Apply pixelation only within the mask
            rows, cols = img.shape[:2]
            for i in range(0, rows, pixelate_level):
                for j in range(0, cols, pixelate_level):
                    # Check if the current block intersects with the person mask
                    block_mask = person_mask[i:i + pixelate_level, j:j + pixelate_level]
                    if np.any(block_mask):
                        # Calculate average color within the block where mask is positive
                        block = img[i:i + pixelate_level, j:j + pixelate_level]
                        avg_color = np.mean(block[block_mask > 0], axis=0)
                        # Apply the average color only to the pixels within the mask in this block
                        pixelated_img[i:i + pixelate_level, j:j + pixelate_level][block_mask > 0] = avg_color

            # Apply the pixelated image only where the mask is positive
            img[person_mask > 0] = pixelated_img[person_mask > 0]
            
            return img
        else:
            return img
    else:
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
    width, height = get_camera_pixel_size(source_index)
    fps = 0
    try:
        yolo_model = YOLO(yolo_model_path)
    except FileNotFoundError:
        print("Error: YOLO model file not found.")
        return
    except Exception as e:
        print(f"Error loading YOLO model: {e}")
        return
    # Convert 
    temp_labels = blackout_labels + ["person"]
    classes = [key for key, value in labels.items() if value in temp_labels]
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    yolo_model.to(device)
    yolo_model.compile()
    print(f"Using {'CUDA' if use_cuda else 'CPU'} for inference.")
    
    frame_counter = 0
    start_time = time.time()
    
    with pyvirtualcam.Camera(width=width, height=height, fps=30, fmt=pyvirtualcam.PixelFormat.BGR, device="cam1") as cam:
        results = yolo_model.predict(retina_masks=False, source=source_index, stream=True,
                                     conf=model_confidence, verbose=False, half=True, imgsz=img_size, batch=5, vid_stride=4,
                                     classes=classes)
        for r in results:
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
                #img = apply_pixelation(img, combined_mask, width, height)
                try:
                    img = experimental_apply_pixelation(img, combined_mask)
                except Exception as e:
                    print(f"Error applying pixelation: {e}")
                    continue
            img = draw_black_polygons(img, combined_mask, blackout_labels)

            if frame_counter % 1 == 0:
                cam.send(img)
                cam.sleep_until_next_frame()
            frame_counter += 1
            if frame_counter % 5 == 0:
                cv2.imshow("Virtual Camera", cv2.resize(
                    img, (int(width / 4), int(height / 4))))

            if frame_counter % 30 == 0:
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