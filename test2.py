import pyvirtualcam
import cv2
import numpy as np
from ultralytics import YOLO
from pathlib import Path


def list_devices():
    devices = []
    for i in range(10):
        try:
            cap = cv2.VideoCapture(i)
            if not cap.isOpened():
                continue
            ret, frame = cap.read()
            if not ret:
                continue
            cv2.imshow(f"Camera {i}", frame)
            devices.append(i)
            cap.release()
            cv2.waitKey(1)
        except:
            continue

    return devices


def select_device():
    devices = list_devices()
    print("Select camera device:")
    device = int(input("Available devices: " +
                 ", ".join(map(str, devices)) + "\n"))
    cv2.destroyAllWindows()
    return device


# Initialize YOLO model
yolo_model = YOLO("models/yolov8x-seg.pt")
yolo_model.cuda()
predict_pose = False
pixelate_state = True

# Initialize virtual camera
with pyvirtualcam.Camera(width=1920, height=1080, fps=30, device="OBS Virtual Camera", fmt=pyvirtualcam.PixelFormat.BGR) as cam:
    # Predict on video stream
    results = yolo_model.predict(source=0, stream=True, conf=0.30)
    for r in results:
        img = np.copy(r.orig_img)
        # Initialize a combined mask for all persons
        combined_mask = {}

        for ci, c in enumerate(r):
            label = c.names[int(c.boxes[0].cls.cpu().tolist()[0])]
            if label not in combined_mask:
                combined_mask[label] = np.zeros_like(img[:, :, 0])
            if label == "person":
                # Create contour mask
                contour = c.masks.xy.pop().astype(np.int32).reshape(-1, 1, 2)
                cv2.drawContours(
                    combined_mask[label], [contour], -1, (255, 255, 255), cv2.FILLED)

            if label in ["tv", "laptop", "cell phone"]:
                # Create contour mask
                contour = c.masks.xy.pop().astype(np.int32).reshape(-1, 1, 2)
                cv2.drawContours(
                    combined_mask[label], [contour], -1, (255, 255, 255), cv2.FILLED)
        # Pixelate all people in the image
        if pixelate_state and np.sum(combined_mask["person"]) > 0:
            isolated = cv2.bitwise_and(img, img, mask=combined_mask["person"])

            # Combine all person masks from the combined_mask["person"]
            person_mask = combined_mask["person"]
            isolated = cv2.bitwise_and(img, img, mask=person_mask)
            w, h = isolated.shape[1], isolated.shape[0]
            # Resize -> Pixelate -> Resize back
            isolated_small = cv2.resize(
                isolated, (w//25, h//25), interpolation=cv2.INTER_NEAREST)
            isolated_large = cv2.resize(
                isolated_small, (w, h), interpolation=cv2.INTER_NEAREST)

            img[combined_mask["person"] >
                0] = isolated_large[combined_mask["person"] > 0]

        # Draw black polygon for all tv's, laptops, and cell phones in the combined mask
        for label in ["tv", "laptop", "cell phone"]:
            try:
                if np.sum(combined_mask[label]) > 0:
                    img[combined_mask[label] > 0] = 0
            except:
                continue
        # Resize the image to 720p
        preview = cv2.resize(img, (1920, 1080))
        cv2.imshow("preview", preview)

        cam.send(img)
        cam.sleep_until_next_frame()

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cam.close()
