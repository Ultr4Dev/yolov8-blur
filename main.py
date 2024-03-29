import pyvirtualcam
import cv2
import numpy as np
from ultralytics import YOLO
from pathlib import Path
import time


def main():
    """
    Main function to process video stream by applying pixelation and labeling using YOLO model.
    """
    # Initialize variables
    inpaint_state = False
    predict_pose = False
    pixelate_state = True
    label_state = False

    # List of object labels recognized by YOLO model
    labels = ["bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
              "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
              "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
              "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
              "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
              "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
              "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard",
              "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
              "scissors", "teddy bear", "hair drier", "toothbrush"
              ]

    # Initialize YOLO model
    yolo_model = YOLO("models/yolov8x-seg.pt")
    yolo_model.cuda()

    # Initialize virtual camera
    with pyvirtualcam.Camera(width=1920, height=1080, fps=30, fmt=pyvirtualcam.PixelFormat.BGR) as cam:
        start_time = time.time()
        frame_count = 0

        # Predict on video stream
        results = yolo_model.predict(
            source=3, stream=True, conf=0.25, classes=[0, 62, 63], verbose=False)
        for r in results:
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

                if label == "person":
                    contour = c.masks.xy.pop().astype(np.int32).reshape(-1, 1, 2)
                    cv2.drawContours(combined_mask[label]["combined"], [
                                     contour], -1, (255, 255, 255), cv2.FILLED)
                    cv2.drawContours(combined_mask[label]["individual"][ci], [
                                     contour], -1, (255, 255, 255), cv2.FILLED)

                if label in labels:
                    contour = c.masks.xy.pop().astype(np.int32).reshape(-1, 1, 2)
                    cv2.drawContours(combined_mask[label]["combined"], [
                                     contour], -1, (255, 255, 255), cv2.FILLED)
                    cv2.drawContours(combined_mask[label]["individual"][ci], [
                                     contour], -1, (255, 255, 255), cv2.FILLED)

            # Pixelate all people in the image
            try:
                if pixelate_state and np.sum(combined_mask["person"]["combined"]) > 0:
                    person_mask = combined_mask["person"]["combined"]
                    w, h = img.shape[1], img.shape[0]
                    isolated_small = cv2.resize(
                        img, (w//50, h//50), interpolation=cv2.INTER_NEAREST)
                    isolated_large = cv2.resize(
                        isolated_small, (w, h), interpolation=cv2.INTER_NEAREST)
                    img[person_mask > 0] = isolated_large[person_mask > 0]

            except:
                pass

            # Draw black polygon for all objects like tv's, laptops, and cell phones in the combined mask
            for label in labels:
                try:
                    if np.sum(combined_mask[label]["combined"]) > 0:
                        img[combined_mask[label]["combined"] > 0] = 0
                    if label_state:
                        for ci, mask in combined_mask[label]["individual"].items():
                            if np.sum(mask) > 0:
                                img[mask > 0] = 0
                                M = cv2.moments(mask)
                                if M["m00"] != 0:
                                    cX = int(M["m10"] / M["m00"])
                                    cY = int(M["m01"] / M["m00"])
                                    cv2.putText(
                                        img, label, (cX, cY), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                except:
                    continue

            # Calculate FPS
            elapsed_time = time.time() - start_time
            fps = frame_count / elapsed_time

            # Add FPS counter to top left corner
            cv2.putText(img, f"FPS: {fps:.2f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            preview = cv2.resize(img, (256, 144))
            cv2.imshow("preview", preview)

            cam.send(img)
            cam.sleep_until_next_frame()
            cv2.waitKey(1)

    cam.close()


if __name__ == "__main__":
    main()
