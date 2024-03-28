import pyvirtualcam
import time
import cv2
import numpy as np
from ultralytics import YOLO, SAM
from pathlib import Path

# Initialize YOLO model
yolo_model = YOLO("yolov8x-seg.pt")
predict_pose = True
pixelate_state = True

if predict_pose:
    pose_model = YOLO("yolov8x-pose.pt")


# Initialize virtual camera
with pyvirtualcam.Camera(width=1920, height=1080, fps=10, device="OBS Virtual Camera", fmt=pyvirtualcam.PixelFormat.BGR) as cam:
    yolo_model.cuda()

    # Predict on video stream
    results = yolo_model.predict(
        source="0", stream=True, conf=0.30, classes=[0, 62])
    for r in results:

        img = np.copy(r.orig_img)

        for ci, c in enumerate(r):
            # Get label from the mask

            label = c.boxes[0].cls.cpu().tolist()[0]
            # Convert label to string
            label = c.names[int(label)]
            if label != "person":
                print(label)
                continue
            b_mask = np.zeros(img.shape[:2], np.uint8)

            # Create contour mask
            contour = c.masks.xy.pop().astype(np.int32).reshape(-1, 1, 2)
            _ = cv2.drawContours(
                b_mask, [contour], -1, (255, 255, 255), cv2.FILLED)
            isolated = np.dstack([img, b_mask])

            if predict_pose:
                pose_results = pose_model(isolated[:, :, :3])

            # Pixelate isolated object
            if pixelate_state:
                try:
                    w, h = isolated[:, :, 2].shape[::-1]
                    isolated = isolated[:, :, :3]

                    # Resize the isolated object
                    isolated = cv2.resize(
                        isolated, (w//25, h//25), interpolation=cv2.INTER_NEAREST)

                    # Resize the isolated object back to original size
                    isolated = cv2.resize(
                        isolated, (1920, 1080), interpolation=cv2.INTER_NEAREST)

                except:
                    continue

            if predict_pose:
                for pose in pose_results:
                    x = pose.plot(
                        img=isolated[:, :, :3], show=False, conf=False, boxes=False)
                    isolated = x

            # Replace object with blurred version
            img[b_mask > 0] = isolated[b_mask > 0]

        cam.send(img)
        cam.sleep_until_next_frame()

        # Close camera if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            cam.close()
            break
