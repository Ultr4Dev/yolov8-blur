"""
This script lists all the cameras connected to the computer. It has to be used as a standalone script and not imported as a module in another script.
This is because it breaks a bunch of stuff and i dont feel like fixing it.
"""

import cv2
def getCams2():
    cams = []
    for index in range(0, 10):
            cap = cv2.VideoCapture(index)
            if cap.isOpened():
                print(f"Device {index}: {cap.get(3)}x{cap.get(4)}")
                cams.append((index, cap.get(3), cap.get(4)))
                index += 1
                cap.release()
            else:
                pass
    return cams
        
        
        

if __name__ == "__main__":
    cams = getCams2()
    print(cams)