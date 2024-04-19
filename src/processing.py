import cv2
import numpy as np


def apply_pixelation(img, combined_mask, width, height) -> np.ndarray:
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

def experimental_apply_pixelation(img, combined_mask, pixelate_level) -> np.ndarray:
    if "person" in combined_mask and img is not None and img.shape[0] > 0 and img.shape[1] > 0:
        person_mask = combined_mask["person"]["combined"]
        if np.any(person_mask > 0):

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
    
def draw_black_polygons(img, combined_mask, blackout_labels) -> np.ndarray:
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
