"""
cad_ogm.py
-------------------
Reusable module for converting CAD images (PNG) to 2D binary occupancy grid numpy arrays (OGM).

Usage:
    from cad_ogm import cad_to_ogm
    ogm = cad_to_ogm('path/to/image.png', grid_size=(200,200))

Main function:
    cad_to_ogm(input_path, ...):
        Input: path to PNG image
        Output: 2D binary numpy array (OGM)

All processing steps are encapsulated for reuse in other projects.
"""

"""
cad_ogm.py
-------------------
Reusable module for converting CAD images (PNG) to 2D binary occupancy grid numpy arrays (OGM).

Usage:
    from cad_ogm import cad_to_ogm
    ogm = cad_to_ogm('path/to/image.png', grid_size=(200,200))

Main function:
    cad_to_ogm(input_path, ...):
        Input: path to PNG image
        Output: 2D binary numpy array (OGM)

All processing steps are encapsulated for reuse in other projects.
"""
import cv2
import numpy as np


def load_image_gray(path):
    """
    Load a grayscale image from the given path.

    Step by step:
    1. Read the image in grayscale mode using OpenCV.
    2. Check if the image was loaded successfully; raise a FileNotFoundError if not.
    3. Return the grayscale image array.
    """
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Input image not found: {path}")
    return img


def preprocess_classical(img):
    """
    Process CAD images with thin lines and faint watermarks.

    Step by step:
    1. Apply thresholding to convert the image to binary: CAD images have white background (255) and black lines (0), with gray watermarks (~230-240). Use THRESH_BINARY_INV.
    2. Remove small noise: Use connected components to filter out small areas below a minimum size.
    3. Dilate the lines: Since the image is downscaled (e.g., from 4000px to 400px, 10x reduction), lines need to be thickened to at least 10px in the original to survive downscaling. Dilate twice with a 5x5 kernel.
    4. Return the dilated binary image (0-255).
    """
    # 1. Threshold:
    # CAD images have white background (255), black lines (0). Watermarks are gray (~230-240).
    # We use THRESH_BINARY_INV.

    img = cv2.GaussianBlur(img, (5, 5), 0)
    _, binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)

    # 2. Noise removal
    # num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
    #     binary, connectivity=8
    # )
    # min_size = 50
    # binary_clean = np.zeros_like(binary)
    # for i in range(1, num_labels):
    #     if stats[i, cv2.CC_STAT_AREA] >= min_size:
    #         binary_clean[labels == i] = 255

    kernel = np.ones((3,3), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)

    # # 3. IMPORTANT: Thicken lines (Dilate)
    # kernel_dilate = np.ones((5, 5), np.uint8)
    # # Dilate twice to make lines truly thick
    # dilated = cv2.dilate(binary, kernel_dilate, iterations=2)

    return binary  # Return 0-255 image

def raster_to_grid(occ_mask_255, grid_size=(400, 400)):
    """
    Resize the occupancy mask to grid size using smart interpolation.

    Step by step:
    1. Resize the image using INTER_AREA interpolation (best for downscaling to avoid losing lines).
    2. The resized result is a grayscale image due to pixel averaging.
    3. Re-binarize: If any pixel in the cell has some black (>10), consider it an obstacle.
    4. Return the binary grid as uint8.
    """
    # 1. Resize using INTER_AREA (best for downscaling)
    # Result is grayscale due to pixel averaging
    resized_gray = cv2.resize(occ_mask_255, grid_size, interpolation=cv2.INTER_AREA)

    # 2. Re-binarize
    # Since resized_gray is grayscale (lines may fade to 50, 100...),
    # Define: If cell has any black (>10), it's an obstacle.
    return (resized_gray > 10).astype(np.uint8)


def fill_closed_regions(grid):
    """Fills closed free space regions in the grid with walls."""
    h, w = grid.shape

    # 1. Pad the image with 1 pixel of value 0 (Free) around the edges
    # This ensures (0,0) is always free space and connected to the outer region
    grid_padded = np.pad(grid, pad_width=1, mode="constant", constant_values=0)

    # 2. Prepare mask for floodFill (OpenCV requires mask 2 pixels larger than image)
    h_pad, w_pad = grid_padded.shape
    mask = np.zeros((h_pad + 2, w_pad + 2), np.uint8)

    grid_filled = grid_padded.copy()

    # 3. FloodFill from (0, 0) with temporary value 2
    # Since padded, (0,0) is 0, so floodFill spreads to all outer free space
    cv2.floodFill(grid_filled, mask, (0, 0), 2)

    # 4. Points still 0 are closed regions (not connected to boundary)
    closed_mask = grid_filled == 0

    # 5. Set closed regions to Obstacle (1)
    grid_padded[closed_mask] = 1

    # 6. Remove padding to return to original size
    result = grid_padded[1:-1, 1:-1]

    return result


def cad_to_ogm(input_path, grid_size=(400, 400), fill_closed_regions_flag=False):
    """
    Convert CAD image to 2D binary occupancy grid map (OGM).

    Step by step:
    1. Load the image in grayscale.
    2. Preprocess to get a high-resolution wall mask (binary, thickened lines).
    3. Resize to grid size while preserving lines.
    4. Optionally fill closed free regions with walls.
    5. Return the binary OGM grid.
    """
    # 1. Load
    img = load_image_gray(input_path)

    # 2. Preprocess (Output: binary image with thick lines)
    wall_mask_highres = preprocess_classical(img)

    # 3. Resize to Grid (Important: preserve lines)
    ogm_grid = raster_to_grid(wall_mask_highres, grid_size)

    if fill_closed_regions_flag:
        ogm_grid = fill_closed_regions(ogm_grid)

    return ogm_grid
