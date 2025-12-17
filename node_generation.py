import numpy as np
import math

def node_generation(cells, occupancy):
    """
    Generates node features and metadata for each decomposed cell.
    
    This module extracts essential information like centroids, boundaries, and 
    optimal cleaning orientations for each area found in the cell decomposition phase.

    Parameters:
    ----------
    cells : list of lists
        A list where each element is a list of (row, col) coordinates belonging to a cell.
    occupancy : np.ndarray
        The 2D Binary Occupancy Grid Map (OGM) used for boundary checking.

    Returns:
    -------
    dict
        A dictionary keyed by cell_id containing geometric and topological metadata.
    """

    processed = {}
    h, w = occupancy.shape

    for cid, coords in enumerate(cells):
        # Skip empty cells to prevent processing errors
        if len(coords) == 0: continue 
        
        # --- 1. Coordinate Preprocessing ---
        # Ensure the coordinate array is always 2D (N, 2) to avoid dimension errors in calculations
        coords_arr = np.array(coords)
        if coords_arr.ndim == 1:
             # Nếu mảng bị phẳng, đưa nó về dạng (N, 2)
             coords_arr = coords_arr.reshape(-1, 2)
        
        # Convert to set for O(1) lookup during boundary detection
        coord_set = set(map(tuple, coords_arr)) 

        # --- 2. Centroid Calculation ---
        # Calculate the geometric mean of all points in the cell
        centroid_f = np.mean(coords_arr, axis=0)
        
        # Safety check: Handle scalar return cases and convert to integer pixel coordinates
        if centroid_f.ndim == 0: # Trường hợp xấu nhất nếu mean trả về 1 số
            centroid = tuple(coords_arr[0].astype(int))
        else:
            centroid = (int(centroid_f[0]), int(centroid_f[1]))

        # find the closest valid point within the cell to serve as the functional centroid.
        if centroid not in coord_set:
            dists = np.sum((coords_arr - centroid_f)**2, axis=1)
            centroid = tuple(coords_arr[np.argmin(dists)].astype(int))
            
        # --- 3. Boundary Detection ---
        # A point is a boundary point if at least one of its 4-neighbors is outside the cell.
        boundary = []
        for (r, c) in coord_set:
            for dr, dc in [(-1,0), (1,0), (0,-1), (0,1)]:
                nr, nc = r + dr, c + dc
                if not (0 <= nr < h and 0 <= nc < w) or (nr, nc) not in coord_set:
                    boundary.append((r, c))
                    break

        if not boundary: boundary = [centroid]

        # Sort boundary points angularly (circularly) around the centroid using atan2
        # This helps in creating a clean perimeter for visualization or path calculations.
        boundary_sorted = sorted(boundary, key=lambda p: math.atan2(p[0]-centroid[0], p[1]-centroid[1]))

        # --- 4. Candidate Node Generation ---
        # Select up to 8 representative points along the boundary. 
        # These are used as entrance/exit candidates for inter-cell path planning.
        B = len(boundary_sorted)
        indices = np.linspace(0, B - 1, min(4, B), dtype=int)
        candidates = [boundary_sorted[i] for i in indices]

        min_r, min_c = np.min(coords_arr, axis=0)
        max_r, max_c = np.max(coords_arr, axis=0)
        height, width = max_r - min_r + 1, max_c - min_c + 1
        orientation = "vertical" if height > width else "horizontal"
        
        processed[cid] = {
            "cell_id": cid,
            "coordinates": [tuple(p) for p in coords_arr],
            "centroid": centroid,
            "boundary": boundary_sorted,
            "candidates": candidates,
            "orientation": orientation,
            "span": ((int(min_r), int(min_c)), (int(max_r), int(max_c)))
        }
    return processed
