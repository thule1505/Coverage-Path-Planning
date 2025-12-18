import numpy as np
import heapq
from reduced_graph import astar, reconstruct_path

# --- 1. CORE LOGIC: RECTILINEAR PATH (L-TURN) ---

def find_rectilinear_path(start, end, safe_coords):
    """
    Tìm đường đi vuông góc (L-turn) an toàn tuyệt đối.
    Đảm bảo robot di chuyển dọc/ngang 90 độ thay vì đi chéo.
    """
    r_start, c_start = int(start[0]), int(start[1])
    r_end, c_end = int(end[0]), int(end[1])

    if (r_start, c_start) == (r_end, c_end): return []

    def is_line_clear(p1, p2):
        """Kiểm tra mọi ô trên đoạn thẳng nối p1 và p2 có thuộc safe_coords không."""
        r1, c1 = int(p1[0]), int(p1[1])
        r2, c2 = int(p2[0]), int(p2[1])
        rr = range(min(r1, r2), max(r1, r2) + 1)
        cc = range(min(c1, c2), max(c1, c2) + 1)
        for r in rr:
            for c in cc:
                if (r, c) not in safe_coords:
                    return False
        return True

    # 1: DỌC TRƯỚC (V), NGANG SAU (H) - Ưu tiên để thấy robot đi xuống biên
    p_turn_v = (r_end, c_start)
    if is_line_clear((r_start, c_start), p_turn_v) and is_line_clear(p_turn_v, (r_end, c_end)):
        return [p_turn_v, (r_end, c_end)]

    # 2: NGANG TRƯỚC (H), DỌC SAU (V)
    p_turn_h = (r_start, c_end)
    if is_line_clear((r_start, c_start), p_turn_h) and is_line_clear(p_turn_h, (r_end, c_end)):
        return [p_turn_h, (r_end, c_end)]

    return [(r_end, c_end)]

# --- 2. SNAKE PATH GENERATORS ---
def generate_horizontal_snake(coords, current_pos):
    """Quét ngang với logic Snake Path và L-Turn chuyển hàng."""
    row_dict = {}
    for r, c in coords:
        if r not in row_dict: row_dict[r] = []
        row_dict[r].append(c)

    sorted_rows = sorted(row_dict.keys())
    # Chọn hướng bắt đầu gần robot nhất
    if abs(current_pos[0] - sorted_rows[-1]) < abs(current_pos[0] - sorted_rows[0]):
        sorted_rows = sorted_rows[::-1]

    path, internal_turns = [], []
    first_row_cols = sorted(row_dict[sorted_rows[0]])
    reverse = abs(current_pos[1] - first_row_cols[-1]) < abs(current_pos[1] - first_row_cols[0])
    safe_coords = set(coords)

    for r in sorted_rows:
        cols_in_row = sorted(row_dict[r], reverse=reverse)
        p_entry = (r, cols_in_row[0])

        if path:
            p_prev_exit = path[-1]
            if p_prev_exit[0] != r: # Chuyển hàng
                rect_path = find_rectilinear_path(p_prev_exit, p_entry, safe_coords)
                if rect_path:
                    internal_turns.append(rect_path[0])
                    for p in rect_path:
                        if p != path[-1]: path.append(p)

        for c in cols_in_row:
            p_curr = (r, c)
            if not path or p_curr != path[-1]: path.append(p_curr)
        reverse = not reverse

    return path, internal_turns

def generate_vertical_snake(coords, current_pos):
    """Quét dọc với logic Snake Path và L-Turn chuyển cột."""
    col_dict = {}
    for r, c in coords:
        if c not in col_dict: col_dict[c] = []
        col_dict[c].append(r)

    sorted_cols = sorted(col_dict.keys())
    if abs(current_pos[1] - sorted_cols[-1]) < abs(current_pos[1] - sorted_cols[0]):
        sorted_cols = sorted_cols[::-1]

    path, internal_turns = [], []
    first_col_rows = sorted(col_dict[sorted_cols[0]])
    reverse = abs(current_pos[0] - first_col_rows[-1]) < abs(current_pos[0] - first_col_rows[0])
    safe_coords = set(coords)

    for c in sorted_cols:
        rows_in_col = sorted(col_dict[c], reverse=reverse)
        p_entry = (rows_in_col[0], c)

        if path:
            p_prev_exit = path[-1]
            if p_prev_exit[1] != c: # Chuyển cột
                rect_path = find_rectilinear_path(p_prev_exit, p_entry, safe_coords)
                if rect_path:
                    internal_turns.append(rect_path[0])
                    for p in rect_path:
                        if p != path[-1]: path.append(p)

        for r in rows_in_col:
            p_curr = (r, c)
            if not path or p_curr != path[-1]: path.append(p_curr)
        reverse = not reverse

    return path, internal_turns

def generate_zigzag_in_cell(cell_data, current_pos):
    """Tự động chọn hướng quét ưu tiên theo chiều dài cell."""
    coords = cell_data['coordinates']
    if not coords: return [], []
    rows, cols = [p[0] for p in coords], [p[1] for p in coords]
    height, width = max(rows) - min(rows) + 1, max(cols) - min(cols) + 1

    return generate_horizontal_snake(coords, current_pos) if width >= height \
           else generate_vertical_snake(coords, current_pos)

# --- 3. MASTER PLANNER ---
def full_coverage_planner(processed_cells, best_sequence, occupancy_grid, charging_station):
    """Hợp nhất lộ trình toàn cục, tối ưu chuyển tiếp giữa các Cell bằng A*."""
    master_path, detailed_segments = [], []
    current_pos = tuple(map(int, charging_station))

    # Grid an toàn cho Transition L-Turn
    safe_grid_coords = set(zip(*np.where(occupancy_grid == 0)))

    for cid in best_sequence:
        cell_data = processed_cells[cid]
        cell_zigzag, internal_turns = generate_zigzag_in_cell(cell_data, current_pos)

        if not cell_zigzag: continue

        # --- Transition: A* logic ---
        entry_point = tuple(map(int, cell_zigzag[0]))
        _, transition_path = astar(occupancy_grid, current_pos, entry_point)

        if transition_path:
            transition_turns = []
            # Tạo đường rẽ vuông góc cho đoạn nối từ vị trí cũ vào đường A* nếu cần
            # (Thường A* đã khá tối ưu, nhưng ta chèn thêm find_rectilinear_path để đồng bộ 90 độ)
            p_start_astar = tuple(map(int, transition_path[0]))
            rect_trans = find_rectilinear_path(current_pos, p_start_astar, safe_grid_coords)

            full_trans = [current_pos] + rect_trans + list(transition_path[1:])
            if rect_trans: transition_turns.append(rect_trans[0])

            detailed_segments.append({
                'type': 'transition',
                'points': full_trans,
                'cell_id': cid,
                'turns': transition_turns
            })
            master_path.extend(full_trans)

        # --- Zigzag: Nội bộ Cell ---
        detailed_segments.append({
            'type': 'zigzag',
            'points': cell_zigzag,
            'entry': entry_point,
            'exit': cell_zigzag[-1],
            'turns': internal_turns
        })
        master_path.extend(cell_zigzag)
        current_pos = tuple(map(int, cell_zigzag[-1]))

    return master_path, detailed_segments, current_pos
