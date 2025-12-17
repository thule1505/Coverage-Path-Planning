import numpy as np
import time

def get_intervals(row):
    intervals = []
    in_free = False
    start = None
    for c, val in enumerate(row):
        if val == 0 and not in_free:
            in_free = True
            start = c
        elif (val == 1 or c == len(row)-1) and in_free:
            end = c-1 if val == 1 else c
            intervals.append((start, end))
            in_free = False
    return intervals

def interval_overlap(i1, i2):
    return not (i1[1] < i2[0] or i2[1] < i1[0])

def boustrophedon_decomposition(occupancy):
    H, W = occupancy.shape
    cells = []               
    active_cells = {}        
    next_cell_id = 0
    prev_intervals = []
    
    # --- ĐIỀU CHỈNH CHIẾN THUẬT ---
    # Giảm TOLERANCE xuống để nhạy bén hơn với các góc tường
    TOLERANCE = 1 
    # Giảm ngưỡng gộp Cell để giữ lại các vùng chia tách ở góc khuất
    MIN_CELL_SIZE = 5 

    for r in range(H):
        row = occupancy[r]
        curr_intervals = get_intervals(row)
        new_active = {}

        for i, curr in enumerate(curr_intervals):
            matched_prev = [j for j, prev in enumerate(prev_intervals) if interval_overlap(curr, prev)]

            # CASE 1: Vùng mới (IN) hoặc CASE 3: Nhập vùng (MERGE)
            if len(matched_prev) != 1:
                cell_id = next_cell_id
                cells.append([])
                next_cell_id += 1
            
            # CASE 2: Tiếp tục hoặc bị tách (SPLIT)
            else:
                p_idx = matched_prev[0]
                prev_start, prev_end = prev_intervals[p_idx]
                curr_start, curr_end = curr
                
                # Đếm số lượng dải hiện tại chạm vào dải cũ này
                num_children = sum(1 for c_int in curr_intervals if interval_overlap(c_int, prev_intervals[p_idx]))
                
                # KIỂM TRA QUAN TRỌNG: 
                # Nếu ranh giới co vào (vật cản nhô ra), ta CẮT CELL NGAY để an toàn.
                # Nếu ranh giới mở rộng (vật cản kết thúc), ta cũng có thể cắt để tối ưu zigzag.
                boundary_shrunk = (curr_start > prev_start + TOLERANCE) or (curr_end < prev_end - TOLERANCE)
                boundary_expanded = (curr_start < prev_start - TOLERANCE) or (curr_end > prev_end + TOLERANCE)

                if num_children == 1 and not (boundary_shrunk or boundary_expanded):
                    cell_id = active_cells[p_idx] 
                else:
                    cell_id = next_cell_id 
                    cells.append([])
                    next_cell_id += 1

            new_active[i] = cell_id
            for c in range(curr[0], curr[1] + 1):
                cells[cell_id].append((r, c))

        active_cells = new_active
        prev_intervals = curr_intervals

    # --- HẬU XỬ LÝ: GỘP CELL QUÁ NHỎ (DƯỚI 15 PIXEL) ---
    # Điều này giúp ACO chạy cực nhanh mà không mất Coverage
    final_cells = []
    id_map_remap = {}
    
    # Lọc bỏ các cell rỗng hoặc quá bé (thường là nhiễu do đường cong)
    actual_id = 0
    for old_id, coords in enumerate(cells):
        if len(coords) > 10: # Ngưỡng tối thiểu để giữ 1 Cell
            final_cells.append(coords)
            id_map_remap[old_id] = actual_id
            actual_id += 1
        else:
            # Nếu cell quá nhỏ, gộp nó vào cell trước đó nếu có
            if actual_id > 0:
                final_cells[actual_id-1].extend(coords)
                id_map_remap[old_id] = actual_id - 1

    # Tạo bản đồ ID vùng cuối cùng
    cell_id_map = -np.ones_like(occupancy, dtype=int)
    for cid, coords in enumerate(final_cells):
        for (r, c) in coords:
            cell_id_map[r, c] = cid
            
    return final_cells, cell_id_map
