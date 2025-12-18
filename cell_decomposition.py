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
    TOLERANCE = 2
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

                if num_children == 1 and not boundary_shrunk:
                    cell_id = active_cells[p_idx]
                else:
                    # Chỉ tạo cell mới khi bị thu hẹp (shrunk), bị chia tách (split) hoặc nhập vào (merge)
                    cell_id = next_cell_id
                    cells.append([])
                    next_cell_id += 1
                # if num_children == 1 and not (boundary_shrunk or boundary_expanded):
                #     cell_id = active_cells[p_idx]
                # else:
                #     cell_id = next_cell_id
                #     cells.append([])
                #     next_cell_id += 1

            new_active[i] = cell_id
            for c in range(curr[0], curr[1] + 1):
                cells[cell_id].append((r, c))

        active_cells = new_active
        prev_intervals = curr_intervals

    # --- HẬU XỬ LÝ: GỘP CELL QUÁ NHỎ (DƯỚI 5 PIXEL) ---
    # Điều này giúp ACO chạy cực nhanh mà không mất Coverage
    final_cells = []
    for coords in cells:
        if len(coords) < 1: continue 
        coord_set = set(coords)
        while coord_set:
            start_node = next(iter(coord_set))
            q = [start_node]; coord_set.remove(start_node)
            current_component = [start_node]
            idx = 0
            while idx < len(q):
                curr = q[idx]; idx += 1
                for dr, dc in [(0,1),(0,-1),(1,0),(-1,0),(1,1),(1,-1),(-1,1),(-1,-1)]:
                    neighbor = (curr[0] + dr, curr[1] + dc)
                    if neighbor in coord_set:
                        coord_set.remove(neighbor)
                        q.append(neighbor); current_component.append(neighbor)
            
            # Không được dùng 'continue', mà phải giữ lại dù nhỏ để không tạo vùng trống
            if len(current_component) > 0:
                final_cells.append(current_component)

    # --- HẬU XỬ LÝ 2: VÉT PIXEL MỒ CÔI (QUAN TRỌNG NHẤT) ---
    cell_id_map = -np.ones_like(occupancy, dtype=int)
    for cid, coords in enumerate(final_cells):
        for (r, c) in coords:
            cell_id_map[r, c] = cid

    # Tìm những pixel free (0) mà chưa được gán ID nào (màu xanh nhạt bạn thấy)
    y_coords, x_coords = np.where((occupancy == 0) & (cell_id_map == -1))
    for r, c in zip(y_coords, x_coords):
        # Kiểm tra láng giềng xung quanh, thấy ID nào thì "nhập gia tùy tục" luôn
        found = False
        for dr in range(-2, 3):
            for dc in range(-2, 3):
                nr, nc = r + dr, c + dc
                if 0 <= nr < H and 0 <= nc < W and cell_id_map[nr, nc] != -1:
                    cell_id_map[r, c] = cell_id_map[nr, nc]
                    final_cells[cell_id_map[nr, nc]].append((r, c))
                    found = True; break
            if found: break

    return final_cells, cell_id_map
