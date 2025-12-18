import heapq
import numpy as np
from sklearn.neighbors import KDTree

def reconstruct_path(came_map, start, goal):
    path = []
    current = goal
    while current != start:
        if current not in came_map: return None
        path.append(current)
        current = came_map[current]
    path.append(start)
    return path[::-1]

def astar(grid, start, goal):
    H, W = grid.shape
    start, goal = tuple(map(int, start)), tuple(map(int, goal))

    def find_valid(p):
        if grid[p[0], p[1]] == 0: return p
        # Nếu kẹt tường, tìm trong bán kính 2px điểm trống gần nhất
        for r in range(p[0]-2, p[0]+3):
            for c in range(p[1]-2, p[1]+3):
                if 0 <= r < H and 0 <= c < W and grid[r, c] == 0:
                    return (r, c)
        return p

    start, goal = find_valid(start), find_valid(goal)    
    if grid[start] == 1 or grid[goal] == 1: return np.inf, None
    # Manhattan heuristic cho 4 hướng
    def h(p): return abs(p[0]-goal[0]) + abs(p[1]-goal[1])
    open_set = []
    heapq.heappush(open_set, (h(start), 0, start))
    came, g = {start: None}, {start: 0.0}
    
    while open_set:
        f_val, g_val, u = heapq.heappop(open_set)
        if u == goal: return g[u], reconstruct_path(came, start, goal)
        if g_val > g[u]: continue
        
        # GIỮ NGUYÊN 4 HƯỚNG
        for dr, dc in [(1,0), (-1,0), (0,1), (0,-1)]:
            v = (u[0] + dr, u[1] + dc)
            if 0 <= v[0] < H and 0 <= v[1] < W and grid[v] == 0:
                new_cost = g[u] + 1.0
                if v not in g or new_cost < g[v]:
                    g[v] = new_cost
                    came[v] = u
                    heapq.heappush(open_set, (new_cost + h(v), new_cost, v))
    return np.inf, None

def find_valid(p, grid):
    """
    Nếu điểm p (r, c) nằm trên vật cản (grid == 1), 
    tìm pixel trống (grid == 0) gần nhất trong phạm vi 3x3 hoặc 5x5.
    """
    r, c = int(p[0]), int(p[1])
    H, W = grid.shape
    
    # Nếu đã là vùng trống thì trả về luôn
    if grid[r, c] == 0:
        return (r, c)
    
    # Quét rộng dần ra xung quanh (bán kính 1, rồi 2)
    for radius in range(1, 3):
        for dr in range(-radius, radius + 1):
            for dc in range(-radius, radius + 1):
                nr, nc = r + dr, c + dc
                if 0 <= nr < H and 0 <= nc < W and grid[nr, nc] == 0:
                    return (nr, nc)
    
    return (r, c) # Trả về gốc nếu không tìm thấy (trường hợp cực hiếm)
    
def build_reduced_graph(grid, processed_cells, k_neighbors=4):
    K = len(processed_cells)
    is_dict = isinstance(processed_cells, dict) or (isinstance(processed_cells, list) and K > 0 and isinstance(processed_cells[0], dict))
    if is_dict:
        centroids = np.array([processed_cells[i]['centroid'] for i in range(K)])
    else:
        centroids = np.array(processed_cells)
    tree = KDTree(centroids)
    G = {i: [] for i in range(K)}
    for i in range(K):
        _, idxs = tree.query([centroids[i]], k=min(k_neighbors + 1, K))
        for j in idxs[0][1:]:
            start_node = (int(centroids[i][0]), int(centroids[i][1]))
            goal_node = (int(centroids[j][0]), int(centroids[j][1]))
            
            # Hàm phụ find_valid giúp dịch chuyển điểm ra khỏi tường nếu lỡ dính 1 pixel đen
            start_node = find_valid(start_node, grid)
            goal_node = find_valid(goal_node, grid)
            
            best_cost = np.inf
            if is_dict:
                cands_i, cands_j = processed_cells[i]['candidates'], processed_cells[j]['candidates']
                for p_i in cands_i:
                    for p_j in cands_j:
                        dist_gates = abs(p_i[0]-p_j[0]) + abs(p_i[1]-p_j[1])
                        if dist_gates <= 2:
                            d1, _ = astar(grid, start_node, p_i)
                            d2, _ = astar(grid, goal_node, p_j)
                            best_cost = min(best_cost, d1 + dist_gates + d2)

            if best_cost == np.inf:
                best_cost, _ = astar(grid, start_node, goal_node)
            
            if best_cost == np.inf:
                best_cost = np.sqrt((start_node[0]-goal_node[0])**2 + (start_node[1]-goal_node[1])**2) * 5.0

            if best_cost != np.inf:
                G[i].append((j, best_cost))
                # Thêm chiều ngược lại để Dijkstra tìm đường tốt hơn
                G[j].append((i, best_cost))
    return G

def dijkstra_from(G, start, K):
    """Hàm Dijkstra được tách ra để phục vụ cả Unit Test."""
    dist = np.full(K, np.inf)
    dist[start] = 0
    pq = [(0, start)]
    while pq:
        d_u, u = heapq.heappop(pq)
        if d_u > dist[u]: continue
        for v, weight in G.get(u, []):
            if d_u + weight < dist[v]:
                dist[v] = d_u + weight
                heapq.heappush(pq, (dist[v], v))
    return dist

def build_distance_matrix(G, K, grid, processed_cells):
    D = np.zeros((K, K))
    centroids = [processed_cells[i]['centroid'] for i in range(K)]
    
    for i in range(K):
        dist_row = dijkstra_from(G, i, K)
        for j in range(K):
            if dist_row[j] == np.inf:
                # --- LỚP PHÒNG THỦ 3: CỨU CÁNH CUỐI CÙNG ---
                # Nếu Dijkstra trong G thất bại, chạy A* trực tiếp trên Grid
                cost, _ = astar(grid, centroids[i], centroids[j])
                
                if cost != np.inf:
                    dist_row[j] = cost
                else:
                    # Nếu A* vẫn thua (do tường kín), dùng Euclidean để tránh 99999
                    p1, p2 = centroids[i], centroids[j]
                    dist_row[j] = np.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2) * 5.0
                    
        D[i] = dist_row
    return D
