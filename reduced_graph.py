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
    if grid[start] == 1 or grid[goal] == 1: return np.inf, None
    def h(p): return abs(p[0]-goal[0]) + abs(p[1]-goal[1])
    open_set = []
    heapq.heappush(open_set, (h(start), 0, start))
    came, g = {start: None}, {start: 0.0}
    while open_set:
        f_val, g_val, u = heapq.heappop(open_set)
        if u == goal: return g[u], reconstruct_path(came, start, goal)
        if g_val > g[u]: continue
        for dr, dc in [(1,0), (-1,0), (0,1), (0,-1)]:
            v = (u[0] + dr, u[1] + dc)
            if 0 <= v[0] < H and 0 <= v[1] < W and grid[v] == 0:
                new_cost = g[u] + 1.0
                if v not in g or new_cost < g[v]:
                    g[v] = new_cost
                    came[v] = u
                    heapq.heappush(open_set, (new_cost + h(v), new_cost, v))
    return np.inf, None

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
            if best_cost != np.inf:
                G[i].append((j, best_cost))
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

def build_distance_matrix(G, K):
    D = np.zeros((K, K))
    for i in range(K):
        dist_row = dijkstra_from(G, i, K)
        # Thay np.inf bằng một con số rất lớn (ví dụ 99999)
        # để ACO không bị crash khi tính xác suất
        dist_row[dist_row == np.inf] = 99999 
        D[i] = dist_row
    return D
