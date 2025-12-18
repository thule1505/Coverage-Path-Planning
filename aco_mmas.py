"""
MMAS (Max-Min Ant System) implementation for Coverage Path Planning (CPP).

Features:
- Fixed Start Point support (e.g., for Charging Station).
- Tau bounds (tau_min, tau_max).
- Only best ant deposits pheromone (iteration-best or global-best).
- 2-opt local search applied to best route per iteration.
- Stagnation detection + light restart.
- Support closed (cycle) or open tours.

Usage:
    from aco_mmas import MMAS
    # D is distance matrix
    solver = MMAS(D, num_ants=20, num_iterations=200, closed_tour=False)
    # Force start at node 0
    best_route, best_cost = solver.run(start_node=0, verbose=True)
"""

import math
import random
import numpy as np
from copy import deepcopy

import math
import random
import numpy as np
import time

class MMAS:
    def __init__(
        self,
        distance_matrix,
        num_ants=20,
        num_iterations=200,
        alpha=1.0,
        beta=3.0, # Tăng beta giúp hội tụ nhanh hơn
        rho=0.02,
        Q=1.0,
        tau_min=None,
        tau_max=None,
        use_global_best=True,
        enable_two_opt=True,
        two_opt_max_iter=30, # Giảm nhẹ để tăng tốc
        stagnation_iters=30,
        closed_tour=False,
        seed=None,
    ):
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        self.distance = np.array(distance_matrix, dtype=float)
        self.n = self.distance.shape[0]

        # Tránh chia cho 0
        self.epsilon = 1e-9
        self.eta = np.zeros_like(self.distance)
        mask = self.distance > self.epsilon
        self.eta[mask] = 1.0 / self.distance[mask]

        self.num_ants = num_ants
        self.num_iterations = num_iterations
        self.alpha = alpha
        self.beta = beta
        self.rho = rho
        self.Q = Q
        self.closed_tour = closed_tour
        self.enable_two_opt = enable_two_opt
        self.two_opt_max_iter = two_opt_max_iter
        self.stagnation_iters = stagnation_iters
        self.use_global_best = use_global_best

        self.pheromone = None
        self.tau_max = 1.0
        self.tau_min = 0.01
        self.best_route = None
        self.best_cost = float('inf')
        self.iter_of_last_improvement = 0

    def _init_pheromone(self, start_node=None):
        approx_best = self._greedy_construction_cost(start_node)
        self.tau_max = 1.0 / (self.rho * max(approx_best, self.epsilon))
        self.tau_min = self.tau_max / (2.0 * self.n) # MMAS standard
        self.pheromone = np.full((self.n, self.n), self.tau_max)

    def _greedy_construction_cost(self, start_node=None):
        start = start_node if start_node is not None else 0
        route = [start]
        unvisited = list(set(range(self.n)) - {start})
        while unvisited:
            curr = route[-1]
            # Tìm nhanh láng giềng gần nhất
            nxt = unvisited[np.argmin(self.distance[curr, unvisited])]
            route.append(nxt)
            unvisited.remove(nxt)
        return self.route_cost(route)

    def route_cost(self, route):
        dists = self.distance[route[:-1], route[1:]]
        cost = np.sum(dists)
        if self.closed_tour:
            cost += self.distance[route[-1], route[0]]
        return cost

    def _select_next_node(self, current, unvisited_list):
        """Vectorized selection - Chỗ này giúp tăng tốc nhiều nhất"""
        if len(unvisited_list) == 1:
            return unvisited_list[0]

        # Lấy pheromone và eta cho tất cả ứng viên cùng lúc
        tau = self.pheromone[current, unvisited_list]
        eta = self.eta[current, unvisited_list]

        # Tính toán xác suất (Alpha và Beta)
        prob = (tau ** self.alpha) * (eta ** self.beta)
        total_prob = prob.sum()

        if total_prob < 1e-15:
            return random.choice(unvisited_list)
        
        prob /= total_prob
        return np.random.choice(unvisited_list, p=prob)

    def build_route(self, start_node):
        route = [start_node]
        unvisited = list(range(self.n))
        unvisited.remove(start_node)
        
        while unvisited:
            curr = route[-1]
            nxt = self._select_next_node(curr, unvisited)
            route.append(nxt)
            unvisited.remove(nxt)
        return route

    def two_opt(self, route):
        """Tối ưu hóa cục bộ để giảm Unproductive Travel"""
        if not self.enable_two_opt: return route
        
        best_r = list(route)
        best_c = self.route_cost(best_r)
        improved = True
        
        for _ in range(self.two_opt_max_iter):
            improved = False
            for i in range(1, self.n - 2):
                for j in range(i + 1, self.n):
                    # Đảo ngược đoạn đường
                    new_r = best_r[:i] + best_r[i:j][::-1] + best_r[j:]
                    new_c = self.route_cost(new_r)
                    if new_c < best_c - 1e-9:
                        best_r = new_r
                        best_cost = new_c
                        improved = True
                if improved: break
            if not improved: break
        return best_r

    def _evaporate(self):
        self.pheromone *= (1.0 - self.rho)
        self.pheromone = np.clip(self.pheromone, self.tau_min, self.tau_max)

    def _deposit(self, route, cost):
        amount = self.Q / max(cost, self.epsilon)
        for i in range(len(route) - 1):
            u, v = route[i], route[i+1]
            self.pheromone[u, v] += amount
            self.pheromone[v, u] += amount # Symmetric
        self.pheromone = np.clip(self.pheromone, self.tau_min, self.tau_max)

    def run(self, start_node=0, verbose=False, early_stopping=25):
        self._init_pheromone(start_node)
        
        for it in range(1, self.num_iterations + 1):
            all_routes = []
            all_costs = []
            
            # 1. Các con kiến xây dựng lộ trình
            for _ in range(self.num_ants):
                r = self.build_route(start_node)
                all_routes.append(r)
                all_costs.append(self.route_cost(r))

            # 2. Tìm con kiến tốt nhất trong vòng lặp này
            best_idx = np.argmin(all_costs)
            it_best_route = all_routes[best_idx]
            it_best_cost = all_costs[best_idx]

            # 3. Chỉ chạy 2-opt cho con tốt nhất (Tiết kiệm rất nhiều thời gian)
            it_best_route = self.two_opt(it_best_route)
            it_best_cost = self.route_cost(it_best_route)

            # 4. Cập nhật Kỷ lục (Global Best)
            improved = False
            if it_best_cost < self.best_cost - 1e-9:
                self.best_cost = it_best_cost
                self.best_route = it_best_route
                self.iter_of_last_improvement = it
                improved = True

            # 5. Bay hơi và để lại Pheromone (Chỉ con tốt nhất được để lại)
            self._evaporate()
            target_route = self.best_route if self.use_global_best else it_best_route
            self._deposit(target_route, self.best_cost if self.use_global_best else it_best_cost)

            # 6. Kiểm tra Dừng sớm hoặc Reset Pheromone nếu bị tắc nghẽn
            if it - self.iter_of_last_improvement >= self.stagnation_iters:
                self.pheromone.fill(self.tau_max) # Reset pheromone
                self.iter_of_last_improvement = it
            
            if it - self.iter_of_last_improvement >= early_stopping:
                if verbose: print(f"-> Early stopping at iter {it}")
                break

            if verbose and (it % 20 == 0 or it == 1):
                print(f"Iter {it:3d} | Best Cost: {self.best_cost:.1f} | {'*' if improved else ''}")

        return self.best_route, self.best_cost
