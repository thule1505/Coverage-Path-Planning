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

class MMAS:
    def __init__(
        self,
        distance_matrix,
        num_ants=20,
        num_iterations=200,
        alpha=1.0,
        beta=2.0,
        rho=0.02,
        Q=1.0,
        tau_min=None,
        tau_max=None,
        use_global_best=True,
        enable_two_opt=True,
        two_opt_max_iter=50,
        stagnation_iters=50,
        closed_tour=True,  # Set False for CPP if robot doesn't need to return
        seed=None,
    ):
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        self.distance = np.array(distance_matrix, dtype=float)
        assert self.distance.ndim == 2 and self.distance.shape[0] == self.distance.shape[1]
        self.n = self.distance.shape[0]

        # Safety for 0 division
        self.epsilon = 1e-9
        safe_dist = self.distance.copy()
        safe_dist[safe_dist <= 0] = self.epsilon
        self.eta = 1.0 / safe_dist

        # Parameters
        self.num_ants = num_ants
        self.num_iterations = num_iterations
        self.alpha = alpha
        self.beta = beta
        self.rho = rho
        self.Q = Q
        self.closed_tour = closed_tour

        # Strategy
        self.use_global_best = use_global_best
        self.enable_two_opt = enable_two_opt
        self.two_opt_max_iter = two_opt_max_iter
        self.stagnation_iters = stagnation_iters

        # Pheromone bounds logic (initialized later or now)
        self.tau_min_input = tau_min
        self.tau_max_input = tau_max

        # Will be initialized in run() or manually
        self.pheromone = None
        self.tau_max = 1.0
        self.tau_min = 0.01

        # Bookkeeping
        self.best_route = None
        self.best_cost = float('inf')
        self.iter_of_last_improvement = 0

    def _init_pheromone(self, start_node=None):
        """Initialize pheromone based on a greedy estimate."""
        # Calculate greedy cost to estimate tau_max
        approx_best = self._greedy_construction_cost(start_node)

        if self.tau_max_input is None:
            self.tau_max = 1.0 / (self.rho * max(approx_best, self.epsilon))
        else:
            self.tau_max = self.tau_max_input

        if self.tau_min_input is None:
            # Formula: tau_min = tau_max / a, where a ~ n
            a = max(10, self.n)
            self.tau_min = self.tau_max / a
        else:
            self.tau_min = self.tau_min_input

        # Fill matrix with tau_max
        self.pheromone = np.full((self.n, self.n), self.tau_max, dtype=float)

    def _greedy_construction_cost(self, start_node=None):
        """Construct a greedy route to estimate initial Pheromone bounds."""
        if start_node is not None:
            candidates = [start_node]
        else:
            # Try a few random starts if no fixed start
            candidates = range(min(self.n, 5))

        best = float('inf')
        for start in candidates:
            route = [start]
            unvisited = set(range(self.n)) - {start}
            while unvisited:
                i = route[-1]
                # Nearest neighbor
                j = min(unvisited, key=lambda x: self.distance[i, x] if self.distance[i, x] > 0 else float('inf'))
                route.append(j)
                unvisited.remove(j)
            cost = self.route_cost(route)
            if cost < best:
                best = cost
        return best

    def route_cost(self, route):
        if len(route) < 2:
            return 0.0
        cost = 0.0
        for i in range(len(route) - 1):
            cost += self.distance[route[i], route[i+1]]
        if self.closed_tour and len(route) > 1:
            cost += self.distance[route[-1], route[0]]
        return cost

    def _select_next(self, current, unvisited):
        candidates = list(unvisited)
        taus = np.array([self.pheromone[current, j] for j in candidates], dtype=float)
        etas = np.array([self.eta[current, j] for j in candidates], dtype=float)

        numerators = (taus ** self.alpha) * (etas ** self.beta)
        denom = numerators.sum()

        if denom <= 0 or np.isnan(denom):
            return random.choice(candidates)

        probs = numerators / denom
        chosen = np.random.choice(len(candidates), p=probs)
        return candidates[chosen]

    def build_route(self, start):
        route = [start]
        unvisited = set(range(self.n))
        unvisited.remove(start)

        while unvisited:
            cur = route[-1]
            nxt = self._select_next(cur, unvisited)
            route.append(nxt)
            unvisited.remove(nxt)
        return route

    def two_opt(self, route, max_iter=None):
        """
        2-opt local search.
        Note: The loop starts at i=1, so route[0] (Start Point) is NEVER moved.
        This is safe for fixed start point problems.
        """
        if not self.enable_two_opt:
            return route
        if max_iter is None:
            max_iter = self.two_opt_max_iter

        best = route[:]
        best_cost = self.route_cost(best)
        improved = True
        iter_count = 0
        n = len(route)

        while improved and iter_count < max_iter:
            improved = False
            iter_count += 1
            # i starts from 1 preserves route[0]
            for i in range(1, n - 2):
                for j in range(i + 1, n):
                    if j - i == 1:
                        continue
                    new_route = best[:]
                    # Reverse segment from i to j-1
                    new_route[i:j] = reversed(best[i:j])

                    new_cost = self.route_cost(new_route)
                    if new_cost + 1e-12 < best_cost:
                        best = new_route
                        best_cost = new_cost
                        improved = True
        return best

    def _evaporate(self):
        self.pheromone *= (1.0 - self.rho)
        np.clip(self.pheromone, self.tau_min, self.tau_max, out=self.pheromone)

    def _deposit(self, route, cost):
        deposit_amount = self.Q / max(cost, 1e-12)
        for i in range(len(route) - 1):
            a, b = route[i], route[i+1]
            self.pheromone[a, b] += deposit_amount
            self.pheromone[b, a] += deposit_amount # Symmetric

        if self.closed_tour and len(route) > 1:
            a, b = route[-1], route[0]
            self.pheromone[a, b] += deposit_amount
            self.pheromone[b, a] += deposit_amount

        np.clip(self.pheromone, self.tau_min, self.tau_max, out=self.pheromone)

    def run(self, start_node=None, verbose=False):
        """
        Args:
            start_node (int): If provided, all ants start here (e.g., 0).
                              If None, ants start randomly.
            verbose (bool): Print logs.
        """
        # Reset / Init
        self._init_pheromone(start_node)
        self.best_route = None
        self.best_cost = float('inf')
        self.iter_of_last_improvement = 0

        for it in range(1, self.num_iterations + 1):
            all_routes = []
            all_costs = []

            # 1. Construct Solutions
            if start_node is not None:
                # Fixed start point
                starts = [start_node] * self.num_ants
            else:
                # Random start points
                starts = [random.randrange(self.n) for _ in range(self.num_ants)]

            for k in range(self.num_ants):
                r = self.build_route(start=starts[k])
                c = self.route_cost(r)
                all_routes.append(r)
                all_costs.append(c)

            # 2. Find Iteration Best
            idx_best = int(np.argmin(all_costs))
            iter_best_route = all_routes[idx_best]
            iter_best_cost = all_costs[idx_best]

            # 3. Local Search (2-opt) on Iteration Best
            if self.two_opt:
                improved_r = self.two_opt(iter_best_route)
                improved_c = self.route_cost(improved_r)
                if improved_c < iter_best_cost:
                    iter_best_route = improved_r
                    iter_best_cost = improved_c

            # 4. Update Global Best
            improved_flag = False
            if iter_best_cost + 1e-12 < self.best_cost:
                self.best_cost = iter_best_cost
                self.best_route = iter_best_route[:]
                self.iter_of_last_improvement = it
                improved_flag = True

            # 5. Pheromone Update
            self._evaporate()

            if self.use_global_best and self.best_route is not None:
                self._deposit(self.best_route, self.best_cost)
            else:
                self._deposit(iter_best_route, iter_best_cost)

            # 6. Stagnation Check
            if it - self.iter_of_last_improvement >= self.stagnation_iters:
                if verbose:
                    print(f"[MMAS] Stagnation at iter {it}, resetting pheromone.")
                self.pheromone.fill(self.tau_max)
                self.iter_of_last_improvement = it

            # Logging
            if verbose and (it % max(1, self.num_iterations // 10) == 0 or it == 1):
                avg = np.mean(all_costs)
                print(f"Iter {it:3d} | Best: {self.best_cost:.2f} | Avg: {avg:.2f} | {'*' if improved_flag else ''}")

        return self.best_route, self.best_cost

if __name__ == "__main__":
    # --- Example Usage for CPP ---
    np.random.seed(42)
    random.seed(42)

    # 1. Generate dummy nodes (e.g., 15 points on a 2D plane)
    num_points = 15
    coords = np.random.rand(num_points, 2) * 100

    # 2. Create Distance Matrix (Euclidean)
    from scipy.spatial.distance import cdist
    dist_matrix = cdist(coords, coords, metric='euclidean')

    # 3. Solve with fixed start point (Node 0)
    # closed_tour=False means we don't need to return to start (Path, not Cycle)
    solver = MMAS(dist_matrix, num_ants=30, num_iterations=100, closed_tour=False)

    print("--- Running ACO with Start Node = 0 ---")
    route, cost = solver.run(start_node=0, verbose=True)

    print("\nFinal Route:", route)
    print("Start Node Check:", route[0] == 0) # Should be True
    print("Final Cost:", cost)
