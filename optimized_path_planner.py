import heapq
import numpy as np

class OptimizedPathPlanner:
    def __init__(self, width=20, height=20, robot_radius=0):
        self.width = width
        self.height = height
        self.robot_radius = robot_radius
        self.raw_grid = np.zeros((self.height, self.width), dtype=int)
        self.inflated_grid = np.zeros((self.height, self.width), dtype=int)

    def set_obstacle(self, x, y):
        if 0 <= x < self.width and 0 <= y < self.height:
            self.raw_grid[y, x] = 1
            self._inflate_obstacles()

    def _inflate_obstacles(self):
        self.inflated_grid = np.copy(self.raw_grid)
        if self.robot_radius <= 0:
            return

        for y in range(self.height):
            for x in range(self.width):
                if self.raw_grid[y, x] == 1:
                    for dy in range(-self.robot_radius, self.robot_radius + 1):
                        for dx in range(-self.robot_radius, self.robot_radius + 1):
                            nx, ny = x + dx, y + dy
                            if 0 <= nx < self.width and 0 <= ny < self.height:
                                self.inflated_grid[ny, nx] = 1

    def a_star(self, start, goal, current_direction=None):
        if start == goal:
            return [start]

        directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]
        open_set = []
        heapq.heappush(open_set, (0, 0, start, current_direction))
        
        came_from = {}
        cost_so_far = {start: 0}

        while open_set:
            _, current_cost, current_node, last_dir = heapq.heappop(open_set)

            if current_node == goal:
                path = []
                curr = current_node
                while curr in came_from:
                    path.append(curr)
                    curr = came_from[curr]
                path.append(start)
                return path[::-1]

            for d_idx, (dx, dy) in enumerate(directions):
                neighbor = (current_node[0] + dx, current_node[1] + dy)

                if not (0 <= neighbor[0] < self.width and 0 <= neighbor[1] < self.height):
                    continue
                if self.inflated_grid[neighbor[1], neighbor[0]] == 1 and neighbor != goal:
                    continue

                turn_penalty = 0.5 if (last_dir is not None and last_dir != d_idx) else 0.0
                new_cost = current_cost + 1.0 + turn_penalty

                if neighbor not in cost_so_far or new_cost < cost_so_far[neighbor]:
                    cost_so_far[neighbor] = new_cost
                    h_score = abs(neighbor[0] - goal[0]) + abs(neighbor[1] - goal[1])
                    f_score = new_cost + h_score
                    heapq.heappush(open_set, (f_score, new_cost, neighbor, d_idx))
                    came_from[neighbor] = current_node

        return None

    def plan_coverage_path(self, orientation='horizontal', start_pos=(0, 0)):
        cells_to_visit = []
        if orientation == 'horizontal':
            for y in range(self.height):
                x_indices = range(self.width) if y % 2 == 0 else range(self.width - 1, -1, -1)
                for x in x_indices:
                    if self.inflated_grid[y, x] == 0:
                        cells_to_visit.append((x, y))
        else:
            for x in range(self.width):
                y_indices = range(self.height) if x % 2 == 0 else range(self.height - 1, -1, -1)
                for y in y_indices:
                    if self.inflated_grid[y, x] == 0:
                        cells_to_visit.append((x, y))

        if not cells_to_visit:
            return []

        visited_set = set()
        executed_path = []
        current = start_pos if self.inflated_grid[start_pos[1], start_pos[0]] == 0 else cells_to_visit[0]
        
        executed_path.append(current)
        visited_set.add(current)

        while len(visited_set) < len(cells_to_visit):
            adjacents = [(current[0] + dx, current[1] + dy) for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]]
            valid_adjacents = [p for p in adjacents if p in cells_to_visit and p not in visited_set]

            if valid_adjacents:
                next_cell = min(valid_adjacents, key=lambda p: cells_to_visit.index(p))
                executed_path.append(next_cell)
                visited_set.add(next_cell)
                current = next_cell
            else:
                unvisited_cells = [p for p in cells_to_visit if p not in visited_set]
                nearest_goal = min(unvisited_cells, key=lambda p: abs(current[0] - p[0]) + abs(current[1] - p[1]))
                
                bypass_path = self.a_star(current, nearest_goal)
                if bypass_path and len(bypass_path) > 1:
                    for step in bypass_path[1:]:
                        executed_path.append(step)
                        visited_set.add(step)
                    current = nearest_goal
                else:
                    visited_set.add(nearest_goal)

        return executed_path