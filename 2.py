from collections import deque

def bfs_shortest_path(grid, start, goal):
    rows, cols = len(grid), len(grid[0])
    moves = [(1,0), (-1,0), (0,1), (0,-1)]

    q = deque([start])
    parent = {start: None}

    while q:
        r, c = q.popleft()
        if (r, c) == goal:
            # Reconstruct path
            path = []
            while True:
                path.append((r, c))
                if parent[(r, c)] is None:
                    break
                r, c = parent[(r, c)]
            return path[::-1]

        for dr, dc in moves:
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols and grid[nr][nc] == 0 and (nr, nc) not in parent:
                parent[(nr, nc)] = (r, c)
                q.append((nr, nc))

    return None


# Example
grid = [
    [0, 0, 0],
    [0, 1, 0],
    [0, 1, 0]
]

print(bfs_shortest_path(grid, (0, 0), (2, 2)))
