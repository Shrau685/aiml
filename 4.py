# astar_maze.py
import heapq

def astar(grid, start, goal):
    rows, cols = len(grid), len(grid[0])
    def neighbors(r,c):
        for dr,dc in ((1,0),(-1,0),(0,1),(0,-1)):
            nr, nc = r+dr, c+dc
            if 0<=nr<rows and 0<=nc<cols and grid[nr][nc]==0:
                yield (nr,nc)
    def manhattan(a,b): return abs(a[0]-b[0]) + abs(a[1]-b[1])

    open_heap=[]
    g = {start:0}
    parent = {start:None}
    heapq.heappush(open_heap, (manhattan(start,goal), 0, start))
    while open_heap:
        f, cg, cur = heapq.heappop(open_heap)
        if cur==goal:
            path=[]; node=cur
            while node: path.append(node); node=parent[node]
            return list(reversed(path))
        for nb in neighbors(*cur):
            tentative_g = g[cur] + 1
            if nb not in g or tentative_g < g[nb]:
                g[nb] = tentative_g
                parent[nb] = cur
                heapq.heappush(open_heap, (tentative_g + manhattan(nb,goal), tentative_g, nb))
    return None

# Example usage similar to BFS.
if __name__ == "__main__":
    maze = [
        [0,0,0,0,1,0],
        [1,1,0,1,1,0],
        [0,0,0,0,0,0],
        [0,1,1,1,1,1],
        [0,0,0,0,0,0]
    ]
    start = (0,0)
    goal = (4,5)
    path = astar(maze, start, goal)
    print("Path from start to goal:", path)