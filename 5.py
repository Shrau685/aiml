# eight_puzzle.py
import heapq

GOAL = (1,2,3,4,5,6,7,8,0)

def manhattan_8(state):
    dist=0
    for idx,val in enumerate(state):
        if val==0: continue
        goal_idx = val-1
        r1,c1 = divmod(idx,3); r2,c2 = divmod(goal_idx,3)
        dist += abs(r1-r2)+abs(c1-c2)
    return dist

def successors_8(state):
    idx_blank = state.index(0)
    r,c = divmod(idx_blank,3)
    moves=[]
    for dr,dc in ((1,0),(-1,0),(0,1),(0,-1)):
        nr,nc = r+dr, c+dc
        if 0<=nr<3 and 0<=nc<3:
            ni = nr*3 + nc
            lst = list(state)
            lst[idx_blank], lst[ni] = lst[ni], lst[idx_blank]
            moves.append(tuple(lst))
    return moves

def a_star_8(start):
    open_heap=[]
    heapq.heappush(open_heap, (manhattan_8(start), 0, start))
    g={start:0}
    parent={start:None}
    while open_heap:
        f,cg,cur = heapq.heappop(open_heap)
        if cur==GOAL:
            path=[]
            node=cur
            while node: path.append(node); node=parent[node]
            return list(reversed(path))
        for nb in successors_8(cur):
            tg = g[cur]+1
            if nb not in g or tg<g[nb]:
                g[nb]=tg; parent[nb]=cur
                heapq.heappush(open_heap, (tg+manhattan_8(nb), tg, nb))
    return None

# Example
if __name__ == "__main__":
    start = (1,0,3,4,2,6,7,5,8)  # small scramble
    #1 0 3
    #4 2 6
    #7 5 8
    path = a_star_8(start)
    print("Moves:", len(path)-1 if path else "no")
    for state in path:
        print(state)