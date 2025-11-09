# dfs_traversal.py
def dfs_recursive(graph, start, visited=None, order=None):
    if visited is None: 
        visited=set()
    if order is None: 
        order=[]
    visited.add(start)
    order.append(start)
    for nb in graph.get(start, []):
        if nb not in visited:
            dfs_recursive(graph, nb, visited, order)
    return order

# def dfs_iterative(graph, start):
#     visited=set(); stack=[start]; order=[]
#     while stack:
#         node = stack.pop()
#         if node in visited: continue
#         visited.add(node); order.append(node)
#         # push neighbors in reverse to preserve natural ordering
#         for nb in reversed(graph.get(node, [])):
#             if nb not in visited:
#                 stack.append(nb)
#     return order

# Example
if __name__ == "__main__":
    graph = {"A":["B","C"], "B":["D"], "C":["E","F"], "D":[], "E":[], "F":[]}
    print(dfs_recursive(graph,"A"))
    #print(dfs_iterative(graph,"A"))
