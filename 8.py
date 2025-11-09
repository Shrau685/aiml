from collections import deque

def water_jug_problem(jug1_capacity, jug2_capacity, target):
    """
    Solves the Water Jug Problem using Breadth-First Search (BFS).

    Args:
        jug1_capacity (int): Capacity of the first jug.
        jug2_capacity (int): Capacity of the second jug.
        target (int): The target amount of water to achieve.

    Returns:
        list: A list of tuples representing the sequence of states (jug1_amount, jug2_amount)
              to reach the target, or None if no solution exists.
    """
    # Queue for BFS, storing (jug1_current, jug2_current, path_to_current_state)
    queue = deque([(0, 0, [(0, 0)])])
    # Set to keep track of visited states to avoid cycles
    visited = set([(0, 0)])

    while queue:
        jug1_current, jug2_current, path = queue.popleft()

        # Check if the target is reached in either jug
        if jug1_current == target or jug2_current == target:
            return path

        # Generate possible next states
        next_states = []

        # 1. Fill Jug 1
        next_states.append((jug1_capacity, jug2_current))

        # 2. Fill Jug 2
        next_states.append((jug1_current, jug2_capacity))

        # 3. Empty Jug 1
        next_states.append((0, jug2_current))

        # 4. Empty Jug 2
        next_states.append((jug1_current, 0))

        # 5. Pour from Jug 1 to Jug 2
        pour_amount = min(jug1_current, jug2_capacity - jug2_current)
        next_states.append((jug1_current - pour_amount, jug2_current + pour_amount))

        # 6. Pour from Jug 2 to Jug 1
        pour_amount = min(jug2_current, jug1_capacity - jug1_current)
        next_states.append((jug1_current + pour_amount, jug2_current - pour_amount))

        for next_jug1, next_jug2 in next_states:
            if (next_jug1, next_jug2) not in visited:
                visited.add((next_jug1, next_jug2))
                queue.append((next_jug1, next_jug2, path + [(next_jug1, next_jug2)]))

    return None  # No solution found

# Example Usage:
jug1_cap = 4
jug2_cap = 3
target_amount = 2

solution_path = water_jug_problem(jug1_cap, jug2_cap, target_amount)

if solution_path:
    print(f"Solution found for target {target_amount} with jugs of capacities {jug1_cap} and {jug2_cap}:")
    for state in solution_path:
        print(state)
else:
    print(f"No solution found for target {target_amount} with jugs of capacities {jug1_cap} and {jug2_cap}.")
