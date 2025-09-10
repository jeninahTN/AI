from collections import deque

def bfs(maze):
    if not maze or not maze[0]:
        return None
    
    rows, cols = len(maze), len(maze[0])
    
    
    start = goal = None
    for r in range(rows):
        for c in range(cols):
            if maze[r][c] == 'S':
                start = (r, c)
            elif maze[r][c] == 'G':
                goal = (r, c)
    
    if not start or not goal:
        return None
    
    
    maze[start[0]][start[1]] = '.'
    maze[goal[0]][goal[1]] = '.'
    
    
    visited = set()
    parent = {}
    queue = deque([(start, 0)])  
    visited.add(start)
    directions = [(-1, 0), (0, 1), (1, 0), (0, -1)]  
    
    found = False
    while queue:
        (r, c), steps = queue.popleft()
        if (r, c) == goal:
            found = True
            break
        for dr, dc in directions:
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols and maze[nr][nc] != '#' and (nr, nc) not in visited:
                visited.add((nr, nc))
                parent[(nr, nc)] = (r, c)
                queue.append(((nr, nc), steps + 1))
    
    if not found:
        return None
    
    
    path = []
    current = goal
    while current != start:
        path.append(current)
        current = parent.get(current)
        if current is None:
            return None  
    path.append(start)
    path.reverse()  
    return path


maze = [
    ['S', '.', '.', '#', '.'],
    ['#', '#', '.', '#', '.'],
    ['.', '.', '.', '.', '.'],
    ['.', '#', '#', '#', '.'],
    ['.', '.', '.', '.', 'G']
]

result = bfs(maze)
print(result)  
