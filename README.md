# AI
Artificial Intelligence
Question
1. Maze Problem
 Given a 5x5 (or 10x10) grid maze, identify:
 a) Start state
 b) Goal state
 c) Possible actions from each cell
 d) A valid solution path using BFS, and all other search strategies
2. You are designing an AI system for ambulance dispatch in a city. Define the state space, actions, goal, and path cost for 
this problem. Suggest a suitable search strategy and justify your choice.

Solution
1. a) (0,0)
   b) (4,4)
   c) UP (-1,0), Right(0,1), Down(1,0), Left(0,-1)
   d) BFS - (0,0) , (0,1) , (0,2) , (1,2) , (2,2) , (2,3) , (2,4) , (3,4) , (4,4)
      DFS - From (0,0), it tries Up (invalid), then Right to (0,1), then (0,2).
            Next it tries Up (blocked), then Right to (0,3) (blocked), then Down to (1,2).
            It keeps going deep until it eventually finds (4,4).
      UCS (Uniform Cost Search)- UCS expands the path with the lowest cost so far.
                                 Since every step costs 1 in the maze, UCS behaves the same as BFS.
                                  UCS path = same as BFS: [(0,0) , (0,1) , (0,2) , (1,2) , (2,2) , (2,3) , (2,4) , (3,4) , (4,4)]
     Greedy Best-First Search - From (0,0), Greedy always picks the node that looks closest to (4,4).
                                This often causes it to head towards the goal in a straight line, even if blocked → it may waste time backtracking.
                               In the  maze, Greedy would likely try to go right/down toward (4,4) quickly, but get stuck at the wall of # around (3,1)–(3,3).
                               It will eventually find the goal, but possibly with a less optimal path than A*.
     A* Search - It balances progress-so-far with closeness-to-goal.
                 It will take the same optimal path as BFS/UCS, but usually expands fewer nodes.
                 Path will be the  same as BFS/UCS, but  a more efficient search.

   2. state space - Each state is the ambulance’s location on the city grid.
                    A state is represented as a tuple (row, col). i.e (0,0) means the ambulance is at the top-left corner.
      actions - From any state (r, c), the ambulance can move: Up, Down, Left, Right. i.e  (1,0), (-1,0), (0,1), (0,-1)
      goal - Reach the patient’s location. i.e (4,4)
      path cost - This refers to the total number of moves from start to goal. Each move has a cost of 1 ( time or distance). Therefore, the path cost is 9 for this problem.

  A suitable search strategy is A* search because it finds the shortest and most effecient path which is required for an ambulance since it handles emergencies.


 
 

