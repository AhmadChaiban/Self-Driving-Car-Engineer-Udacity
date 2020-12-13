# ----------
# User Instructions:
#
# Create a function compute_value which returns
# a grid of values. The value of a cell is the minimum
# number of moves required to get from the cell to the goal.
#
# If a cell is a wall or it is impossible to reach the goal from a cell,
# assign that cell a value of 99.
# ----------

grid = [[0, 0, 1, 0, 0, 0],
        [0, 0, 1, 0, 0, 0],
        [0, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 1, 0],
        [0, 0, 1, 1, 1, 0],
        [0, 0, 0, 0, 1, 0]]

goal = [len(grid)-1, len(grid[0])-1]
init = [0, 0]
cost_step = 1  # the cost associated with moving from a cell to an adjacent one

action = [[-1 for i in range(len(grid[0]))] for i in range(len(grid))]
policy = [[' ' for i in range(len(grid[0]))] for j in range(len(grid))]

delta = [[-1, 0 ], # go up
         [ 0, -1], # go left
         [ 1, 0 ], # go down
         [ 0, 1 ]] # go right

delta_name = ['^', '<', 'v', '>']

value = [[99 for i in range(len(grid[0]))] for i in range(len(grid))]


def compute_value(grid, goal, cost_step):
    # ----------------------------------------
    # insert code below
    # ----------------------------------------

    # make sure your function returns a grid of values as
    # demonstrated in the previous video.
    change = True
    while change:
        change = False

        for x in range(len(grid)):
            for y in range(len(grid[0])):
                if goal[0] == x and goal[1] == y:
                    if value[x][y] > 0:
                        value[x][y] = 0
                        change = True

                elif grid[x][y] == 0:
                    for a in range(len(delta)):
                        x2 = x + delta[a][0]
                        y2 = y + delta[a][1]

                        if 0 <= x2 < len(grid) and 0 <= y2 < len(grid[0]) and grid[x2][y2] == 0:
                            v2 = value[x2][y2] + cost_step

                            if v2 < value[x][y]:
                                change = True
                                value[x][y] = v2
                                action[x][y] = a
                                policy[x][y] = delta_name[a]

    return policy



action_matrix = compute_value(grid, goal, cost_step)

for row in action_matrix:
    print(row)

