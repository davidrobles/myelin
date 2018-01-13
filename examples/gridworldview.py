import matplotlib.pyplot as plt
import numpy as np

from myelin.dp import ValueIteration
from myelin.mdps.gridworld import GridWorld

ROWS, COLS = 44, 44
mdp = GridWorld(ROWS, COLS)
vfunction = np.zeros((ROWS, COLS))
# vi = ValueIteration(mdp, 0.001, 0.99, vfunction)
vi = ValueIteration(mdp, 0.001, 1, vfunction)
vi.learn()

print(mdp)
np.set_printoptions(precision=4, linewidth=200)
print(vi.table)
plt.matshow(vi.table)
plt.colorbar()
plt.show()
