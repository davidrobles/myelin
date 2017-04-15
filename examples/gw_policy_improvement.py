# -*- encoding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from myelin.mdps.gridworld import GridWorld
from myelin.dp import PolicyIteration
from myelin.policies.random_policy import RandomPolicy

gamma = 1.0
theta = 0.1
rows, cols = 5, 5
mdp = GridWorld(rows, cols)
vf = np.zeros((rows, cols))
policy = RandomPolicy(action_space=mdp.get_actions)
pe = PolicyIteration(mdp, gamma, policy, vf, theta)
pe.iter_policy_eval()
pe.policy_improvement()

# print(mdp)
# print(vi.table)
np.set_printoptions(precision=4, linewidth=200)
print(vf)

for row in range(rows):
    for col in range(cols):
        val = pe.policy.vf[(row, col)]
        if val == (0, -1):
            val = '←'
        elif val == (0, 1):
            val = '→'
        elif val == (-1, 0):
            val = '↑'
        elif val == (1, 0):
            val = '↓'
        else:
            val = '*'
        print(val, end=', ')
    print(end='\n')

plt.matshow(vf)
plt.colorbar()
plt.show()
