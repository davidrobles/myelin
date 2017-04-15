import numpy as np
import matplotlib.pyplot as plt
from myelin.mdps.gridworld import GridWorld
from myelin.dp import PolicyIteration
from myelin.policies.random_policy import RandomPolicy

gamma = 1.0
theta = 0.1
rows, cols = 14, 14
mdp = GridWorld(rows, cols)
vf = np.zeros((rows, cols))
policy = RandomPolicy(action_space=mdp.get_actions)
pe = PolicyIteration(mdp, gamma, policy, vf, theta)
pe.iter_policy_eval()

# print(mdp)
np.set_printoptions(precision=4, linewidth=200)
# print(vi.table)
print(vf)
plt.matshow(vf)
plt.colorbar()
plt.show()
