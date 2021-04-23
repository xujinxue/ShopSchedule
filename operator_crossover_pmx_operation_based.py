from src import *
import numpy as np

n, m, p = 3, 3, [3, 3, 3]
tech, proc = [], []
for i in range(n):
    tech.append(np.random.permutation(p[i]).tolist())
    proc.append(np.random.randint(1, 6, p[i]).tolist())
jsp = Utils.create_schedule(Jsp, n, m, p, tech, proc)
# a1=jsp.sequence_operation_based(n,p)
# a2 = jsp.sequence_operation_based(n,p)
a1 = np.array([1, 3, 2, 3, 2, 1, 3, 2, 1], dtype=int) - 1
a2 = np.array([1, 1, 2, 3, 2, 2, 3, 1, 3], dtype=int) - 1
p1 = jsp.decode_operation_based_active(a1, direction=0)
p2 = jsp.decode_operation_based_active(a2, direction=0)
c1, c2 = p1.ga_crossover_sequence_pmx(p2, a=3, b=6)
print(a1 + 1, a2 + 1, "# p1,p2")
print(c1 + 1, c2 + 1, "# c1,c2")
