from src import *
import numpy as np

n, m = 4, 4
p, tech, proc = [n] * m, [], []
for i in range(n):
    tech.append(np.random.permutation(p[i]).tolist())
    proc.append(np.random.randint(1, 6, p[i]).tolist())
jsp = Utils.create_schedule(Jsp, n, m, p, tech, proc)


def test_ox_pmx(f1, f2, a, b, name="ox"):
    f1 = np.array(list(map(int, f1.split())), dtype=int) - 1
    f2 = np.array(list(map(int, f2.split())), dtype=int) - 1
    info1 = jsp.decode_operation_based_active(f1, direction=0)
    info2 = jsp.decode_operation_based_active(f2, direction=0)
    if name == "ox":
        c1, c2 = info1.ga_crossover_sequence_ox(info2, a, b)
    else:
        c2, c1 = info1.ga_crossover_sequence_pmx(info2, a, b)
    print(""" locus1: {}  locus2: {}
father1: {}
father2: {}
 child1: {}
 child2: {}
""".format(a, b, f1 + 1, f2 + 1, c1 + 1, c2 + 1))


if __name__ == '__main__':
    print("顺序交叉:")
    # 1
    p1 = "1 3 3 4 2 4 3 4 4 2 2 1 3 1 2 1"
    p2 = "3 4 3 2 1 3 4 3 2 1 4 1 4 2 2 1"
    i, j = 3, 6
    test_ox_pmx(p1, p2, i, j, "ox")
    # 2
    p1 = "3 4 4 1 4 2 2 3 3 4 3 1 1 2 2 1"
    p2 = "3 1 3 4 3 4 1 3 1 4 2 2 2 2 4 1"
    i, j = 2, 6
    test_ox_pmx(p1, p2, i, j, "ox")
    # 3
    p1 = "2 3 2 3 4 3 4 2 4 1 4 1 3 2 1 1"
    p2 = "4 2 4 3 4 4 1 3 3 2 2 1 3 1 2 1"
    i, j = 2, 3
    test_ox_pmx(p1, p2, i, j, "ox")
    # 4
    p1 = "1 2 4 1 2 4 1 3 3 3 4 2 3 2 4 1"
    p2 = "3 1 4 4 1 2 1 3 2 4 3 2 3 4 2 1"
    i, j = 4, 8
    test_ox_pmx(p1, p2, i, j, "ox")
    print("部分映射交叉:")
    # 1
    p1 = "4 3 2 3 3 1 4 4 4 2 1 3 2 1 2 1"
    p2 = "4 2 1 3 2 2 2 4 3 3 1 3 4 1 4 1"
    i, j = 11, 12
    test_ox_pmx(p1, p2, i, j, "pmx")
    # 2
    p1 = "4 3 4 4 2 1 2 3 3 2 1 3 4 2 1 1"
    p2 = "3 4 3 4 3 1 3 4 2 4 2 1 2 2 1 1"
    i, j = 6, 14
    test_ox_pmx(p1, p2, i, j, "pmx")
    # 3
    p1 = "1 2 4 2 3 3 3 2 4 3 1 4 2 1 4 1"
    p2 = "2 4 3 4 2 1 3 4 3 2 1 2 1 3 4 1"
    i, j = 6, 11
    test_ox_pmx(p1, p2, i, j, "pmx")
    # 4
    p1 = "2 4 4 3 4 2 4 2 1 1 2 3 3 3 1 1"
    p2 = "3 2 2 1 3 4 1 2 3 1 3 4 4 4 2 1"
    i, j = 12, 15
    test_ox_pmx(p1, p2, i, j, "pmx")
