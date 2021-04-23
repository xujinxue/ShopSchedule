import numpy as np


class Code:  # 编码类
    @staticmethod
    def sequence_permutation(length):
        """
        基于排列的编码
        :param length: 编码长度
        :return:
from src import Code
a = Code.sequence_permutation(10)
print(a)
        """
        return np.random.permutation(length)

    @staticmethod
    def sequence_random_key(length):
        """
        基于随机键的编码
        :param length: 编码长度
        :return:
from src import Code
a = Code.sequence_random_key(10)
print(a)
        """
        return np.random.uniform(0, 1, length)

    @staticmethod
    def sequence_operation_based(n, p):
        """
        基于工序的编码
        :param n: 工件数量
        :param p: 工序数量
        :return:
from src import Code
a = Code.sequence_operation_based(3, [3, 2, 3]) # 工件1~3的工序数量分别为3, 2, 3
print(a)
        """
        a = np.array([], dtype=int)
        for i in range(n):
            a = np.append(a, [i] * p[i])
        np.random.shuffle(a)
        return a

    @staticmethod
    def sequence_machine_based(n, m, job):
        """
        基于机器的编码
        :param n: 工件数量
        :param m: 机器数量
        :param job: 工件字典
        :return:
import numpy as np
from src import Job, Code
n, m, p = 3, 3, [3, 2, 3]
a = {} # job
for i in range(n):
    a[i] = Job(i)
    b = np.random.choice(range(3), p[i], replace=False)
    c = np.random.randint(1, 6, p[i])
    for j, k in enumerate(b):
        a[i].add_task(k, c[j])
d = Code.sequence_machine_based(n, m, a)
print(d)
        """
        a = []
        for i in range(m):
            b = []
            for j in range(n):
                machine = [task.machine for task in job[j].task.values()]
                if i in machine:
                    k = machine.index(i)
                    b.append((j, k))  # (job, operation)
            np.random.shuffle(b)
            a.append(b)
        return a

    @staticmethod
    def assignment_job_based(n, p, tech):
        """
        基于工件的编码（机器分配）
        :param n: 工件数量
        :param p: 工序数量
        :param tech: 机器集
        :return:
from src import Code
n, m, p = 3, 3, [3, 2, 3]
tech = [
[[0, 1], [1, 2], [1]], # 工件1的第1道工序的加工机器集合为[0, 1], 第2道工序的加工机器集合为[1, 2] ......
[[0, 2], [1, 2]],
[[0, 1], [2], [1, 2]]]
a = Code.assignment_job_based(n, p, tech)
print(a)
        """
        a = []
        for i in range(n):
            a.append([])
            for j in range(p[i]):
                a[i].append(np.random.choice(tech[i][j], 1, replace=False)[0])
        return a

    @staticmethod
    def route_job_based(n, p):
        """
        基于工件的编码（加工路径）
        :param n: 工件数量
        :param p: 工序数量
        :return:
from src import Code
a = Code.route_job_based(3, [3, 2, 3])
print(a)
        """
        a = []
        for i in range(n):
            a.append(np.random.permutation(p[i]))
        return a
