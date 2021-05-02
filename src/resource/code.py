import numpy as np


class Code:  # 编码类
    @staticmethod
    def sequence_permutation(length):
        """
        基于排列的编码
        """
        return np.random.permutation(length)

    @staticmethod
    def sequence_random_key(length):
        """
        基于随机键的编码
        """
        return np.random.uniform(0, 1, length)

    @staticmethod
    def sequence_operation_based(n, p):
        """
        基于工序的编码
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
        机器指派问题-基于工件的编码
        """
        a = []
        for i in range(n):
            a.append([])
            for j in range(p[i]):
                a[i].append(np.random.choice(tech[i][j], 1, replace=False)[0])
        return a
