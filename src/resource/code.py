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

    @staticmethod
    def assignment_job_based_route(n, p, tech, route):
        """
        机器指派问题-基于工件的编码
        """
        a = []
        for i in range(n):
            a.append([])
            for j in range(p[i]):
                a[i].append(np.random.choice(tech[i][route[i]][j], 1, replace=False)[0])
        return a

    @staticmethod
    def assignment_route(n, r):
        """
        加工路径问题
        """
        a = []
        for i in range(n):
            a.append(np.random.choice(range(r[i]), 1, replace=False)[0])
        return a

    @staticmethod
    def assignment_route_min_avg_jsp(n, r, job):
        """
        选择平均加工时间短的加工路径
        """
        a = []
        for i in range(n):
            b = []
            for j in range(r[i]):
                c = []
                for u in job[i].route[j].task.values():
                    if u.duration != 0:
                        c.append(u.duration)
                b.append(np.mean(c))
            d = np.argwhere(np.array(b) == min(b))[:, 0]
            a.append(np.random.choice(d, 1, replace=False)[0])
        return a

    @staticmethod
    def assignment_route_min_total_jsp(n, r, job):
        """
        选择总加工时间短的加工路径
        """
        a = []
        for i in range(n):
            b = []
            for j in range(r[i]):
                b.append(0)
                for u in job[i].route[j].task.values():
                    b[j] += u.duration
            d = np.argwhere(np.array(b) == min(b))[:, 0]
            a.append(np.random.choice(d, 1, replace=False)[0])
        return a

    @staticmethod
    def assignment_route_min_avg_fjsp(n, r, job):
        """
        选择平均加工时间短的加工路径
        """
        a = []
        for i in range(n):
            b = []
            for j in range(r[i]):
                c = []
                for u in job[i].route[j].task.values():
                    for v in u.duration:
                        if v != 0:
                            c.append(v)
                b.append(np.mean(c))
            d = np.argwhere(np.array(b) == min(b))[:, 0]
            a.append(np.random.choice(d, 1, replace=False)[0])
        return a

    @staticmethod
    def assignment_route_min_total_fjsp(n, r, job):
        """
        选择总加工时间短的加工路径
        """
        a = []
        for i in range(n):
            b = []
            for j in range(r[i]):
                b.append(0)
                for u in job[i].route[j].task.values():
                    b[j] += sum(u.duration)
            d = np.argwhere(np.array(b) == min(b))[:, 0]
            a.append(np.random.choice(d, 1, replace=False)[0])
        return a

    @staticmethod
    def assignment_worker(n, p, tech, worker, mac):
        """工人安排问题"""
        a = []
        for i in range(n):
            a.append([])
            for j in range(p[i]):
                index = tech[i][j].index(mac[i][j])
                a[i].append(np.random.choice(worker[i][j][index], 1, replace=False)[0])
        return a

    @staticmethod
    def assignment_machine_bml(n, m, p, tech, proc):
        a, b = [], [0] * m
        for i in range(n):
            a.append([])
            for j in range(p[i]):
                c = []
                for m, d in zip(tech[i][j], proc[i][j]):
                    c.append(b[m] + d)
                index = c.index(min(c))
                k = tech[i][j][index]
                b[k] += proc[i][j][index]
                a[i].append(k)
        return a

    @staticmethod
    def assignment_machine_bml_worker(n, m, p, tech, proc):
        a, b = [], [0] * m
        for i in range(n):
            a.append([])
            for j in range(p[i]):
                c = []
                proc_mean = np.mean(proc[i][j])
                for u in tech[i][j]:
                    c.append(b[u] + proc_mean)
                index = c.index(min(c))
                k = tech[i][j][index]
                b[k] += proc_mean
                a[i].append(k)
        return a

    @staticmethod
    def assignment_worker_bwl(n, w, p, tech, worker, proc, mac):
        a, b = [], [0] * w
        for i in range(n):
            a.append([])
            for j in range(p[i]):
                index = tech[i][j].index(mac[i][j])
                c = []
                for u, v in zip(worker[i][j][index], proc[i][j][index]):
                    c.append(b[u] + v)
                index_c = c.index(min(c))
                d = worker[i][j][index][index_c]
                b[d] += proc[i][j][index][index_c]
                a[i].append(d)
        return a
