import numpy as np

from ..info import Info
from ..resource import Schedule
from ..utils import Utils


class Fjsp(Schedule):
    def __init__(self):
        Schedule.__init__(self)

    def decode_operation_based_active(self, code, mac, route=None, direction=None):
        self.clear()
        self.with_key_block = True
        if direction not in [0, 1]:
            self.direction = Utils.direction()
        else:
            self.direction = direction
        if self.direction == 1:
            code = code[::-1]
        for g, i in enumerate(code):
            u = self.job[i].nd
            if route is None:
                if self.direction == 0:
                    j, v = u, u - 1
                else:
                    j, v = self.job[i].nop - u - 1, self.job[i].nop - u
            else:
                if self.direction == 0:
                    j, v = route[i][u], route[i][u - 1]
                else:
                    try:
                        j, v = route[i][self.job[i].nop - u - 1], route[i][self.job[i].nop - u]
                    except IndexError:
                        j, v = route[i][self.job[i].nop - u - 1], -1
            try:
                a = self.job[i].task[v].end
            except KeyError:
                a = 0
            k = mac[i][j]
            p = self.job[i].task[j].duration[self.job[i].task[j].machine.index(k)]
            for r, (b, c) in enumerate(zip(self.machine[k].idle[0], self.machine[k].idle[1])):
                try:
                    early_start = max([a, b])
                except TypeError:
                    early_start = max([0, b])
                if early_start + p <= c:
                    self.job[i].task[j].start = early_start
                    self.job[i].task[j].end = early_start + p
                    if self.job[i].task[j].resumable is not None:
                        res1, res2 = self.constrain_timetable(i, j, k, p, c)
                        if res1 is False:
                            continue
                        self.job[i].task[j].start = res1
                        self.job[i].task[j].end = res2
                    self.decode_update_machine_idle(i, j, k, r, self.job[i].task[j].start)
                    self.save_update_decode(i, j, k, g)
                    break
            if self.job[i].task[j].limited_wait is not None:
                if self.direction == 0:
                    index = range(u, -1, -1)
                else:
                    index = range(self.job[i].nop - u - 1, self.job[i].nop, 1)
                if route is not None:
                    index = [route[i][v] for v in index]
                while self.constrain_limited_wait(i, index, mac) is False:
                    pass
        return Info(code, self, mac, route)

    def decode_only_operation_based_active(self, code, route=None, direction=None):
        self.clear()
        self.with_key_block = True
        if direction not in [0, 1]:
            self.direction = Utils.direction()
        else:
            self.direction = direction
        if self.direction == 1:
            code = code[::-1]
        mac = [[None for _ in range(job.nop)] for job in self.job.values()]
        for g, i in enumerate(code):
            u = self.job[i].nd
            if route is None:
                if self.direction == 0:
                    j, v = u, u - 1
                else:
                    j, v = self.job[i].nop - u - 1, self.job[i].nop - u
            else:
                if self.direction == 0:
                    j, v = route[i][u], route[i][u - 1]
                else:
                    try:
                        j, v = route[i][self.job[i].nop - u - 1], route[i][self.job[i].nop - u]
                    except IndexError:
                        j, v = route[i][self.job[i].nop - u - 1], -1
            try:
                a = self.job[i].task[v].end
            except KeyError:
                a = 0
            start, end, index, duration = [], [], [], []
            for k, p in zip(self.job[i].task[j].machine, self.job[i].task[j].duration):
                for r, (b, c) in enumerate(zip(self.machine[k].idle[0], self.machine[k].idle[1])):
                    try:
                        early_start = max([a, b])
                    except TypeError:
                        early_start = max([0, b])
                    if early_start + p <= c:
                        res1, res2 = early_start, early_start + p
                        if self.job[i].task[j].resumable is not None:
                            res1, res2 = self.constrain_timetable(i, j, k, p, c)
                            if res1 is False:
                                continue
                        start.append(res1)
                        end.append(res2)
                        index.append(r)
                        duration.append(p)
                        break
            index_min_end = np.argwhere(np.array(end) == min(end))[:, 0]
            duration_in_min_end = np.array([duration[i] for i in index_min_end])
            choice_min_end_and_duration = np.argwhere(duration_in_min_end == np.min(duration_in_min_end))[:, 0]
            choice = index_min_end[np.random.choice(choice_min_end_and_duration, 1, replace=False)[0]]
            k, p, r = self.job[i].task[j].machine[choice], duration[choice], index[choice]
            mac[i][j] = k
            self.job[i].task[j].start = start[choice]
            self.job[i].task[j].end = end[choice]
            self.decode_update_machine_idle(i, j, k, r, start[choice])
            self.save_update_decode(i, j, k, g)
            if self.job[i].task[j].limited_wait is not None:
                if self.direction == 0:
                    index = range(u, -1, -1)
                else:
                    index = range(self.job[i].nop - u - 1, self.job[i].nop, 1)
                if route is not None:
                    index = [route[i][v] for v in index]
                while self.constrain_limited_wait(i, index, mac) is False:
                    pass
        return Info(code, self, mac, route)

    def decode_only_random_key_active(self, code, route=None, direction=None):
        info = self.decode_only_operation_based_active(self.trans_random_key2operation_based(code), route, direction)
        info.code = code
        return info

    def decode_no_wait_active(self, code, mac, p, route=None, direction=None):
        info = self.decode_operation_based_active(self.trans_job2operation_based(code, p), mac, route, direction)
        info.code = code
        return info

    def decode_no_wait_only_job_active(self, code, p, route=None, direction=None):
        info = self.decode_only_operation_based_active(self.trans_job2operation_based(code, p), route, direction)
        info.code = code
        return info
