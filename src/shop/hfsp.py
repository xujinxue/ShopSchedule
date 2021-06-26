import copy

import numpy as np

from .schedule import Schedule
from ..info import Info

deepcopy = copy.deepcopy


class Hfsp(Schedule):
    def __init__(self):
        Schedule.__init__(self)

    def decode(self, code):
        self.clear()
        copy_code = deepcopy(code)
        mac, j = [[None for _ in range(job.nop)] for job in self.job.values()], 0
        while self.any_task_not_done():
            for i in copy_code:
                try:
                    a = self.job[i].task[j - 1].end
                except KeyError:
                    a = 0
                start, end, duration, index = [], [], [], []
                for k, p in zip(self.job[i].task[j].machine, self.job[i].task[j].duration):
                    for r, (b, c) in enumerate(zip(self.machine[k].idle[0], self.machine[k].idle[1])):
                        early_start = max([a, b])
                        if early_start + p <= c:
                            res1, res2 = early_start, early_start + p
                            if self.job[i].task[j].resumable is not None:
                                res1, res2 = self.constrain_timetable(i, j, k, p, c)
                                if res1 is False:
                                    continue
                            start.append(res1)
                            end.append(res2)
                            duration.append(p)
                            index.append(r)
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
                if self.job[i].task[0].limited_wait is not None:
                    while self.constrain_limited_wait(i, range(j, -1, -1), mac) is False:
                        pass
            # copy_code = code[np.argsort([self.job[i].task[j].end for i in code])]
            copy_code = copy_code[np.argsort([self.job[i].task[j].end for i in copy_code])]
            j += 1
        return Info(self, code, mac=mac)

    def decode_work_timetable(self, code):
        return self.decode(code)

    def decode_hfsp(self, code, mac):
        self.clear()
        copy_code, j = deepcopy(code), 0
        while self.any_task_not_done():
            for i in copy_code:
                k = mac[i][j]
                p = self.job[i].task[j].duration[self.job[i].task[j].machine.index(k)]
                try:
                    a = self.job[i].task[j - 1].end
                except KeyError:
                    a = 0
                for r, (b, c) in enumerate(zip(self.machine[k].idle[0], self.machine[k].idle[1])):
                    early_start = max([a, b])
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
                        break
                if self.job[i].task[0].limited_wait is not None:
                    while self.constrain_limited_wait(i, range(j, -1, -1), mac) is False:
                        pass
            # copy_code = code[np.argsort([self.job[i].task[j].end for i in code])]
            copy_code = copy_code[np.argsort([self.job[i].task[j].end for i in copy_code])]
            j += 1
        return Info(self, code, mac=mac)
