import copy

import numpy as np

from ..info import Info
from ..resource import Schedule

deepcopy = copy.deepcopy


class Fsp(Schedule):
    def __init__(self):
        Schedule.__init__(self)

    def decode_permutation(self, code):
        self.clear()
        for i in code:
            for j in range(self.job[i].nop):
                k = self.job[i].task[j].machine
                p = self.job[i].task[j].duration
                try:
                    a = self.job[i].task[j - 1].end
                except KeyError:
                    a = 0
                self.job[i].task[j].start = max([a, self.machine[k].end])
                self.job[i].task[j].end = self.job[i].task[j].start + p
                if self.job[i].task[j].resumable is not None:
                    res1, res2 = self.constrain_timetable(i, j, k, p)
                    if res1 is False:
                        continue
                    self.job[i].task[j].start = res1
                    self.job[i].task[j].end = res2
                if self.machine[k].end < self.job[i].task[j].end:
                    self.machine[k].end = self.job[i].task[j].end
            if self.job[i].task[0].limited_wait is not None:
                for j_end2head in range(self.job[i].nop - 1, 0, -1):
                    limited_wait = self.job[i].task[j_end2head - 1].limited_wait
                    end = self.job[i].task[j_end2head - 1].end
                    start = self.job[i].task[j_end2head].start
                    if start - end - limited_wait > 0:
                        k = self.job[i].task[j_end2head - 1].machine
                        self.job[i].task[j_end2head - 1].start = start - self.job[i].task[j_end2head - 1].duration
                        self.job[i].task[j_end2head - 1].end = start
                        if self.machine[k].end < self.job[i].task[j_end2head - 1].end:
                            self.machine[k].end = self.job[i].task[j_end2head - 1].end
        return Info(code, self)

    def decode_permutation_timetable(self, code):
        self.clear()
        copy_code = deepcopy(code)
        j = 0
        while self.any_task_not_done():
            for i in copy_code:
                try:
                    a = self.job[i].task[j - 1].end
                except KeyError:
                    a = 0
                k, p = self.job[i].task[j].machine, self.job[i].task[j].duration
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
                if self.job[i].task[j].limited_wait is not None:
                    while self.constrain_limited_wait(i, range(j, -1, -1), None) is False:
                        pass
            copy_code = copy_code[np.argsort([self.job[i].task[j].start for i in copy_code])]
            j += 1
        return Info(code, self)

    def decode_random_key(self, code):
        info = self.decode_permutation(np.argsort(code))
        info.code = code
        return info
