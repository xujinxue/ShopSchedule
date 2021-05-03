from .rule import RuleJsp
from ..info import Info
from .schedule import Schedule
from ..utils import Utils


class Jsp(Schedule, RuleJsp):
    def __init__(self):
        Schedule.__init__(self)
        RuleJsp.__init__(self, self.job)

    def decode(self, code, direction=None):
        self.clear()
        if direction not in [0, 1]:
            self.direction = Utils.direction()
        else:
            self.direction = direction
        if self.direction == 1:
            code = code[::-1]
        for g, i in enumerate(code):
            u = self.job[i].nd
            if self.direction == 0:
                j, v = u, u - 1
            else:
                j, v = self.job[i].nop - u - 1, self.job[i].nop - u
            k = self.job[i].task[j].machine
            p = self.job[i].task[j].duration
            self.decode_common(i, j, k, p, v, g)
            self.decode_add_limited_wait(i, j, u)
        return Info(self, code)

    def decode_limited_wait(self, code, direction=None):
        info = self.decode(code, direction)
        info.std_code()
        info2 = self.decode(info.code, info.schedule.direction)
        info = info if info.schedule.makespan < info2.schedule.makespan else info2
        return info

    def decode_no_wait(self, code, p, direction=None):
        info = self.decode(self.trans_job2operation_based(code, p), direction)
        info.code = code
        return info

    def decode_machine_based(self, code):
        self.clear()
        index = [0 for _ in self.machine.keys()]
        status = [False] * self.m
        g = 0
        while self.any_task_not_done():
            while True:
                for k in range(self.m):
                    i, j = code[k][index[k]]
                    try:
                        f1 = self.job[i].task[j - 1].end
                    except KeyError:
                        f1 = 0
                    if self.job[i].task[j].start is None and f1 is not None:
                        status[k] = True
                        break
                    else:
                        status[k] = False
                if any(status):
                    break
                else:
                    for k in range(self.m):
                        index[k] += 1
                        if index[k] == len(code[k]):
                            index[k] = 0
            for k in range(self.m):
                i, j = code[k][index[k]]
                try:
                    f1 = self.job[i].task[j - 1].end
                except KeyError:
                    f1 = 0
                if self.job[i].task[j].start is None and f1 is not None:
                    p = self.job[i].task[j].duration
                    self.decode_common(i, j, k, p, j - 1, g=g)
                    self.decode_add_limited_wait(i, j, j)
                    g += 1
            for k in range(self.m):
                index[k] = 0
        return Info(self, code)

    def decode_machine_based_limited_wait(self, code):
        info = self.decode_machine_based(code)
        info.std_code_machine_based()
        info2 = self.decode_machine_based(info.code)
        info = info if info.schedule.makespan < info2.schedule.makespan else info2
        return info
