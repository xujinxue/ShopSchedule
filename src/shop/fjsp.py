import numpy as np

from .schedule import Schedule
from ..info import Info
from ..utils import Utils


class Fjsp(Schedule):
    def __init__(self):
        Schedule.__init__(self)

    def decode(self, code, mac, route=None, direction=None):
        self.clear(route)
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
            k = mac[i][j]
            p = self.job[i].task[j].duration[self.job[i].task[j].machine.index(k)]
            self.decode_common(i, j, k, p, v, g)
            self.decode_add_limited_wait(i, j, u, mac=mac)
        return Info(self, code, mac=mac, route=route)

    def decode_new(self, code, route=None, direction=None):
        self.clear(route)
        if direction not in [0, 1]:
            self.direction = Utils.direction()
        else:
            self.direction = direction
        if self.direction == 1:
            code = code[::-1]
        mac = [[None for _ in range(job.nop)] for job in self.job.values()]
        for g, i in enumerate(code):
            u = self.job[i].nd
            if self.direction == 0:
                j, v = u, u - 1
            else:
                j, v = self.job[i].nop - u - 1, self.job[i].nop - u
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
            self.decode_add_limited_wait(i, j, u, mac=mac)
        return Info(self, code, mac=mac, route=route)

    def decode_limited_wait(self, code, mac, route=None, direction=None):
        info = self.decode(code, mac, route, direction)
        info.std_code()
        info2 = self.decode(info.code, info.mac, info.route, info.schedule.direction)
        info = info if info.schedule.makespan < info2.schedule.makespan else info2
        return info

    def decode_limited_wait_new(self, code, route=None, direction=None):
        info = self.decode_new(code, route, direction)
        info.std_code()
        info2 = self.decode(info.code, info.mac, info.route, info.schedule.direction)
        info = info if info.schedule.makespan < info2.schedule.makespan else info2
        return info

    def decode_no_wait(self, code, mac, p, route=None, direction=None):
        info = self.decode(self.trans_job2operation_based(code, p), mac, route, direction)
        info.code = code
        return info

    def decode_no_wait_new(self, code, p, route=None, direction=None):
        info = self.decode_new(self.trans_job2operation_based(code, p), route, direction)
        info.code = code
        return info

    def decode_worker(self, code, mac, wok=None, direction=None):
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
            k = mac[i][j]
            index_k = self.job[i].task[j].machine.index(k)
            if wok is not None:
                w = wok[i][j]
                index = self.job[i].task[j].worker[index_k].index(w)
                p = self.job[i].task[j].duration[index_k][index]
            else:
                p = self.job[i].task[j].duration[index_k]
            self.decode_common(i, j, k, p, v, g, wok)
            self.decode_add_limited_wait(i, j, u, mac=mac, wok=wok)
        return Info(self, code, mac=mac, wok=wok)

    def decode_worker_new(self, code, direction=None):
        self.clear()
        if direction not in [0, 1]:
            self.direction = Utils.direction()
        else:
            self.direction = direction
        if self.direction == 1:
            code = code[::-1]
        mac = [[None for _ in range(job.nop)] for job in self.job.values()]
        wok = [[None for _ in range(job.nop)] for job in self.job.values()]
        for g, i in enumerate(code):
            u = self.job[i].nd
            if self.direction == 0:
                j, v = u, u - 1
            else:
                j, v = self.job[i].nop - u - 1, self.job[i].nop - u
            try:
                a = self.job[i].task[v].end
            except KeyError:
                a = 0
            start, end, index, duration, worker = [], [], [], [], []
            for k, w_list, p_list in zip(self.job[i].task[j].machine, self.job[i].task[j].worker,
                                         self.job[i].task[j].duration):
                t_start, t_end, t_index = [], [], []
                for w, p in zip(w_list, p_list):
                    try:
                        d = max([a, self.worker[w].end])
                    except TypeError:
                        d = a if a is not None else self.worker[w].end
                    for r, (b, c) in enumerate(zip(self.machine[k].idle[0], self.machine[k].idle[1])):
                        try:
                            early_start = max([d, b])
                        except TypeError:
                            early_start = max([0, b])
                        if early_start + p <= c:
                            res1, res2 = early_start, early_start + p
                            if self.job[i].task[j].resumable is not None:
                                res1, res2 = self.constrain_timetable(i, j, k, p, c)
                                if res1 is False:
                                    continue
                            t_start.append(res1)
                            t_end.append(res2)
                            t_index.append(r)
                            break
                index_min_end = np.argwhere(np.array(t_end) == min(t_end))[:, 0]
                duration_in_min_end = np.array([p_list[i] for i in index_min_end])
                choice_min_end_and_duration = np.argwhere(duration_in_min_end == np.min(duration_in_min_end))[:, 0]
                choice = index_min_end[np.random.choice(choice_min_end_and_duration, 1, replace=False)[0]]
                start.append(t_start[choice])
                end.append(t_end[choice])
                index.append(t_index[choice])
                duration.append(p_list[choice])
                worker.append(w_list[choice])
            index_min_end = np.argwhere(np.array(end) == min(end))[:, 0]
            duration_in_min_end = np.array([duration[i] for i in index_min_end])
            choice_min_end_and_duration = np.argwhere(duration_in_min_end == np.min(duration_in_min_end))[:, 0]
            choice = index_min_end[np.random.choice(choice_min_end_and_duration, 1, replace=False)[0]]
            k, w, p, r = self.job[i].task[j].machine[choice], worker[choice], duration[choice], index[choice]
            mac[i][j], wok[i][j] = k, w
            self.job[i].task[j].start = self.worker[w].start = start[choice]
            self.job[i].task[j].end = self.worker[w].end = end[choice]
            self.decode_update_machine_idle(i, j, k, r, start[choice])
            self.save_update_decode(i, j, k, g, w)
            self.decode_add_limited_wait(i, j, u, mac=mac, wok=wok)
        return Info(self, code, mac=mac, wok=wok)
