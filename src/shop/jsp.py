from .rule_jsp import RuleJsp
from ..info import Info
from ..resource import Schedule
from ..utils import Utils


class Jsp(Schedule, RuleJsp):
    def __init__(self):
        Schedule.__init__(self)
        RuleJsp.__init__(self, self.job)

    def decode_inner(self, a, i, j, k, g, u, route):
        p = self.job[i].task[j].duration
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
            while self.constrain_limited_wait(i, index, None) is False:
                pass

    def decode_operation_based_active_classic(self, code, route=None, direction=None):
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
            k = self.job[i].task[j].machine
            self.decode_inner(a, i, j, k, g, u, route)
        return Info(code, self, route=route)

    def decode_operation_based_active(self, code, route=None, direction=None):
        # 常规解码
        info = self.decode_operation_based_active_classic(code, route, direction)
        if self.job[0].task[0].limited_wait is not None:
            info.std_code()
            info2 = self.decode_operation_based_active_classic(info.code, route, info.schedule.direction)
            info = info if info.schedule.makespan < info2.schedule.makespan else info2
        # 跳跃式解码
        # info = self.decode_limited_wait(code, route, direction)
        # if self.job[0].task[0].limited_wait is not None:
        #     info.std_code()
        #     info2 = self.decode_limited_wait(info.code, route, info.schedule.direction)
        #     info = info if info.schedule.makespan < info2.schedule.makespan else info2
        return info

    def decode_random_key_active(self, code, route=None, direction=None):
        info = self.decode_operation_based_active(self.trans_random_key2operation_based(code), route, direction)
        info.code = code
        return info

    def decode_no_wait_active(self, code, p, route=None, direction=None):
        info = self.decode_operation_based_active(self.trans_job2operation_based(code, p), route, direction)
        info.code = code
        return info

    def decode_machine_based(self, code, route=None):
        self.clear()
        index = [0 for _ in self.machine.keys()]
        status = [False] * self.m
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
                    for r, (b, c) in enumerate(zip(self.machine[k].idle[0], self.machine[k].idle[1])):
                        early_start = max([f1, b])
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
            for k in range(self.m):
                index[k] = 0
        return Info(code, self, route=route)

    def is_satisfy_limited_wait_constrain(self, i, j, k, a, w):
        for r, (b, c) in enumerate(zip(self.machine[k].idle[0], self.machine[k].idle[1])):
            early_start = max([a, b])
            if early_start + self.job[i].task[j].duration <= c:
                if early_start - a <= w:
                    return True
        return False

    def decode_limited_wait(self, code, route=None, direction=None):
        self.clear()
        self.with_key_block = True
        if direction not in [0, 1]:
            self.direction = Utils.direction()
        else:
            self.direction = direction
        if self.direction == 1:
            code = code[::-1]
        m_list = [[] for _ in range(self.m)]
        for g, i in enumerate(code):
            u = self.job[i].nd
            if route is None:
                if self.direction == 0:
                    j = u
                else:
                    j = self.job[i].nop - u - 1
            else:
                if self.direction == 0:
                    j = route[i][u]
                else:
                    j = route[i][self.job[i].nop - u - 1]
            k = self.job[i].task[j].machine
            m_list[k].append(g)
            self.job[i].nd += 1
        for i in range(self.n):
            self.job[i].nd = 0
        g, g_jump = 0, []
        while self.any_task_not_done():
            while True:
                if g not in g_jump:
                    break
                else:
                    g += 1
            i = code[g]
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
                lw = self.job[i].task[v].limited_wait
            except KeyError:
                a = 0
                lw = None
            k = self.job[i].task[j].machine
            if lw is not None and self.is_satisfy_limited_wait_constrain(i, j, k, a, lw) is True:
                try:
                    g_next = m_list[k][1]
                    i_next = code[g_next]
                    u_next = self.job[i_next].nd
                    if route is None:
                        if self.direction == 0:
                            j_next, v_next = u_next, u_next - 1
                        else:
                            j_next, v_next = self.job[i_next].nop - u_next - 1, self.job[i_next].nop - u_next
                    else:
                        if self.direction == 0:
                            j_next, v_next = route[i_next][u_next], route[i_next][u_next - 1]
                        else:
                            try:
                                j_next, v_next = route[i_next][self.job[i_next].nop - u_next - 1], route[i_next][
                                    self.job[i_next].nop - u_next]
                            except IndexError:
                                j_next, v_next = route[i_next][self.job[i_next].nop - u_next - 1], -1
                    if k == self.job[i_next].task[j_next].machine:
                        try:
                            a_next = self.job[i_next].task[v_next].end
                            lw_next = self.job[i_next].task[v_next].limited_wait
                        except KeyError:
                            a_next = 0
                            lw_next = None
                        if a_next is not None and lw_next is not None and \
                                self.is_satisfy_limited_wait_constrain(i_next, j_next, k, a_next, lw_next) is False:
                            if self.job[i_next].task[j_next].start is None:
                                self.decode_inner(a_next, i_next, j_next, k, g_next, u_next, route)
                                try:
                                    g_jump.append(g_next)
                                    m_list[k].remove(g_next)
                                except ValueError:
                                    pass
                except IndexError:
                    pass
            if self.job[i].task[j].start is None:
                self.decode_inner(a, i, j, k, g, u, route)
                try:
                    m_list[k].remove(g)
                except ValueError:
                    pass
            g += 1
            if g == self.length:
                break
        return Info(code, self, route=route)
