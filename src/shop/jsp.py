from .rule import RuleJsp
from .schedule import Schedule
from ..info import Info
from ..utils import Utils


class Jsp(Schedule, RuleJsp):
    def __init__(self):
        Schedule.__init__(self)
        RuleJsp.__init__(self, self.job)

    def decode(self, code, route=None, direction=None):
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
            k = self.job[i].task[j].machine
            p = self.job[i].task[j].duration
            self.decode_common(i, j, k, p, v, g)
            self.decode_add_limited_wait(i, j, u)
        return Info(self, code, route=route)

    def decode_limited_wait(self, code, route=None, direction=None):
        info = self.decode(code, route, direction)
        info.std_code()
        info2 = self.decode(info.code, info.route, info.schedule.direction)
        info = info if info.schedule.makespan < info2.schedule.makespan else info2
        return info

    def decode_no_wait(self, code, p, route=None, direction=None):
        info = self.decode(self.trans_job2operation_based(code, p), route, direction)
        info.code = code
        return info

    def decode_machine_based(self, code, route=None):
        self.clear(route)
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
        return Info(self, code, route=route)

    def decode_machine_based_limited_wait(self, code, route=None):
        info = self.decode_machine_based(code, route)
        info.std_code_machine_based()
        info2 = self.decode_machine_based(info.code, info.route)
        info = info if info.schedule.makespan < info2.schedule.makespan else info2
        return info

    def decode_new(self, code, route=None, direction=None):
        self.clear(route)
        if direction not in [0, 1]:
            self.direction = Utils.direction()
        else:
            self.direction = direction
        if self.direction == 1:
            code = code[::-1]
        m_list = [[] for _ in range(self.m)]
        for g, i in enumerate(code):
            u = self.job[i].nd
            if self.direction == 0:
                j = u
            else:
                j = self.job[i].nop - u - 1
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
            if self.direction == 0:
                j, v = u, u - 1
            else:
                j, v = self.job[i].nop - u - 1, self.job[i].nop - u
            try:
                a = self.job[i].task[v].end
                a = 0 if a is None else a
            except KeyError:
                a = 0
            k = self.job[i].task[j].machine
            try:
                g_next = m_list[k][1]
                i_next = code[g_next]
                u_next = self.job[i_next].nd
                if self.direction == 0:
                    j_next, v_next = u_next, u_next - 1
                else:
                    j_next, v_next = self.job[i_next].nop - u_next - 1, self.job[i_next].nop - u_next
                if k == self.job[i_next].task[j_next].machine:
                    try:
                        a_next = self.job[i_next].task[v_next].end
                        a_next = 0 if a_next is None else a_next
                    except KeyError:
                        a_next = 0
                    if a_next < a:
                        if self.job[i_next].task[j_next].start is None:
                            p = self.job[i_next].task[j_next].duration
                            self.decode_common(i_next, j_next, k, p, v_next, g_next)
                            try:
                                g_jump.append(g_next)
                                m_list[k].remove(g_next)
                            except ValueError:
                                pass
            except IndexError:
                pass
            if self.job[i].task[j].start is None:
                p = self.job[i].task[j].duration
                self.decode_common(i, j, k, p, v, g)
                try:
                    m_list[k].remove(g)
                except ValueError:
                    pass
            g += 1
            if g == self.length:
                break
        return Info(self, code, route=route)

    def decode_new_twice(self, code, route=None, direction=None):
        info = self.decode_new(code, route, direction)
        info.std_code()
        info2 = self.decode_new(info.code, info.route, info.schedule.direction)
        info = info if info.schedule.makespan < info2.schedule.makespan else info2
        return info

    def is_satisfy_limited_wait_constrain(self, i, j, k, a, w):
        for r, (b, c) in enumerate(zip(self.machine[k].idle[0], self.machine[k].idle[1])):
            early_start = max([a, b])
            if early_start + self.job[i].task[j].duration <= c:
                if early_start - a <= w:
                    return True
        return False

    def decode_limited_wait_new(self, code, route=None, direction=None):
        self.clear(route)
        if direction not in [0, 1]:
            self.direction = Utils.direction()
        else:
            self.direction = direction
        if self.direction == 1:
            code = code[::-1]
        m_list = [[] for _ in range(self.m)]
        for g, i in enumerate(code):
            u = self.job[i].nd
            if self.direction == 0:
                j = u
            else:
                j = self.job[i].nop - u - 1
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
            if self.direction == 0:
                j, v = u, u - 1
            else:
                j, v = self.job[i].nop - u - 1, self.job[i].nop - u
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
                    if self.direction == 0:
                        j_next, v_next = u_next, u_next - 1
                    else:
                        j_next, v_next = self.job[i_next].nop - u_next - 1, self.job[i_next].nop - u_next
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
                                p = self.job[i_next].task[j_next].duration
                                self.decode_common(i_next, j_next, k, p, v_next, g_next)
                                try:
                                    g_jump.append(g_next)
                                    m_list[k].remove(g_next)
                                except ValueError:
                                    pass
                except IndexError:
                    pass
            if self.job[i].task[j].start is None:
                p = self.job[i].task[j].duration
                self.decode_common(i, j, k, p, v, g)
                try:
                    m_list[k].remove(g)
                except ValueError:
                    pass
            g += 1
            if g == self.length:
                break
        return Info(self, code, route=route)

    def decode_limited_wait_new_twice(self, code, route=None, direction=None):
        info = self.decode_limited_wait_new(code, route, direction)
        info.std_code()
        info2 = self.decode_limited_wait_new(info.code, info.route, info.schedule.direction)
        info = info if info.schedule.makespan < info2.schedule.makespan else info2
        return info
