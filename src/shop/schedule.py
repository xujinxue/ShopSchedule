import numpy as np

from ..resource.code import Code
from ..resource.job import Job
from ..resource.machine import Machine


class Schedule(Code):  # 调度资源融合类
    def __init__(self):
        self.job = {}  # 工件
        self.machine = {}  # 机器
        self.best_known = None  # 已知下界值
        self.time_unit = 1  # 加工时间单位
        self.direction = 0  # 解码：正向时间表、反向时间表（仅Jsp, Fjsp）
        # 找关键路径：加工开始时间、加工任务（工序）索引、工件索引、机器索引、加工完成时间
        self.sjike = {0: np.zeros(self.length), 1: np.zeros(self.length, dtype=int),
                      2: np.zeros(self.length, dtype=int), 3: np.zeros(self.length, dtype=int),
                      4: np.zeros(self.length)}

    def clear(self):  # 解码前要进行清空, 方便快速地进行下一次解码
        for i in self.job.keys():
            self.job[i].clear()
        for i in self.machine.keys():
            self.machine[i].clear()
        self.direction = 0
        self.sjike = {0: np.zeros(self.length), 1: np.zeros(self.length, dtype=int),
                      2: np.zeros(self.length, dtype=int), 3: np.zeros(self.length, dtype=int),
                      4: np.zeros(self.length)}

    @property
    def n(self):  # 工件数量
        return len(self.job)

    @property
    def m(self):  # 机器数量
        return len(self.machine)

    @property
    def length(self):  # 总的工序数量
        return sum([job.nop for job in self.job.values()])

    @property
    def makespan(self):  # 工期
        return max([machine.end for machine in self.machine.values()])

    @property
    def a_operation_based_code(self):
        a = np.array([], dtype=int)
        for i in self.job.keys():
            a = np.append(a, [i] * self.job[i].nop)
        return a

    @staticmethod
    def trans_job2operation_based(code, p):  # 转码：基于工件的编码->基于工序的编码
        a = np.array([], dtype=int)
        for i in code:
            a = np.append(a, [i] * p[i])
        return a

    def trans_random_key2operation_based(self, code):  # 转码：基于随机键的编码->基于工序的编码
        return self.a_operation_based_code[np.argsort(code)]

    def any_task_not_done(self):  # 解码：判断是否解码结束（基于机器的编码、混合流水车间、考虑作息时间的流水车间）
        return any([any([task.start is None for task in job.task.values()]) for job in self.job.values()])

    def add_machine(self, name=None, timetable=None, index=None):  # 添加机器
        if index is None:
            index = self.m
        self.machine[index] = Machine(index, name, timetable)

    def add_job(self, due_date=None, name=None, index=None):  # 添加工件
        if index is None:
            index = self.n
        self.job[index] = Job(index, due_date, name)

    def save_sjike(self, i, j, k, g):  # 保存信息: 加工开始时间、加工任务（工序）索引、工件索引、机器索引、加工完成时间
        for u, v in enumerate([self.job[i].task[j].start, j, i, k, self.job[i].task[j].end]):
            self.sjike[u][g] = v

    def update_saved_start_end(self, i, j, g):  # 更新信息: 工件索引、 加工任务（工序）索引、基因座位置
        try:
            self.sjike[0][g] = self.job[i].task[j].start
            self.sjike[4][g] = self.job[i].task[j].end
        except TypeError:
            pass

    def save_update_decode(self, i, j, k, g):  # 解码用: 保存更新信息
        self.job[i].nd += 1
        self.job[i].index_list[j] = g
        self.machine[k].index_list.append(g)
        self.save_sjike(i, j, k, g)

    def decode_update_machine_idle(self, i, j, k, r, early_start):  # 解码：更新机器空闲时间
        if self.machine[k].idle[1][r] - self.job[i].task[j].end > 0:  # 添加空闲时间段
            self.machine[k].idle[0].insert(r + 1, self.job[i].task[j].end)
            self.machine[k].idle[1].insert(r + 1, self.machine[k].idle[1][r])
        if self.machine[k].idle[0][r] == early_start:  # 删除空闲时间段
            self.machine[k].idle[0].pop(r)
            self.machine[k].idle[1].pop(r)
        else:
            self.machine[k].idle[1][r] = early_start  # 更新空闲时间段
        if self.machine[k].end < self.job[i].task[j].end:  # 更新机器上的最大完工时间
            self.machine[k].end = self.job[i].task[j].end

    def decode_common(self, i, j, k, p, v, g=None, save=True):
        try:
            a = self.job[i].task[v].end
        except KeyError:
            a = 0
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
                if save is True:
                    self.save_update_decode(i, j, k, g)
                else:
                    self.update_saved_start_end(i, j, g)
                break

    def decode_add_limited_wait(self, i, j, u):
        if self.job[i].task[j].limited_wait is not None:
            if self.direction == 0:
                index = range(u, -1, -1)
            else:
                index = range(self.job[i].nop - u - 1, self.job[i].nop, 1)
            while self.constrain_limited_wait(i, index, None) is False:
                pass

    def constrain_timetable(self, i, j, k, p, c=None):  # 工作时间表约束
        start, end = self.job[i].task[j].start, self.job[i].task[j].end
        if self.machine[k].timetable is not None and len(self.machine[k].timetable[0]) != 0:
            try:  # 加工开始时刻恰好在停工开始时刻
                index = np.argwhere(self.machine[k].timetable[0] == start)[:, 0][0]
                start = self.machine[k].timetable[1][index]
                end = start + p
            except IndexError:
                pass
            if self.job[i].task[j].resumable == 0:  # 加工不可恢复
                while True:
                    try:
                        constrain1 = np.argwhere(start <= self.machine[k].timetable[0])[:, 0]
                        constrain2 = np.argwhere(self.machine[k].timetable[0] < end)[:, 0]
                        index = list(set(constrain1) & set(constrain2))[0]
                        start = self.machine[k].timetable[1][index]
                        end = start + p
                    except IndexError:
                        break
            else:  # 加工可恢复
                try:
                    index = np.argwhere(start < self.machine[k].timetable[0])[:, 0][0]
                    t_left = end - self.machine[k].timetable[0][index]
                    if t_left > 0:
                        end += self.machine[k].timetable[1][index] - self.machine[k].timetable[0][index]
                        while True:
                            try:
                                constrain1 = np.argwhere(self.machine[k].timetable[0] < end)[:, 0]
                                constrain2 = np.argwhere(end <= self.machine[k].timetable[1])[:, 0]
                                index = list(set(constrain1) & set(constrain2))[0]
                                end += self.machine[k].timetable[1][index] - self.machine[k].timetable[0][index]
                            except IndexError:
                                break
                except IndexError:
                    pass
            if c is None or end <= c:  # c是机器空闲时段的结束时刻
                return start, end
            else:
                return False, False
        return start, end

    def constrain_limited_wait_release_job(self, i, j, k, p):  # 等待时间有限约束：释放工件占用机器的时间
        try:  # 工件紧前是机器的空闲时间段
            index_pre = self.machine[k].idle[1].index(self.job[i].task[j].start)
        except ValueError:  # 工件紧前的是工件
            index_pre = None
        if self.job[i].task[j].end == self.machine[k].end:  # 工件是机器上的当前最后一个加工的
            try:  # 工件紧前是机器的空闲时间段, 更新机器上的最大完工时间, 删除此空闲时间段
                self.machine[k].end = self.machine[k].idle[0][index_pre]
                self.machine[k].idle[0].pop(index_pre)
                self.machine[k].idle[1].pop(index_pre)
            except TypeError:  # 工件紧前是工件, 更新机器上的最大完工时间
                self.machine[k].end -= p
            self.machine[k].idle[0][-1] = self.machine[k].end
        else:  # 工件不是机器上的当前最后一个加工的
            try:  # 工件紧后是机器的空闲时间段
                index_next = self.machine[k].idle[0].index(self.job[i].task[j].end)
            except ValueError:  # 工件紧后是工件
                index_next = None
            if index_pre is None and index_next is None:  # 工件紧前、紧后都是工件
                self.machine[k].idle[0].append(self.job[i].task[j].start)
                self.machine[k].idle[1].append(self.job[i].task[j].end)
                self.machine[k].idle[0].sort()
                self.machine[k].idle[1].sort()
            elif index_pre is not None and index_next is not None:  # 工件紧前、紧后都是机器的空闲时间段
                # 更新紧前空闲时间段的空闲结束开始为紧后空闲时间段的空闲结束时刻
                self.machine[k].idle[1][index_pre] = self.machine[k].idle[1][index_next]
                self.machine[k].idle[0].pop(index_next)
                self.machine[k].idle[1].pop(index_next)
            elif index_pre is not None and index_next is None:  # 工件紧前是机器的空闲时间段, 紧后是工件
                self.machine[k].idle[1][index_pre] = self.job[i].task[j].end
            else:  # 工件紧前是工件, 紧后是机器的空闲时间段
                self.machine[k].idle[0][index_next] = self.job[i].task[j].start

    def constrain_limited_wait_repair_interval(self, i, index, cursor, mac=None):  # 等待时间有限约束：合法化加工时间间隔
        for j, j_pre in zip(index[:cursor + 1][::-1], index[1:cursor + 2][::-1]):
            if mac is None:
                k = self.job[i].task[j].machine
                p = self.job[i].task[j].duration
            else:
                k = mac[i][j]
                p = self.job[i].task[j].duration[self.job[i].task[j].machine.index(k)]
            self.constrain_limited_wait_release_job(i, j, k, p)
            self.decode_common(i, j, k, p, j_pre, g=self.job[i].index_list[j], save=False)

    def constrain_limited_wait(self, i, index, mac=None):  # 等待时间有限约束
        for cursor, (j, j_next) in enumerate(zip(index[1:], index[:-1])):  # index为工序索引
            if self.direction == 0:  # 正向时间表
                limited_wait = self.job[i].task[j].limited_wait
            else:  # 反向时间表
                limited_wait = self.job[i].task[j_next].limited_wait
            cur_end = self.job[i].task[j].end
            next_start = self.job[i].task[j_next].start
            interval_time = next_start - cur_end
            if interval_time > limited_wait:  # 不满足等待时间有限约束
                if mac is None:
                    k = self.job[i].task[j].machine
                    p = self.job[i].task[j].duration
                else:
                    k = mac[i][j]
                    p = self.job[i].task[j].duration[self.job[i].task[j].machine.index(k)]
                delay_start = max([next_start - limited_wait - p, self.job[i].task[j].start])
                self.constrain_limited_wait_release_job(i, j, k, p)
                for r, (b, c) in enumerate(zip(self.machine[k].idle[0], self.machine[k].idle[1])):
                    early_start = max([delay_start, b])  # delay_start是满足等待时间有限约束的最早开始时间
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
                        self.update_saved_start_end(i, j, self.job[i].index_list[j])
                        if next_start - self.job[i].task[j].end < 0:  # 相邻工序的时间间隔不合法
                            self.constrain_limited_wait_repair_interval(i, index, cursor, mac)
                            return False
                        break
        return True
