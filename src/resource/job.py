import numpy as np

from .route import Route
from .task import Task


class Job:  # 工件类
    def __init__(self, index, due_date=None, name=None):
        """
        初始化工件
        """
        self.index = index
        self.due_date = due_date
        self.name = name
        self.task = {}  # 工序集合
        self.route = {}  # 加工路径集合
        self.nd = 0  # 解码用：已加工的工序数量
        self.index_list = []  # 解码用：工件在编码中的位置索引（基于工序的编码）

    def clear(self):  # 解码用：重置
        for i in self.task.keys():
            self.task[i].clear()
        self.nd = 0
        self.index_list = [None for _ in range(self.nop)]

    @property
    def nop(self):  # 工序数量
        return len(self.task)

    @property
    def nor(self):  # 加工路径数量
        return len(self.route)

    def add_task(self, machine, duration, name=None, limited_wait=None, resumable=None, index=None):
        """
        添加加工任务（工序）
        """
        if index is None:
            index = self.nop
        self.task[index] = Task(index, machine, duration, name, limited_wait, resumable)

    def add_route(self, name=None, index=None):
        """
        添加加工路径
        """
        if index is None:
            index = self.nor
        self.route[index] = Route(name, index)

    @property
    def start(self):  # 工件的加工开始时间
        return min([task.start for task in self.task.values() if task.start != task.end])

    @property
    def end(self):  # 工件的加工完成时间
        return max([task.end for task in self.task.values() if task.start != task.end])

    def done(self, index):  # 调度规则用：已完成的加工时间（index为工件索引）
        a = 0
        for i, j in self.task.items():
            if i <= index:
                a += j.duration
        return a

    def remain(self, index):  # 调度规则用：剩余的加工时间
        a = 0
        for i, j in self.task.items():
            if i >= index:
                a += j.duration
        return a

    def done_limited_wait(self, index):  # 调度规则用：已完成的允许等待时间
        a = 0
        for i, j in self.task.items():
            if i <= index:
                if j.limited_wait != np.inf:
                    a += j.limited_wait
        return a

    def remain_limited_wait(self, index):  # 调度规则用：剩余的允许等待时间
        a = 0
        for i, j in self.task.items():
            if i >= index:
                if j.limited_wait != np.inf:
                    a += j.limited_wait
        return a
