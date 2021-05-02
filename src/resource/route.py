from .task import Task


class Route:  # 加工路线类
    def __init__(self, index, name=None):
        """
        初始化工件
        :param index: 加工路线索引
        :param name: 加工路线名称
        """
        self.index = index
        self.name = name
        self.task = {}  # 工序集合

    @property
    def nop(self):  # 工序数量
        return len(self.task)

    def add_task(self, machine, duration, name=None, limited_wait=None, resumable=None, index=None):
        """
        添加加工任务（工序）
        :param machine: 加工机器（集）
        :param duration: 加工时间（集）
        :param name: 加工完任务名称
        :param limited_wait: 允许等待时间
        :param resumable: 加工是否可恢复
        :param index: 加工任务（工序）索引
        :return:
        """
        if index is None:
            index = self.nop
        self.task[index] = Task(index, machine, duration, name, limited_wait, resumable)
