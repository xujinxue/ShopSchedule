class Task:  # 加工任务（工序）类
    def __init__(self, index, machine, duration, name=None, limited_wait=None, resumable=None):
        """
        初始化加工任务（工序）
        :param index: 加工任务（工序）索引
        :param machine: 加工机器（集)
        :param duration: 加工时间（集）
        :param name: 加工任务（工序）名称
        :param limited_wait: 允许的等待时间
        :param resumable: 加工是否可恢复
        """
        self.index = index
        self.machine = machine
        self.duration = duration
        self.resumable = resumable
        self.name = name
        self.limited_wait = limited_wait
        self.start = None  # 解码用：加工开始时间
        self.end = None  # 解码用：加工完成时间
        self.block = None  # 标记关键块

    def clear(self):  # 解码用：重置
        self.start = None
        self.end = None
        self.block = None
