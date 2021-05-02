import numpy as np


class Machine:  # 机器类
    def __init__(self, index, name=None, timetable=None):
        """
        初始化机器
        :param index: 机器索引
        :param name: 机器名称
        :param timetable: 工作时间表
        """
        self.index = index
        # 时间表数据类型： 字典， 0：休息开始时刻（工作结束时刻），1：休息结束时刻（工作开始时刻）
        # self.timetable = {0: np.array([], dtype=int), 1: np.array([], dtype=int) # 0: restStart, 1: restEnd
        self.timetable = timetable
        self.name = name
        self.end = 0  # 工件上的最大完成时间
        # 机器空闲时间数据类型：字典，0：空闲开始时刻，1：空闲结束时刻
        self.idle = {0: [0, ], 1: [np.inf, ]}
        self.index_list = []  # 解码用： 机器在编码中的位置索引（基于工序的编码）

    def clear(self):  # 解码用：重置
        self.end = 0
        self.idle = {0: [0, ], 1: [np.inf, ]}
        self.index_list = []
