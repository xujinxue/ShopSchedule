import copy
import datetime
import random

import matplotlib.pyplot as plt
import numpy as np
import requests

deepcopy = copy.deepcopy
dt = datetime.datetime
time_delta = datetime.timedelta
requests_get = requests.get
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
api_holiday = "http://timor.tech/api/holiday/info/"


class TimeTable:  # 工作时间表类
    @staticmethod
    def format_api_holiday(cur_date):
        return "%s-%s-%s" % (cur_date.year, cur_date.month, cur_date.day)

    @staticmethod
    def get_holiday(cur_date):
        return requests_get("%s/%s" % (api_holiday, TimeTable.format_api_holiday(cur_date))).json()

    @staticmethod
    def crt_value(val_range):
        if val_range[0] == val_range[1]:
            return val_range[0]
        return val_range[0] + int((val_range[1] - val_range[0]) * random.random())

    @staticmethod
    def plot(m_start_date, m_day_period_work_time, rest_start, rest_duration, scale=1, height=0.25,
             ylabel=r'${Machine}$', spines=True):
        ax = plt.gca()
        if spines is False:
            [ax.spines[name].set_color('none') for name in ["top", "right", "bottom", "left"]]
        xticks, x_ticks_label = [], []
        for i, (p, q) in enumerate(zip(rest_start, rest_duration)):
            u = [val * scale for val in p]
            v = [val * scale for val in q]
            w = [val1 + val2 for val1, val2 in zip(u, v)]
            plt.vlines(u, i - height, i + height, colors='red', linestyles="--")
            plt.vlines(w, i - height, i + height, colors='green', linestyles="--")
            if i == 0:
                w0 = [val1 + val2 for val1, val2 in zip(p, q)]
                step = len(m_day_period_work_time[0])
                for t, f in zip(w[::step], w0[::step]):
                    xticks.append(t)
                    label = m_start_date[i] + time_delta(0, int(f))
                    x_ticks_label.append(dt.strftime(label, "%Y-%m-%d %H:%M:%S"))
        plt.xticks(xticks, x_ticks_label, rotation="90")
        m = len(rest_duration)
        plt.yticks(range(m), range(1, m + 1))
        plt.ylabel(ylabel)
        plt.margins()
        plt.tight_layout()

    @staticmethod
    def timetable(m=None, m_start_date=None, m_end_date=None, m_workweek=None,
                  start_max_delay=None, m_day_work_start=None, m_day_work_times=None,
                  m_day_period_work_time=None, m_day_period_rest_duration=None, connect_internet=False,
                  ):
        """
        工作时间表生成
        :param m: 机器数量
        :param m_start_date: 每台机器的开机日期
        :param m_end_date: 每台机器的停机日期
        :param m_workweek: 每台机器的工作制
        :param start_max_delay: 每个工作日的最大延迟开工时间
        :param m_day_work_start: 每个工作日的开始工作时刻
        :param m_day_work_times: 每个工作日的工作时段数
        :param m_day_period_work_time: 每个工作时段的工作时间
        :param m_day_period_rest_duration: 每个工作时间对应的休息时间
        :param connect_internet: 是否联网获取节假日、调休等信息
        :return:
        """
        rest_start, rest_duration, rest_end = [], [], []
        for k in range(m):
            rest_start.append([])
            rest_duration.append([])
            rest_end.append([])
        data = [m, m_start_date, m_end_date, m_workweek, start_max_delay, m_day_work_start,
                m_day_work_times, m_day_period_work_time, m_day_period_rest_duration, connect_internet]
        if all([i is not None for i in data]):
            time_unit = 3600
            total_time_a_day = 24 * time_unit
            for i in range(m):
                m_day_period_work_time[i] = [time_unit * j for j in m_day_period_work_time[i]]
                m_day_period_rest_duration[i] = [time_unit * j for j in m_day_period_rest_duration[i]]
            m_start_date = deepcopy(m_start_date)
            for k in range(m):
                day = 0
                while m_start_date[k] <= m_end_date[k]:
                    status = 0
                    rest_days = 0
                    if connect_internet is True:
                        while True:
                            try:
                                resp = TimeTable.get_holiday(m_start_date[k])
                                status = resp["type"]["type"]
                                if resp["code"] == 0 and status == 2:  # 不属于工作日
                                    m_start_date[k] += time_delta(1)
                                    rest_days += 1
                                else:
                                    break
                            except Exception:
                                break
                    day += rest_days
                    day_start = day * total_time_a_day
                    day_work_start = [m_day_work_start[k], m_day_work_start[k] + start_max_delay / 3600]
                    day_work_start = [time_unit * i for i in day_work_start]
                    day_start_break = TimeTable.crt_value(day_work_start)
                    total_work_break = day_start_break
                    len_rest_start_k = len(rest_start[k])
                    if rest_days > 0:
                        rest = total_time_a_day * rest_days
                        if len_rest_start_k == 0:
                            rest += m_day_work_start[k] * time_unit
                            rest += random.randint(0, start_max_delay)
                        try:
                            rest_duration[k][-1] += rest
                        except IndexError:
                            if len(rest_duration[k]) == 1:
                                rest_duration[k][0] += rest
                            else:
                                rest_start[k].append(0)
                                rest_duration[k].append(rest)
                    if len_rest_start_k == 0:
                        value1 = day_start
                        value2 = day_start_break
                        if value2 == 0:
                            pass
                        else:
                            rest_start[k].append(value1)
                            rest_duration[k].append(value2)
                    if m_start_date[k].weekday() in m_workweek[k] or status == 3:
                        for times in range(m_day_work_times[k]):
                            try:
                                last_start = rest_start[k][-1] + rest_duration[k][-1]
                            except IndexError:
                                last_start = 0
                            w_time = m_day_period_work_time[k][times]
                            b_time = m_day_period_rest_duration[k][times]
                            total_work_break += w_time + b_time
                            if b_time < 0:
                                total_time_a_day_b_time = total_time_a_day
                                total_time_a_day_b_time += m_day_work_start[k] * time_unit
                                total_time_a_day_b_time += random.randint(0, start_max_delay)
                                total_work_break -= b_time
                                b_time = max([0, total_time_a_day_b_time - total_work_break])
                            value_p = last_start + w_time
                            value_t = b_time
                            rest_start[k].append(value_p)
                            rest_duration[k].append(value_t)
                    else:
                        rest = total_time_a_day
                        try:
                            rest_duration[k][-1] += rest
                        except IndexError:
                            if len_rest_start_k == 1:
                                rest_duration[k][0] += rest
                            else:
                                rest_start[k].append(0)
                                rest_duration[k].append(rest)
                    m_start_date[k] += time_delta(1)
                    day += 1
            for k in range(m):
                index_pop = []
                for i, j in enumerate(rest_duration[k]):
                    if j == 0:
                        index_pop.append(i)
                for i, j in enumerate(index_pop):
                    index = j - i
                    rest_start[k].pop(index)
                    rest_duration[k].pop(index)
            for k in range(m):
                rest_start[k] = np.array(rest_start[k], dtype=int)
                rest_duration[k] = np.array(rest_duration[k], dtype=int)
                rest_end[k] = rest_start[k] + rest_duration[k]
        return rest_start, rest_duration, rest_end

    @staticmethod
    def consistent_timetable_b8w5t2(m, year, month, day, duration, connect_internet=False, save=False, show=False):
        m_start_date = [dt(year, month, day) for _ in range(m)]
        m_end_date = [m_start_date[k] + time_delta(duration) for k in range(m)]
        m_day_work_start = [8 for _ in range(m)]
        start_max_delay = 0
        m_workweek = [[0, 1, 2, 3, 4] for _ in range(m)]
        m_day_work_times = [2 for _ in range(m)]
        m_day_period_work_time = [[4, 4] for _ in range(m)]
        m_day_period_rest_duration = [[1, -1] for _ in range(m)]
        rest_start, rest_duration, rest_end = TimeTable.timetable(
            m, m_start_date, m_end_date, m_workweek, start_max_delay, m_day_work_start,
            m_day_work_times, m_day_period_work_time, m_day_period_rest_duration, connect_internet
        )
        if save or show:
            plt.figure(figsize=[11, 6])
            TimeTable.plot(m_start_date, m_day_period_work_time, rest_start, rest_duration,
                           scale=1, height=0.38, spines=False)
            if save:
                plt.savefig("%s-%s-%s %s.png" % (year, month, day, duration), dpi=200)
            if show:
                plt.show()
        return rest_start, rest_end
