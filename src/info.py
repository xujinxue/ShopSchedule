import copy
import datetime
import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly as py
import plotly.figure_factory as ff
from matplotlib import colors as mcolors

from .resource import Schedule
from .utils import Utils

deepcopy = copy.deepcopy
pyplt = py.offline.plot
dt = datetime.datetime
tmdelta = datetime.timedelta
COLORS = list(mcolors.CSS4_COLORS)
COLORS_REMOVE = ['black', "white"]
COLORS_REMOVE.extend([i for i in COLORS if i.startswith('dark')])
COLORS_REMOVE.extend([i for i in COLORS if i.startswith('light')])
[COLORS.remove(i) for i in COLORS_REMOVE]
LEN_COLORS = len(COLORS)
[COLORS.pop(j - i) for i, j in enumerate(range(11))]
[COLORS.pop(j - i) for i, j in enumerate([6, ])]
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


class GanttChartFromCsv:
    def __init__(self, file=None, schedule=None, mac=None):
        self.schedule = schedule
        self.mac = mac
        if file is not None:
            self.data = pd.read_csv(file)
            self.n = max(self.data.loc[:, "Job"])
            self.m = max(self.data.loc[:, "Machine"])
            self.makespan = max(self.data.loc[:, "End"])
            self.schedule = Schedule()
            if self.data.loc[0, "Start"] > self.data.loc[1, "Start"]:
                self.schedule.direction = 1
            else:
                self.schedule.direction = 0
            self.schedule.with_key_block = True
            for i in range(self.m):
                self.schedule.add_machine(name=i, index=i)
            for i in range(self.n):
                self.schedule.add_job(name=i, index=i)
            for g, (start, operation, job, machine, end, duration) in enumerate(zip(
                    self.data.loc[:, "Start"], self.data.loc[:, "Operation"], self.data.loc[:, "Job"],
                    self.data.loc[:, "Machine"], self.data.loc[:, "End"], self.data.loc[:, "Duration"])):
                job, operation, machine = job - 1, operation - 1, machine - 1
                for i, val in enumerate([start, operation, job, machine, end]):
                    self.schedule.start_number_jme[i] = np.append(self.schedule.start_number_jme[i], val)
                self.schedule.job[job].add_task(machine=machine, duration=duration, name=operation, index=operation)
                self.schedule.job[job].task[operation].start = start
                self.schedule.job[job].task[operation].end = end
                self.schedule.job[job].index_list.append(g)
                self.schedule.machine[machine].index_list.append(g)
                if end > self.schedule.machine[machine].end:
                    self.schedule.machine[machine].end = end

    def key_route(self, index):
        a = np.argwhere(self.schedule.start_number_jme[4] == self.schedule.makespan)[:, 0]
        if a.shape[0] > 1:
            b = np.random.choice(a, 1, replace=False)[0]
        else:
            b = a[0]
        if b not in index:
            index.append(b)
        while True:
            c = self.schedule.start_number_jme[0][b]
            d = self.schedule.start_number_jme[2][b]
            e = self.schedule.start_number_jme[3][b]
            try:
                tmp = self.schedule.start_number_jme[4][self.schedule.job[d].index_list]
                f = self.schedule.job[d].index_list[np.argwhere(tmp == c)[:, 0][0]]
            except IndexError:
                f = None
            try:
                tmp = self.schedule.start_number_jme[4][self.schedule.machine[e].index_list]
                g = self.schedule.machine[e].index_list[np.argwhere(tmp == c)[:, 0][0]]
            except IndexError:
                g = None
            if f is not None and g is not None:
                if f not in index:
                    b = f
                elif g not in index:
                    b = g
                else:
                    b = np.random.choice([f, g], 1, replace=False)[0]
            elif f is not None and g is None:
                b = f
            elif f is None and g is not None:
                b = g
            if b not in index:
                index.append(b)
            if f is None and g is None:
                break
        return index

    def key_block(self, n=5):
        index = []
        for i in range(n):
            index = self.key_route(index)
        index_start = self.schedule.start_number_jme[0][index]
        index = [index[i] for i in np.argsort(-index_start)]
        a = self.schedule.start_number_jme[3][index]
        b = set(a)
        c = 0
        block = {}
        for i in b:
            block[c] = np.array([], dtype=int)
            d = np.argwhere(a == i)[:, 0]
            start = self.schedule.start_number_jme[0][[index[j] for j in d]].tolist()
            for cur, j in enumerate(d):
                g = index[j]
                try:
                    end = self.schedule.start_number_jme[4][g]
                    start.index(end)
                except ValueError:
                    if cur != 0:
                        c += 1
                        block[c] = np.array([], dtype=int)
                self.schedule.job[self.schedule.start_number_jme[2][g]].task[
                    self.schedule.start_number_jme[1][g]].block = c
                block[c] = np.append(block[c], g)
            c += 1
        return block

    def ganttChart_png(self, fig_width=9, fig_height=5, filename="GanttChart", random_colors=False, lang=1, dpi=200,
                       height=0.8, scale_more=None, x_step=None, y_based=0, text_rotation=0,
                       with_operation=True, with_start_end=False, show=False):
        if random_colors:
            random.shuffle(COLORS)
        plt.figure(figsize=[fig_width, fig_height])
        plt.yticks(range([self.schedule.m, self.schedule.n][y_based]),
                   range(1, [self.schedule.m, self.schedule.n][y_based] + 1))
        plt.xticks([], [])
        scale_more = 12 if scale_more is None else scale_more
        x_step = self.schedule.makespan // 10 if x_step is None else x_step
        ax = plt.gca()
        for job in self.schedule.job.values():
            for task in job.task.values():
                if self.mac is None:
                    machine = task.machine
                else:
                    machine = self.mac[job.index][task.index]
                y_values = [machine, job.index]
                y = y_values[y_based]
                width = task.end - task.start
                left = [task.start, self.schedule.makespan - task.end][self.schedule.direction]
                plt.barh(
                    y=y, width=width,
                    left=left, color=COLORS[(y_values[y_based - 1] + 1) % LEN_COLORS],
                    edgecolor="black", linewidth=0.5,
                )
                if with_operation:
                    if y_based == 0:
                        mark = r"$O_{%s,%s}$" % (job.index + 1, task.index + 1)
                    else:
                        mark = r"$O_{%s,%s}^{%s}$" % (job.index + 1, task.index + 1, machine + 1)
                    plt.text(
                        x=left + 0.5 * width, y=y,
                        s=mark, c="black",
                        ha="center", va="center", rotation="vertical",
                    )
                if with_start_end:
                    if self.schedule.direction == 0:
                        val = [task.start, task.end]
                    else:
                        val = [self.schedule.makespan - task.end, self.schedule.makespan - task.start]
                    for x in val:
                        s = r"$_{%s}$" % int(x)
                        rotation = text_rotation
                        if text_rotation in [0, 1]:
                            rotation = ["horizontal", "vertical"][text_rotation]
                        plt.text(
                            x=x, y=y - height * 0.5,
                            s=s, c="black",
                            ha="center", va="top",
                            rotation=rotation,
                        )
        if y_based == 0:
            for job in self.schedule.job.values():
                plt.barh(0, 0, color=COLORS[(job.index + 1) % LEN_COLORS], label=job.index + 1)
            plt.barh(y=0, width=self.schedule.makespan / scale_more, left=self.schedule.makespan, color="white")
            if lang == 0:
                title = r"${Job}$"
            else:
                title = "工件"
            plt.legend(loc="best", title=title)
        if y_based == 1:
            for machine in self.schedule.machine.values():
                plt.barh(0, 0, color=COLORS[(machine.index + 1) % LEN_COLORS], label=machine.index + 1)
            plt.barh(y=0, width=self.schedule.makespan / scale_more, left=self.schedule.makespan, color="white")
            if lang == 0:
                title = r"${Machine}$"
            else:
                title = "机器"
            plt.legend(loc="best", title=title)
        if not with_start_end:
            ymin = -0.5
            ymax = [self.schedule.m, self.schedule.n][y_based] + ymin
            plt.vlines(self.schedule.makespan, ymin, ymax, colors="red", linestyles="--")
            plt.text(self.schedule.makespan, ymin, "{}".format(int(self.schedule.makespan / self.schedule.time_unit)))
            x_ticks = range(0, self.schedule.makespan + x_step, x_step)
            plt.xticks(x_ticks, [int(i / self.schedule.time_unit) for i in x_ticks])
            [ax.spines[name].set_color('none') for name in ["top", "right"]]
        else:
            [ax.spines[name].set_color('none') for name in ["top", "right", "bottom", "left"]]
        if lang == 0:
            if self.schedule.time_unit == 1:
                plt.xlabel(r"${Time}$")
            else:
                plt.xlabel(r"${Time}({%s}seconds/1)$" % self.schedule.time_unit)
            plt.ylabel([r"${Machine}$", r"${Job}$"][y_based])
        else:
            plt.ylabel(["机器", "工件"][y_based])
            if self.schedule.time_unit == 1:
                plt.xlabel("时间")
            else:
                plt.xlabel("时间（%s秒/1）" % self.schedule.time_unit)
        plt.margins()
        plt.tight_layout()
        plt.gcf().subplots_adjust(left=0.08, bottom=0.12)
        if not filename.endswith(".png"):
            filename += ".png"
        plt.savefig("{}".format(filename), dpi=dpi)
        if show:
            plt.show()
        plt.clf()
        Utils.print("Create {}".format(filename), fore=Utils.fore().LIGHTCYAN_EX)

    @property
    def rgb(self):
        return random.randint(0, 255)

    def ganttChart_html(self, date=None, filename="GanttChart", show=False, lang=1):
        if date is None:
            today = dt.today()
            date = dt(today.year, today.month, today.day)
        else:
            tmp = list(map(int, date.split()))
            date = dt(tmp[0], tmp[1], tmp[2])
        df = []
        for job in self.schedule.job.values():
            for task in job.task.values():
                if self.mac is None:
                    machine = task.machine
                else:
                    machine = self.mac[job.index][task.index]
                mark = machine + 1
                if self.schedule.m >= 100:
                    if mark < 10:
                        mark = "00" + str(mark)
                    elif mark < 100:
                        mark = "0" + str(mark)
                elif self.schedule.m >= 10:
                    if mark < 10:
                        mark = "0" + str(mark)
                start, finish = task.start, task.end
                if self.schedule.direction == 1:
                    start, finish = self.schedule.makespan - task.end, self.schedule.makespan - task.start
                df.append(dict(Task="M%s" % mark, Start=date + tmdelta(0, int(start)),
                               Finish=date + tmdelta(0, int(finish)),
                               Resource=str(job.index + 1), complete=job.index + 1))
        df = sorted(df, key=lambda val: (val['Task'], val['complete']), reverse=True)
        colors = {}
        for i in self.schedule.job.keys():
            key = "%s" % (i + 1)
            colors[key] = "rgb(%s, %s, %s)" % (self.rgb, self.rgb, self.rgb)
        fig = ff.create_gantt(df, colors=colors, index_col='Resource', group_tasks=True, show_colorbar=True)
        if lang == 0:
            label = "Job"
        else:
            label = "工件"
        fig.update_layout(showlegend=True, legend_title_text=label)
        if not filename.endswith(".html"):
            filename += ".html"
        pyplt(fig, filename="{}".format(filename), auto_open=show)
        Utils.print("Create {}".format(filename), fore=Utils.fore().LIGHTCYAN_EX)


class GanttChart(GanttChartFromCsv):
    def __init__(self, schedule, code, mac=None, route=None):
        self.schedule = deepcopy(schedule)
        self.code = deepcopy(code)
        self.mac = deepcopy(mac)
        self.route = deepcopy(route)
        GanttChartFromCsv.__init__(self, schedule=self.schedule, mac=self.mac)

    def std_code(self, std_direction=None):
        if std_direction not in [0, 1]:
            std_direction = Utils.direction()
        if self.schedule.direction == 0:
            if std_direction == 0:
                index = np.argsort(self.schedule.start_number_jme[0])
            else:
                index = np.argsort(-self.schedule.start_number_jme[0])[::-1]
        else:
            if std_direction == 0:
                index = np.argsort(self.schedule.start_number_jme[0])[::-1]
            else:
                index = np.argsort(-self.schedule.start_number_jme[0])
        self.code = self.code[index]
        for i in self.schedule.job.keys():
            self.schedule.job[i].nd = 0
            self.schedule.job[i].index_list = [None for _ in range(self.schedule.job[i].nop)]
        for i in self.schedule.machine.keys():
            self.schedule.machine[i].index_list = []
        try:
            for g, i in enumerate(self.code):
                u = self.schedule.job[i].nd
                if self.route is None:
                    if self.schedule.direction == 0:
                        j = u
                    else:
                        j = self.schedule.job[i].nop - u - 1
                else:
                    if self.schedule.direction == 0:
                        j = self.route[i][u]
                    else:
                        try:
                            j = self.route[i][self.schedule.job[i].nop - u - 1]
                        except IndexError:
                            j = self.route[i][self.schedule.job[i].nop - u - 1]
                if self.mac is None:
                    k = self.schedule.job[i].task[j].machine
                else:
                    k = self.mac[i][j]
                self.schedule.job[i].index_list[j] = g
                self.schedule.machine[k].index_list.append(g)
                for u, v in enumerate([self.schedule.job[i].task[j].start, j, i, k, self.schedule.job[i].task[j].end]):
                    self.schedule.start_number_jme[u][g] = v
                self.schedule.job[i].nd += 1
        except KeyError:
            code = self.schedule.trans_random_key2operation_based(self.code)
            for g, i in enumerate(code):
                u = self.schedule.job[i].nd
                if self.route is None:
                    if self.schedule.direction == 0:
                        j = u
                    else:
                        j = self.schedule.job[i].nop - u - 1
                else:
                    if self.schedule.direction == 0:
                        j = self.route[i][u]
                    else:
                        try:
                            j = self.route[i][self.schedule.job[i].nop - u - 1]
                        except IndexError:
                            j = self.route[i][self.schedule.job[i].nop - u - 1]
                if self.mac is None:
                    k = self.schedule.job[i].task[j].machine
                else:
                    k = self.mac[i][j]
                self.schedule.job[i].index_list[j] = g
                self.schedule.machine[k].index_list.append(g)
                for u, v in enumerate([self.schedule.job[i].task[j].start, j, i, k, self.schedule.job[i].task[j].end]):
                    self.schedule.start_number_jme[u][g] = v
                self.schedule.job[i].nd += 1

    def std_code_machine_based(self):
        for k in self.schedule.machine.keys():
            start = []
            for i, j in self.code[k]:
                start.append(self.schedule.job[i].task[j].start)
            self.code[k] = [self.code[k][u] for u in np.argsort(start)]

    def save_gantt_chart_to_csv(self, file):
        if not file.endswith(".csv"):
            file = file + ".csv"
        with open(file, "w", encoding="utf-8") as f:
            f.writelines("Job,Operation,Machine,Start,Duration,End\n")
            for job in self.schedule.job.values():
                for task in job.task.values():
                    if self.mac is None:
                        machine = task.machine
                        duration = task.duration
                    else:
                        machine = self.mac[job.index][task.index]
                        duration = self.schedule.job[job.index].task[task.index].duration[
                            self.schedule.job[job.index].task[task.index].machine.index(machine)]
                    f.writelines("{},{},{},{},{},{}\n".format(
                        job.index + 1, task.index + 1, machine + 1, task.start, duration, task.end))


class Info(GanttChart):
    def __init__(self, code, schedule, mac=None, route=None):
        GanttChart.__init__(self, schedule, code, mac, route)

    def ha_sequence_machine_based_end_move(self, code=None):
        if code is None:
            self.std_code_machine_based()
            code = deepcopy(self.code)
        end = [machine.end for machine in self.schedule.machine.values()]
        i = end.index(max(end))
        len_code_i = len(code[i])
        j = np.random.choice([i for i in range(len_code_i - 1)], 1, replace=False)[0]
        k = len_code_i - 1
        obj = code[i][k]
        code[i].remove(obj)
        code[i].insert(j, obj)
        return code

    def ha_sequence_machine_based_last_job_shift(self, code=None):
        if code is None:
            self.std_code_machine_based()
            code = deepcopy(self.code)
        end = [machine.end for machine in self.schedule.machine.values()]
        job = self.schedule.job[code[end.index(max(end))][-1][0]]
        machine = [task.machine for task in job.task.values()]
        for i in machine:
            len_code_i = len(code[i])
            k = len_code_i - 1
            j = np.random.choice([i for i in range(k)], 1, replace=False)[0]
            obj = code[i][k]
            code[i].remove(obj)
            code[i].insert(j, obj)
        return code

    def ha_sequence_machine_based_one_job_shift(self, code=None):
        if code is None:
            self.std_code_machine_based()
            code = deepcopy(self.code)
        job = self.schedule.job[np.random.choice(range(self.schedule.n), 1, replace=False)[0]]
        machine = [task.machine for task in job.task.values()]
        for i in machine:
            j, k = np.random.choice(range(len(code[i])), 2, replace=False)
            if j > k:
                j, k = k, j
            obj = code[i][k]
            code[i].remove(obj)
            code[i].insert(j, obj)
        return code

    def ha_sequence_machine_based_two_exchange(self, code=None):
        if code is None:
            self.std_code_machine_based()
            code = deepcopy(self.code)
        i = np.random.choice(range(self.schedule.m), 1, replace=False)[0]
        j, k = np.random.choice(range(len(code[i])), 2, replace=False)
        code[i][j], code[i][k] = code[i][k], code[i][j]
        return code

    def ha_sequence_machine_based_two_insert(self, code=None):
        if code is None:
            self.std_code_machine_based()
            code = deepcopy(self.code)
        i = np.random.choice(range(self.schedule.m), 1, replace=False)[0]
        j, k = np.random.choice(range(len(code[i])), 2, replace=False)
        if j > k:
            j, k = k, j
        obj = code[i][k]
        code[i].remove(obj)
        code[i].insert(j, obj)
        return code

    def ha_sequence_machine_based_all_machine_move(self, code=None):
        if code is None:
            self.std_code_machine_based()
            code = deepcopy(self.code)
        for i in range(self.schedule.m):
            for n_do in range(Utils.n_time(None)):
                j, k = np.random.choice(range(len(code[i])), 2, replace=False)
                if j > k:
                    j, k = k, j
                obj = code[i][k]
                code[i].remove(obj)
                code[i].insert(j, obj)
        return code

    def ha_sequence_machine_based_hybrid(self, n_time=None, func_list=None):
        self.std_code_machine_based()
        code = deepcopy(self.code)
        if func_list is None:
            func_list = [self.ha_sequence_machine_based_end_move, self.ha_sequence_machine_based_last_job_shift,
                         self.ha_sequence_machine_based_one_job_shift, self.ha_sequence_machine_based_two_exchange,
                         self.ha_sequence_machine_based_two_insert, self.ha_sequence_machine_based_all_machine_move, ]
        n_time = Utils.n_time(n_time)
        for n_do in range(n_time):
            func = np.random.choice(func_list, 1, replace=False)[0]
            code = func(code)
        return code

    def ts_sequence_machine_based(self, tabu_list, max_tabu):
        self.std_code_machine_based()
        code = deepcopy(self.code)
        for i in self.schedule.machine.keys():
            n_try = 0
            while n_try < max_tabu:
                n_try += 1
                try:
                    j, k = np.random.choice(range(len(code[i])), 2, replace=False)
                    if j > k:
                        j, k = k, j
                    a = {"m%s" % i, j, k}
                    if a not in tabu_list:
                        obj = code[i][k]
                        code[i].remove(obj)
                        code[i].insert(j, obj)
                        tabu_list.append(a)
                        break
                except ValueError:
                    pass
        return code

    def ts_sequence_operation_based(self, tabu_list, max_tabu, n_time=None):
        self.std_code()
        code = deepcopy(self.code)
        for n_do in range(Utils.n_time(n_time)):
            n_try = 0
            while n_try < max_tabu:
                n_try += 1
                try:
                    k = np.random.choice(list(self.schedule.machine.keys()), 1, replace=False)[0]
                    i, j = np.random.choice(self.schedule.machine[k].index_list, 2, replace=False)
                    tabu = {"m%s" % k, i, j}
                    if tabu not in tabu_list:
                        if i > j:
                            i, j = j, i
                        value = code[j]
                        obj = np.delete(code, j)
                        code = np.insert(obj, i, value)
                        tabu_list.append(tabu)
                        break
                except ValueError:
                    pass
        return code

    def ts_sequence_operation_based_insert1set(self, tabu_list, max_tabu):
        self.std_code()
        code = deepcopy(self.code)
        n_try = 0
        while n_try < max_tabu:
            n_try += 1
            try:
                k = np.random.choice(list(self.schedule.machine.keys()), 1, replace=False)[0]
                i, j = np.random.choice(self.schedule.machine[k].index_list, 2, replace=False)
                if i > j:
                    i, j = j, i
                job = code[j]
                index_job = self.schedule.job[job].index_list
                index_job.sort()
                index_list = [j, ]
                # [index_list.append(v) for v in index_job if v > j]
                [index_list.append(v) for v in index_job if v > j and np.random.random() < 0.5]
                tabu = {"m%s" % k, "i%s" % i}
                [tabu.add(v) for v in index_list]
                if tabu not in tabu_list:
                    for cur, g in enumerate(index_list):
                        code = np.delete(code, g - cur)
                    code = np.insert(code, i, [job] * len(index_list))
                    tabu_list.append(tabu)
                    break
            except ValueError:
                pass
        return code

    def ts_sequence_operation_based_insert2pre(self, tabu_list, max_tabu, n_time=None):
        self.std_code()
        code = deepcopy(self.code)
        for n_do in range(Utils.n_time(n_time)):
            n_try = 0
            while n_try < max_tabu:
                n_try += 1
                try:
                    a = np.random.choice(list(self.schedule.job.keys()), 1, replace=False)[0]
                    b = np.random.choice(range(self.schedule.job[a].nop - 1), 1, replace=False)[0]
                    if self.schedule.job[a].task[b].limited_wait != np.inf:
                        index_list = self.schedule.job[a].index_list
                        index_list.sort()
                        i, j = index_list[b], index_list[b + 1]
                        tabu = {"j%s" % a, i, j}
                        if tabu not in tabu_list:
                            value = code[j]
                            obj = np.delete(code, j)
                            code = np.insert(obj, i, value)
                            tabu_list.append(tabu)
                            break
                except ValueError:
                    pass
        return code

    def ts_sequence_operation_based_hybrid(self, tabu_list, max_tabu, func_list=None):
        if func_list is None:
            func_list = [self.ts_sequence_operation_based, self.ts_sequence_operation_based_insert1set,
                         self.ts_sequence_operation_based_insert2pre]
        func = np.random.choice(func_list, 1, replace=False)[0]
        return func(tabu_list, max_tabu)

    def ts_sequence_permutation(self, tabu_list, max_tabu, n_time=None):
        code = deepcopy(self.code)
        n_time = Utils.n_time(n_time)
        tabu = set()
        for n_do in range(n_time):
            n_try = 0
            while n_try < max_tabu:
                i, j = np.random.choice(range(self.schedule.n), 2, replace=False)
                a = {i, j}
                tabu = tabu | a
                if tabu not in tabu_list or (tabu in tabu_list and n_time - n_do == 2):
                    if i > j:
                        i, j = j, i
                    value = code[j]
                    obj = np.delete(code, j)
                    code = np.insert(obj, i, value)
                    break
                n_try += 1
                for val in a:
                    tabu.remove(val)
        tabu_list.append(tabu)
        return code

    def ts_assignment_job_based(self, tabu_list, max_tabu, n_time=None):
        mac = deepcopy(self.mac)
        n_time = Utils.n_time(n_time)
        tabu = set()
        for n_do in range(n_time):
            n_try = 0
            while n_try < max_tabu:
                try:
                    i = np.random.choice(list(self.schedule.job.keys()), 1, replace=False)[0]
                    j = np.random.choice(range(self.schedule.job[i].nop), 1, replace=False)[0]
                    k = np.random.choice([v for v in self.schedule.job[i].task[j].machine if v != mac[i][j]], 1,
                                         replace=False)[0]
                    a = {"j%s" % i, "s%s" % mac[i][j], "r%s" % k}
                    tabu = tabu | a
                    if tabu not in tabu_list or (tabu in tabu_list and n_time - n_do == 2):
                        mac[i][j] = k
                        break
                    n_try += 1
                    for val in a:
                        tabu.remove(val)
                except ValueError:
                    pass
        tabu_list.append(tabu)
        return mac

    def ts_route_job_based(self, tabu_list, max_tabu, n_time=None):
        route = deepcopy(self.route)
        n_time = Utils.n_time(n_time)
        tabu = set()
        for n_do in range(n_time):
            n_try = 0
            while n_try < max_tabu:
                i = np.random.choice(list(self.schedule.job.keys()), 1, replace=False)[0]
                j, k = np.random.choice(range(self.schedule.job[i].nop), 2, replace=False)
                a = {"j%s" % i, j, k}
                tabu = tabu | a
                if tabu not in tabu_list or (tabu in tabu_list and n_time - n_do == 2):
                    if j > k:
                        j, k = k, j
                    value = route[i][k]
                    obj = np.delete(route[i], k)
                    route[i] = np.insert(obj, j, value)
                    break
                n_try += 1
                for val in a:
                    tabu.remove(val)
        tabu_list.append(tabu)
        return route

    def similarity_assignment(self, info):
        a = 0
        for i in self.schedule.job.keys():
            for j in self.schedule.job[i].task.keys():
                if self.mac[i][j] == info.mac[i][j]:
                    a += 1
        return a / self.schedule.length

    def similarity_route(self, info):
        a = 0
        for i in self.schedule.job.keys():
            for j in self.schedule.job[i].task.keys():
                if self.route[i][j] == info.route[i][j]:
                    a += 1
        return a / self.schedule.length

    def similarity(self, info, a):
        if self.mac is not None and self.route is not None:
            b = self.similarity_assignment(info)
            c = self.similarity_route(info)
            return min([a, b, c])
        elif self.mac is not None:
            b = self.similarity_assignment(info)
            return min([a, b])
        elif self.route is not None:
            c = self.similarity_route(info)
            return min([a, c])
        return a

    def similarity_sequence_operation_based(self, info):
        a = 1 - np.count_nonzero(self.code - info.code) / self.schedule.length
        return self.similarity(info, a)

    def similarity_sequence_permutation(self, info):
        a = 1 - np.count_nonzero(self.code - info.code) / self.schedule.n
        return self.similarity(info, a)

    def ga_crossover_sequence_pox(self, info):
        code1 = deepcopy(self.code)
        code2 = deepcopy(info.code)
        a = np.random.choice(range(self.schedule.n), 1, replace=False)[0]
        b, c = np.argwhere(code1 != a)[:, 0], np.argwhere(code2 != a)[:, 0]
        code1[b], code2[c] = code2[c], code1[b]
        return code1, code2

    def ga_crossover_sequence_ipox(self, info):
        code1 = deepcopy(self.code)
        code2 = deepcopy(info.code)
        a = np.random.choice(self.schedule.n, 2, replace=False)
        b, c = [[], []], [self.schedule.job, info.schedule.job]
        for i in range(2):
            for j in a:
                b[i].extend(c[i][j].index_list)
        d = np.delete(range(self.schedule.length), b[0])
        e = np.delete(range(self.schedule.length), b[1])
        code1[d], code2[e] = code2[e], code1[d]
        return code1, code2

    def ga_crossover_sequence_dpox(self, info):
        code1 = deepcopy(self.code)
        code2 = deepcopy(info.code)
        index1 = np.zeros(self.schedule.length, dtype=int)
        index2 = np.zeros(self.schedule.length, dtype=int)
        for i in range(self.schedule.n):
            tmp1 = np.argwhere(code1 == i)[:, 0]
            tmp2 = np.argwhere(code2 == i)[:, 0]
            index1[tmp1] = range(tmp1.shape[0])
            index2[tmp2] = range(tmp2.shape[0])
        a = np.random.randint(int(0.2 * self.schedule.length), int(0.5 * self.schedule.length) + 1)
        b = np.random.choice(range(self.schedule.length), a, replace=False)
        c = []
        for i in b:
            c.append(list(set(np.argwhere(code2 == code1[i])[:, 0]) & set(np.argwhere(index2 == index1[i])[:, 0])))
        d, e = np.delete(range(self.schedule.length), b), np.delete(range(self.schedule.length), c)
        code1[d], code2[e] = code2[e], code1[d]
        return code1, code2

    def ga_crossover_sequence_ox(self, info):
        code1 = deepcopy(self.code)
        code2 = deepcopy(info.code)
        a, b = np.random.choice(range(1, self.schedule.length - 1), 2, replace=False)
        if a > b:
            a, b = b, a
        r_a_b = range(a, b + 1)
        left_a, right_b = range(a), range(b + 1, self.schedule.length)
        left_b_a = np.hstack([right_b, left_a])
        middle1, middle2 = code1[r_a_b], code2[r_a_b]
        number1, number2 = self.schedule.start_number_jme[1][r_a_b], info.schedule.start_number_jme[1][r_a_b]
        left1, left2 = code1[left_a], code2[left_a]
        right1, right2 = code1[right_b], code2[right_b]
        cycle1, cycle2 = np.hstack([right1, left1, middle1]), np.hstack([right2, left2, middle2])
        c_number1 = np.zeros(self.schedule.length, dtype=int)
        c_number2 = np.zeros(self.schedule.length, dtype=int)
        for i in self.schedule.job.keys():
            index1 = np.argwhere(cycle1 == i)[:, 0]
            index2 = np.argwhere(cycle2 == i)[:, 0]
            c_number1[index1] = range(index1.shape[0])
            c_number2[index2] = range(index2.shape[0])
        index_exchange1, index_exchange2 = [], []
        for i, (j, k) in enumerate(zip(middle1, middle2)):
            index_job1 = np.argwhere(cycle2 == j)[:, 0]
            index_number1 = np.argwhere(c_number2 == number1[i])[:, 0]
            index_exchange1.extend(list(set(index_job1) & set(index_number1)))
            index_job2 = np.argwhere(cycle1 == k)[:, 0]
            index_number2 = np.argwhere(c_number1 == number2[i])[:, 0]
            index_exchange2.extend(list(set(index_job2) & set(index_number2)))
        change1 = cycle2[np.delete(range(self.schedule.length), index_exchange1)]
        change2 = cycle1[np.delete(range(self.schedule.length), index_exchange2)]
        code1[left_b_a], code2[left_b_a] = change1, change2
        return code1, code2

    def ga_crossover_sequence_hybrid(self, info, func_list=None):
        if func_list is None:
            func_list = [self.ga_crossover_sequence_pox, self.ga_crossover_sequence_ipox,
                         self.ga_crossover_sequence_dpox, self.ga_crossover_sequence_ox]
        func = np.random.choice(func_list, 1, replace=False)[0]
        return func(info)

    def ga_crossover_sequence_permutation_pmx(self, info):
        code1 = deepcopy(self.code)
        code2 = deepcopy(info.code)
        a, b = np.random.choice(self.schedule.n, 2, replace=False)
        if a > b:
            a, b = b, a
        r_a_b = range(a, b)
        r_left = np.delete(range(self.schedule.n), r_a_b)
        left_1, left_2 = code2[r_left], code1[r_left]
        middle_1, middle_2 = code2[r_a_b], code1[r_a_b]
        code1[r_a_b], code2[r_a_b] = middle_2, middle_1
        mapping = [[], []]
        for i, j in zip(middle_1, middle_2):
            if j in middle_1 and i not in middle_2:
                index = np.argwhere(middle_1 == j)[0, 0]
                value = middle_2[index]
                while True:
                    if value in middle_1:
                        index = np.argwhere(middle_1 == value)[0, 0]
                        value = middle_2[index]
                    else:
                        break
                mapping[0].append(i)
                mapping[1].append(value)
            elif i in middle_2:
                pass
            else:
                mapping[0].append(i)
                mapping[1].append(j)
        for i, j in zip(mapping[0], mapping[1]):
            if i in left_1:
                left_1[np.argwhere(left_1 == i)[0, 0]] = j
            elif i in left_2:
                left_2[np.argwhere(left_2 == i)[0, 0]] = j
            if j in left_1:
                left_1[np.argwhere(left_1 == j)[0, 0]] = i
            elif j in left_2:
                left_2[np.argwhere(left_2 == j)[0, 0]] = i
        code1[r_left], code2[r_left] = left_1, left_2
        return code1, code2

    def ga_crossover_sequence_permutation_ox(self, info):
        code1 = deepcopy(self.code)
        code2 = deepcopy(info.code)
        a, b = np.random.choice(range(1, self.schedule.n - 1), 2, replace=False)
        if a > b:
            a, b = b, a
        r_a_b = range(a, b + 1)
        left_a, right_b = range(a), range(b + 1, self.schedule.n)
        left_b_a = np.hstack([right_b, left_a])
        middle1, middle2 = code1[r_a_b], code2[r_a_b]
        left1, left2 = code1[left_a], code2[left_a]
        right1, right2 = code1[right_b], code2[right_b]
        cycle1, cycle2 = np.hstack([right1, left1, middle1]), np.hstack([right2, left2, middle2])
        change1, change2 = [], []
        for i, j in zip(cycle1, cycle2):
            if j not in middle1:
                change1.append(j)
            if i not in middle2:
                change2.append(i)
        code1[left_b_a], code2[left_b_a] = change1, change2
        return code1, code2

    def ga_crossover_sequence_permutation_hybrid(self, info, func_list=None):
        if func_list is None:
            func_list = [self.ga_crossover_sequence_permutation_pmx, self.ga_crossover_sequence_permutation_ox]
        func = np.random.choice(func_list, 1, replace=False)[0]
        return func(info)

    def ga_crossover_assignment_job_based_random(self, info):
        mac1 = deepcopy(self.mac)
        mac2 = deepcopy(info.mac)
        for i, (p, q) in enumerate(zip(mac1, mac2)):
            for j, (u, v) in enumerate(zip(p, q)):
                if np.random.random() < 0.5:
                    mac1[i][j], mac2[i][j] = v, u
        return mac1, mac2

    def ga_crossover_route_pmx(self, info):
        route1 = deepcopy(self.route)
        route2 = deepcopy(info.route)
        for job in range(self.schedule.n):
            a, b = np.random.choice(self.schedule.job[job].nop, 2, replace=False)
            min_a_b, max_a_b = min([a, b]), max([a, b])
            r_a_b = range(min_a_b, max_a_b)
            r_left = np.delete(range(self.schedule.job[job].nop), r_a_b)
            left_1, left_2 = route1[job][r_left], route2[job][r_left]
            middle_1, middle_2 = route1[job][r_a_b], route2[job][r_a_b]
            route1[job][r_a_b], route2[job][r_a_b] = middle_2, middle_1
            mapping = [[], []]
            for i, j in zip(middle_1, middle_2):
                if j in middle_1 and i not in middle_2:
                    index = np.argwhere(middle_1 == j)[0, 0]
                    value = middle_2[index]
                    while True:
                        if value in middle_1:
                            index = np.argwhere(middle_1 == value)[0, 0]
                            value = middle_2[index]
                        else:
                            break
                    mapping[0].append(i)
                    mapping[1].append(value)
                elif i in middle_2:
                    pass
                else:
                    mapping[0].append(i)
                    mapping[1].append(j)
            for i, j in zip(mapping[0], mapping[1]):
                if i in left_1:
                    left_1[np.argwhere(left_1 == i)[0, 0]] = j
                elif i in left_2:
                    left_2[np.argwhere(left_2 == i)[0, 0]] = j
                if j in left_1:
                    left_1[np.argwhere(left_1 == j)[0, 0]] = i
                elif j in left_2:
                    left_2[np.argwhere(left_2 == j)[0, 0]] = i
            route1[job][r_left], route2[job][r_left] = left_1, left_2
        return route1, route2

    def ga_mutation_sequence_reverse(self, code=None):
        if code is None:
            code = deepcopy(self.code)
        return code[::-1]

    def ga_mutation_sequence_substring_rand(self, code=None):
        if code is None:
            code = deepcopy(self.code)
        a = np.random.choice(range(len(code)), 4, replace=False).tolist()
        a.sort()
        b = [code[0:a[0]], code[a[0]:a[1]], code[a[1]:a[2]], code[a[2]:a[3]], code[a[3]:]]
        np.random.shuffle(b)
        return np.hstack(b)

    def ga_mutation_sequence_substring_insert(self, code=None):
        if code is None:
            code = deepcopy(self.code)
        a = np.random.choice(range(len(code)), 3, replace=False).tolist()
        a.sort()
        b = [code[0:a[0]], code[a[0]:a[1]], code[a[1]:a[2]], code[a[2]:]]
        val = b[2]
        b.pop(2)
        b.insert(a[0], val)
        return np.hstack(b)

    def ga_mutation_sequence_operation_based_tpe(self, code=None, n_time=None):
        if code is None:
            code = deepcopy(self.code)
        for n_do in range(Utils.n_time(n_time)):
            a = np.random.choice(range(self.schedule.m), 1, replace=False)[0]
            try:
                b = np.random.choice(self.schedule.machine[a].index_list, 2, replace=False)
                code[b] = code[b[::-1]]
            except ValueError:
                pass
        return code

    def ga_mutation_sequence_operation_based_insert(self, code=None, n_time=None):
        if code is None:
            code = deepcopy(self.code)
        for n_do in range(Utils.n_time(n_time)):
            try:
                a, b = np.random.choice(range(self.schedule.length), 2, replace=False)
                if a > b:
                    a, b = b, a
                val = code[b]
                obj = np.delete(code, b)
                code = np.insert(obj, a, val)
            except ValueError:
                code = code[::-1]
        return code

    def ga_mutation_sequence_operation_based_oji(self, code=None):
        if code is None:
            code = deepcopy(self.code)
        a = np.random.choice(list(self.schedule.job.keys()), 1, replace=False)[0]
        for operation, i in enumerate(self.schedule.job[a].index_list):
            if self.mac is None:
                b = self.schedule.job[a].task[operation].machine
            else:
                b = self.mac[a][operation]
            machine_index = self.schedule.machine[b].index_list
            c = [j for j in machine_index if j < i]
            try:
                j = np.random.choice(c, 1, replace=False)[0]
                value = code[j]
                obj = np.delete(code, j)
                code = np.insert(obj, i, value)
            except ValueError:
                pass
        return code

    def ga_mutation_sequence_operation_based_krm(self, code=None, index=None):
        if code is None:
            code = deepcopy(self.code)
        if index is None:
            self.std_code()
            index = []
            index = self.key_route(index)
        for i in index:
            a = self.schedule.start_number_jme[3][i]
            b = [val for val in self.schedule.machine[a].index_list if val < i]
            try:
                j = np.random.choice(b, 1, replace=False)[0]
                if i > j:
                    i, j = j, i
                value = code[j]
                obj = np.delete(code, j)
                code = np.insert(obj, i, value)
            except ValueError:
                pass
        return code

    def ga_mutation_sequence_operation_based_dpm(self, code=None):
        if code is None:
            code = deepcopy(self.code)
        n_g = np.random.randint(int(0.15 * self.schedule.length), int(0.35 * self.schedule.length) + 1)
        index = np.random.choice(range(self.schedule.length), n_g, replace=False)
        for i in index:
            a = self.schedule.start_number_jme[3][i]
            b = [val for val in self.schedule.machine[a].index_list if val < i]
            try:
                j = np.random.choice(b, 1, replace=False)[0]
                if i > j:
                    i, j = j, i
                value = code[j]
                obj = np.delete(code, j)
                code = np.insert(obj, i, value)
            except ValueError:
                pass
        return code

    def ga_mutation_sequence_operation_based_insert1set(self, code=None):
        if code is None:
            self.std_code()
            code = deepcopy(self.code)
        while True:
            try:
                k = np.random.choice(list(self.schedule.machine.keys()), 1, replace=False)[0]
                i, j = np.random.choice(self.schedule.machine[k].index_list, 2, replace=False)
                if i > j:
                    i, j = j, i
                job = code[j]
                index_job = self.schedule.job[job].index_list
                index_job.sort()
                index_list = [j, ]
                [index_list.append(v) for v in index_job if v > j]
                for cur, g in enumerate(index_list):
                    code = np.delete(code, g - cur)
                code = np.insert(code, i, [job] * len(index_list))
                break
            except ValueError:
                pass
        return code

    def ga_mutation_sequence_operation_based_hybrid(self, func_list=None):
        code = deepcopy(self.code)
        if func_list is None:
            func_list = [self.ga_mutation_sequence_operation_based_tpe,
                         self.ga_mutation_sequence_operation_based_insert]
        func = np.random.choice(func_list, 1, replace=False)[0]
        return func(code)

    def ga_mutation_sequence_permutation_tpe(self, code=None, n_time=None):
        if code is None:
            code = deepcopy(self.code)
        for n_do in range(Utils.n_time(n_time)):
            a = np.random.choice(range(self.schedule.n), 2, replace=False)
            code[a] = code[a[::-1]]
        return code

    def ga_mutation_sequence_permutation_insert(self, code=None, n_time=None):
        if code is None:
            code = deepcopy(self.code)
        for n_do in range(Utils.n_time(n_time)):
            try:
                b, c = np.random.choice(range(self.schedule.n), 2, replace=False)
                if b > c:
                    b, c = c, b
                val = code[c]
                obj = np.delete(code, c)
                code = np.insert(obj, b, val)
            except ValueError:
                code = code[::-1]
        return code

    def ga_mutation_sequence_permutation_hybrid(self, n_time=None, func_list=None):
        code = deepcopy(self.code)
        n_time = Utils.n_time(n_time)
        if func_list is None:
            func_list = [self.ga_mutation_sequence_permutation_tpe, self.ga_mutation_sequence_permutation_insert,
                         self.ga_mutation_sequence_reverse]
        for n_do in range(n_time):
            func = np.random.choice(func_list, 1, replace=False)[0]
            code = func(code)
        return code

    def ga_mutation_assignment_job_based_random_replace(self, tech):
        mac = deepcopy(self.mac)
        a = np.random.choice(self.schedule.n, np.random.randint(1, self.schedule.n), replace=False)
        for i in a:
            for j, k in enumerate(mac[i]):
                if np.random.random() < 0.5:
                    try:
                        mac[i][j] = np.random.choice([v for v in tech[i][j] if v != k], 1, replace=False)[0]
                    except ValueError:
                        pass
        return mac

    def ga_mutation_route_tpe(self):
        route = deepcopy(self.route)
        a = np.random.choice(list(self.schedule.job.keys()), np.random.randint(1, self.schedule.n, 1)[0], replace=False)
        for i in a:
            b = np.random.choice(range(self.schedule.job[i].nop), 2, replace=False)
            route[i][b] = route[i][b[::-1]]
        return route

    @staticmethod
    def repair_random_key(code):
        for i, j in enumerate(code):
            if j < 0:
                code[i] = 0
            if j > 1:
                code[i] = 1
        return code

    def de_mutation_sequence_rand1(self, f, info2, info3):
        code1 = self.code
        code2 = info2.code
        code3 = info3.code
        new = code1 + f * (code2 - code3)
        return self.repair_random_key(new)

    def de_mutation_sequence_best1(self, f, info2, info_best):
        code1 = self.code
        code2 = info2.code
        code_best = info_best.code
        new = code_best + f * (code1 - code2)
        return self.repair_random_key(new)

    def de_mutation_sequence_c2best1(self, f, info2, info_best):
        code1 = self.code
        code2 = info2.code
        code_best = info_best.code
        new = code1 + f * (code_best - code1) + f * (code1 - code2)
        return self.repair_random_key(new)

    def de_mutation_sequence_best2(self, f, info2, info3, info4, info_best):
        code1 = self.code
        code2 = info2.code
        code3 = info3.code
        code4 = info4.code
        code_best = info_best.code
        new = code_best + f * (code1 - code2) + f * (code3 - code4)
        return self.repair_random_key(new)

    def de_mutation_sequence_rand2(self, f, info2, info3, info4, info5):
        code1 = self.code
        code2 = info2.code
        code3 = info3.code
        code4 = info4.code
        code5 = info5.code
        new = code1 + f * (code2 - code3) + f * (code4 - code5)
        return self.repair_random_key(new)

    def de_mutation_sequence_hybrid(self, f, info2, info3, info4, info5, info_best):
        a = np.random.random()
        if a < 0.2:
            return self.de_mutation_sequence_rand1(f, info2, info3)
        elif a < 0.4:
            return self.de_mutation_sequence_best1(f, info2, info_best)
        elif a < 0.6:
            return self.de_mutation_sequence_c2best1(f, info2, info_best)
        elif a < 0.8:
            return self.de_mutation_sequence_best2(f, info2, info3, info4, info_best)
        return self.de_mutation_sequence_rand2(f, info2, info3, info4, info5)

    def de_crossover_sequence_normal(self, cr, info2):
        code1 = deepcopy(self.code)
        code2 = deepcopy(info2.code)
        for i, (j, k) in enumerate(zip(code1, code2)):
            if np.random.random() < cr:
                code1[i], code2[i] = k, j
        return code1, code2

    def jaya_sequence_classic(self, best, worst):
        a, b = np.random.random(2)
        code = np.abs(self.code)
        new = self.code + a * (best.code - code) - b * (worst.code - code)
        return self.repair_random_key(new)

    def jaya_sequence_rand(self, best, worst, rand):
        a, b, c = np.random.random(3)
        code = np.abs(self.code)
        new = self.code + a * (best.code - code) - b * (worst.code - code) + c * (rand.code - code)
        return self.repair_random_key(new)

    def jaya_sequence_hybrid(self, best, worst, rand):
        p = np.random.random()
        if p < 0.5:
            return self.jaya_sequence_classic(best, worst)
        else:
            return self.jaya_sequence_rand(best, worst, rand)

    def method_sequence_key_block_move(self, block=None):
        self.std_code()
        code = deepcopy(self.code)
        if block is None:
            block = self.key_block()
        for i, j in block.items():
            if j.shape[0] >= 2:
                head, tail = j[0], j[-1]
                if np.random.random() < 0.5:
                    index = np.random.choice(j[1:], 1, replace=False)[0]
                    if np.random.random() < 0.5:
                        value = code[head]
                        obj = np.delete(code, head)
                        code = np.insert(obj, index - 1, value)
                        code[index], code[index - 1] = code[index - 1], code[index]
                    else:
                        code[head], code[index] = code[index], code[head]
                else:
                    index = np.random.choice(j[:-1], 1, replace=False)[0]
                    if np.random.random() < 0.5:
                        value = code[tail]
                        obj = np.delete(code, tail)
                        code = np.insert(obj, index, value)
                    else:
                        code[tail], code[index] = code[index], code[tail]
        return code

    def method_sequence_key_block_move_mac(self, block=None):
        mac = deepcopy(self.mac)
        if block is None:
            block = self.key_block()
        for i, j in block.items():
            for g in j:
                if np.random.random() < 0.5:
                    try:
                        a, b = self.schedule.start_number_jme[2][g], self.schedule.start_number_jme[1][g]
                        c = [v for v in self.schedule.job[a].task[b].machine if v != mac[a][b]]
                        mac[a][b] = np.random.choice(c, 1, replace=False)[0]
                    except ValueError:
                        pass
        return mac

    def dislocation(self):
        code = deepcopy(self.code)
        return np.hstack([code[1:], code[0]])
