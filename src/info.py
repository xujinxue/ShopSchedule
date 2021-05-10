import copy
import datetime
import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly as py
import plotly.figure_factory as ff
from matplotlib import colors as mcolors

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
[COLORS.pop(j - i) for i, j in enumerate(range(12))]
[COLORS.pop(j - i) for i, j in enumerate([6, ])]
BLOCK_COLORS = ["dimgray", "darkred", "darkorange", "darkgoldenrod", "darkgreen",
                "darkblue", "darkmagenta", "darkviolet", "crimson", "indigo",
                "darkcyan", "springgreen", "firebrick", "chocolate", "darkgoldenrod"
                ]
LEN_BLOCK_COLORS = len(BLOCK_COLORS)
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


class Node:
    def __init__(self, index):
        self.index = index


class GanttChart:
    def __init__(self, file=None, schedule=None, mac=None):
        self.schedule = schedule
        self.mac = mac
        if file is not None:
            from .shop.schedule import Schedule
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
                    self.schedule.sjike[i] = np.append(self.schedule.sjike[i], val)
                self.schedule.job[job].add_task(machine=machine, duration=duration, name=operation, index=operation)
                self.schedule.job[job].task[operation].start = start
                self.schedule.job[job].task[operation].end = end
                self.schedule.job[job].index_list.append(g)
                self.schedule.machine[machine].index_list.append(g)
                if end > self.schedule.machine[machine].end:
                    self.schedule.machine[machine].end = end

    def not_dummy(self, i):
        return True if self.schedule.sjike[0][i] != self.schedule.sjike[4][i] else False

    def key_route(self):
        critical_path = []
        node_list = []
        job_end = {}
        machine_end = {}
        for i in self.schedule.job.keys():
            job_end[i] = self.schedule.sjike[4][self.schedule.job[i].index_list]
        for i in self.schedule.machine.keys():
            machine_end[i] = self.schedule.sjike[4][self.schedule.machine[i].index_list]
        a = np.argwhere(self.schedule.sjike[4] == self.schedule.makespan)[:, 0]
        if a.shape[0] > 1:
            for i in a:
                node_list.append(Node([i]))
        else:
            node_list.append(Node([a[0]]))
        while len(node_list):
            while True:
                index = node_list[0].index
                b = index[-1]
                c = self.schedule.sjike[0][b]
                d = self.schedule.sjike[2][b]
                e = self.schedule.sjike[3][b]
                try:
                    f = self.schedule.job[d].index_list[np.argwhere(job_end[d] == c)[:, 0][0]]
                except IndexError:
                    f = None
                try:
                    g = self.schedule.machine[e].index_list[np.argwhere(machine_end[e] == c)[:, 0][0]]
                except IndexError:
                    g = None
                if f is not None and self.not_dummy(f) and g is not None and self.not_dummy(g):
                    index_f, index_g = deepcopy(index), deepcopy(index)
                    index_f.append(f)
                    index_g.append(g)
                    node_list.append(Node(index_f))
                    node_list.append(Node(index_g))
                    node_list.pop(0)
                elif f is not None and self.not_dummy(f) and g is None:
                    node_list[0].index.append(f)
                elif f is None and g is not None and self.not_dummy(g):
                    node_list[0].index.append(g)
                else:
                    critical_path.extend(node_list[0].index)
                    node_list.pop(0)
                    break
        return critical_path

    def key_block(self):
        index = list(set(self.key_route()))
        index_start = self.schedule.sjike[0][index]
        index = [index[i] for i in np.argsort(-index_start)]
        a = self.schedule.sjike[3][index]
        b = set(a)
        c = 0
        block = {}
        for i in b:
            block[c] = np.array([], dtype=int)
            d = np.argwhere(a == i)[:, 0]
            start = self.schedule.sjike[0][[index[j] for j in d]].tolist()
            for cur, j in enumerate(d):
                g = index[j]
                try:
                    end = self.schedule.sjike[4][g]
                    start.index(end)
                except ValueError:
                    if cur != 0:
                        c += 1
                        block[c] = np.array([], dtype=int)
                self.schedule.job[self.schedule.sjike[2][g]].task[
                    self.schedule.sjike[1][g]].block = c
                block[c] = np.append(block[c], g)
            c += 1
        return block

    def gantt_chart_png(self, filename="GanttChart", fig_width=9, fig_height=5, random_colors=False, lang=1, dpi=200,
                        height=0.8, scale_more=None, x_step=None, y_based=0, text_rotation=0,
                        with_operation=True, with_start_end=False, key_block=False, show=False):
        if random_colors:
            random.shuffle(COLORS)
        if key_block:  # JSP,FJSP
            self.key_block()
        plt.figure(figsize=[fig_width, fig_height])
        plt.yticks(range([self.schedule.m, self.schedule.n][y_based]),
                   range(1, [self.schedule.m, self.schedule.n][y_based] + 1))
        plt.xticks([], [])
        scale_more = 12 if scale_more is None else scale_more
        x_step = max([1, self.schedule.makespan // 10 if x_step is None else x_step])
        ax = plt.gca()
        for job in self.schedule.job.values():
            for task in job.task.values():
                if task.start != task.end:
                    if self.mac is None:
                        machine = task.machine
                    else:
                        machine = self.mac[job.index][task.index]
                    y_values = [machine, job.index]
                    y = y_values[y_based]
                    width = task.end - task.start
                    left = [task.start, self.schedule.makespan - task.end][self.schedule.direction]
                    edgecolor, linewidth = "black", 0.5
                    if task.block is not None:
                        edgecolor, linewidth = BLOCK_COLORS[task.block % LEN_BLOCK_COLORS], 2
                    plt.barh(
                        y=y, width=width,
                        left=left, color=COLORS[y_values[y_based - 1] % LEN_COLORS],
                        edgecolor=edgecolor, linewidth=linewidth,
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
                plt.barh(0, 0, color=COLORS[job.index % LEN_COLORS], label=job.index + 1)
            plt.barh(y=0, width=self.schedule.makespan / scale_more, left=self.schedule.makespan, color="white")
            if lang == 0:
                title = r"${Job}$"
            else:
                title = "工件"
            plt.legend(loc="best", title=title)
        if y_based == 1:
            for machine in self.schedule.machine.values():
                plt.barh(0, 0, color=COLORS[machine.index % LEN_COLORS], label=machine.index + 1)
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

    def gantt_chart_html(self, filename="GanttChart", date=None, show=False, lang=1):
        if date is None:
            today = dt.today()
            date = dt(today.year, today.month, today.day)
        else:
            tmp = list(map(int, date.split()))
            date = dt(tmp[0], tmp[1], tmp[2])
        df = []
        for job in self.schedule.job.values():
            for task in job.task.values():
                if task.start != task.end:
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


class Info(GanttChart):
    def __init__(self, schedule, code, mac=None, route=None):
        self.schedule = deepcopy(schedule)
        self.code = code
        self.mac = mac
        self.route = route
        GanttChart.__init__(self, schedule=self.schedule, mac=self.mac)

    def trans_operation_based2machine_based(self):  # 转码：基于工序的编码->基于机器的编码
        code = [[] for _ in self.schedule.machine.keys()]
        for i, machine in self.schedule.machine.items():
            job = self.schedule.sjike[2][machine.index_list]
            operation = self.schedule.sjike[1][machine.index_list]
            for a, b in zip(job, operation):
                code[i].append((a, b))
        return code

    def std_code(self, std_direction=None):
        if std_direction not in [0, 1]:
            std_direction = Utils.direction()
        if self.schedule.direction == 0:
            if std_direction == 0:
                index = np.argsort(self.schedule.sjike[0])
            else:
                index = np.argsort(-self.schedule.sjike[0])[::-1]
        else:
            if std_direction == 0:
                index = np.argsort(self.schedule.sjike[0])[::-1]
            else:
                index = np.argsort(-self.schedule.sjike[0])
        self.code = self.code[index]
        for i in self.schedule.job.keys():
            self.schedule.job[i].nd = 0
            self.schedule.job[i].index_list = [None for _ in range(self.schedule.job[i].nop)]
        for i in self.schedule.machine.keys():
            self.schedule.machine[i].index_list = []
        try:
            for g, i in enumerate(self.code):
                u = self.schedule.job[i].nd
                if self.schedule.direction == 0:
                    j = u
                else:
                    j = self.schedule.job[i].nop - u - 1
                if self.mac is None:
                    k = self.schedule.job[i].task[j].machine
                else:
                    k = self.mac[i][j]
                self.schedule.job[i].index_list[j] = g
                self.schedule.machine[k].index_list.append(g)
                for u, v in enumerate([self.schedule.job[i].task[j].start, j, i, k, self.schedule.job[i].task[j].end]):
                    self.schedule.sjike[u][g] = v
                self.schedule.job[i].nd += 1
        except KeyError:
            code = self.schedule.trans_random_key2operation_based(self.code)
            for g, i in enumerate(code):
                u = self.schedule.job[i].nd
                if self.schedule.direction == 0:
                    j = u
                else:
                    j = self.schedule.job[i].nop - u - 1
                if self.mac is None:
                    k = self.schedule.job[i].task[j].machine
                else:
                    k = self.mac[i][j]
                self.schedule.job[i].index_list[j] = g
                self.schedule.machine[k].index_list.append(g)
                for u, v in enumerate([self.schedule.job[i].task[j].start, j, i, k, self.schedule.job[i].task[j].end]):
                    self.schedule.sjike[u][g] = v
                self.schedule.job[i].nd += 1

    def std_code_machine_based(self):
        for k in self.schedule.machine.keys():
            start = []
            for i, j in self.code[k]:
                start.append(self.schedule.job[i].task[j].start)
            self.code[k] = [self.code[k][u] for u in np.argsort(start)]

    def save_code_to_txt(self, file):
        try:
            Utils.save_code_to_txt(file, {"code": self.code.tolist(), "route": self.route, "mac": self.mac})
        except AttributeError:
            Utils.save_code_to_txt(file, {"code": self.code, "route": self.route, "mac": self.mac})

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
                        duration = task.duration[task.machine.index(machine)]
                    f.writelines("{},{},{},{},{},{}\n".format(
                        job.index + 1, task.index + 1, machine + 1, task.start, duration, task.end))

    """"
    =============================================================================
    Genetic operator: operation based code
    =============================================================================
    """

    def ga_crossover_sequence_pox(self, info):
        code1 = deepcopy(self.code)
        code2 = deepcopy(info.code)
        a = np.random.choice(range(self.schedule.n), 1, replace=False)[0]
        b, c = np.argwhere(code1 != a)[:, 0], np.argwhere(code2 != a)[:, 0]
        code1[b], code2[c] = code2[c], code1[b]
        return code1, code2

    def ga_crossover_sequence_mox(self, info):
        code1 = deepcopy(self.code)
        code2 = deepcopy(info.code)
        a = np.random.choice(range(self.schedule.m), 1, replace=False)[0]
        b, c = self.schedule.machine[a].index_list, info.schedule.machine[a].index_list
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

    def ga_crossover_sequence_ox(self, info, a=None, b=None):
        code1 = deepcopy(self.code)
        code2 = deepcopy(info.code)
        if a is None or b is None:
            a, b = np.random.choice(range(1, self.schedule.length - 1), 2, replace=False)
        if a > b:
            a, b = b, a
        r_a_b = range(a, b + 1)
        left_a, right_b = range(a), range(b + 1, self.schedule.length)
        left_b_a = np.hstack([right_b, left_a])
        middle1, middle2 = code1[r_a_b], code2[r_a_b]
        number1, number2 = self.schedule.sjike[1][r_a_b], info.schedule.sjike[1][r_a_b]
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

    def ga_crossover_sequence_pmx(self, info):
        code1 = deepcopy(self.code)
        code2 = deepcopy(info.code)
        number1 = self.schedule.sjike[1]
        number2 = info.schedule.sjike[1]
        a, b = np.random.choice(range(1, self.schedule.length - 1), 2, replace=False)
        if a > b:
            a, b = b, a
        r_a_b = range(a, b + 1)
        r_left = np.delete(range(self.schedule.length), r_a_b)
        middle_1 = [(code1[i], number1[i]) for i in r_a_b]
        middle_2 = [(code2[i], number2[i]) for i in r_a_b]
        left_1 = [(code1[i], number1[i]) for i in r_left]
        left_2 = [(code2[i], number2[i]) for i in r_left]
        code1[r_a_b], code2[r_a_b] = code2[r_a_b], code1[r_a_b]
        mapping = [[], []]
        for i, j in zip(middle_1, middle_2):
            if j in middle_1 and i not in middle_2:
                index = middle_1.index(j)
                value = middle_2[index]
                while True:
                    if value in middle_1:
                        index = middle_1.index(value)
                        value = middle_2[index]
                    else:
                        break
                if i[0] != value[0]:
                    mapping[0].append(i)
                    mapping[1].append(value)
            elif j not in middle_1 and i not in middle_2:
                mapping[0].append(i)
                mapping[1].append(j)
        for i, j in zip(mapping[0], mapping[1]):
            if i in left_1:
                left_1[left_1.index(i)] = j
            elif i in left_2:
                left_2[left_2.index(i)] = j
            if j in left_1:
                left_1[left_1.index(j)] = i
            elif j in left_2:
                left_2[left_2.index(j)] = i
        code1[r_left], code2[r_left] = [val[0] for val in left_1], [val[0] for val in left_2]
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

    def ga_mutation_sequence_tpe(self, length=None):
        code = deepcopy(self.code)
        length = self.schedule.length if length is None else length
        while True:
            if length == self.schedule.length:
                try:
                    a = np.random.randint(0, self.schedule.m, 1)[0]
                    b = self.schedule.machine[a].index_list
                    c = np.random.choice(b, 2, replace=False)
                    if code[c[0]] != code[c[1]]:
                        code[c] = code[c[::-1]]
                        break
                except ValueError:
                    pass
            else:
                a = np.random.choice(range(length), 2, replace=False)
                if code[a[0]] != code[a[1]]:
                    code[a] = code[a[::-1]]
                    break
        return code

    def ga_mutation_sequence_insert(self, length=None):
        code = deepcopy(self.code)
        length = self.schedule.length if length is None else length
        a, b = np.random.choice(range(length), 2, replace=False)
        if a > b:
            a, b = b, a
        if np.random.random() < 0.5:
            c = np.delete(code, b)
            code = np.insert(c, a, code[b])
        else:
            c = np.delete(code, a)
            code = np.insert(c, b - 1, code[a])
            code[b], code[b - 1] = code[b - 1], code[b]
        return code

    def ga_mutation_sequence_sr(self, length=None):
        code = deepcopy(self.code)
        length = self.schedule.length if length is None else length
        a, b = np.random.choice(range(length), 2, replace=False)
        if a > b:
            a, b = b, a
        c = range(a, b + 1)
        code[c] = code[c[::-1]]
        return code

    """"
    =============================================================================
    Genetic operator: permutation based code
    =============================================================================
    """

    def ga_crossover_sequence_permutation_pmx(self, info):
        code1 = deepcopy(self.code)
        code2 = deepcopy(info.code)
        a, b = np.random.choice(self.schedule.n, 2, replace=False)
        if a > b:
            a, b = b, a
        r_a_b = range(a, b + 1)
        r_left = np.delete(range(self.schedule.n), r_a_b)
        middle_1, middle_2 = code1[r_a_b], code2[r_a_b]
        left_1, left_2 = code1[r_left], code2[r_left]
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
            elif j not in middle_1 and i not in middle_2:
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

    def ga_mutation_sequence_permutation_tpe(self):
        return self.ga_mutation_sequence_tpe(self.schedule.n)

    def ga_mutation_sequence_permutation_insert(self):
        return self.ga_mutation_sequence_insert(self.schedule.n)

    def ga_mutation_sequence_permutation_sr(self):
        return self.ga_mutation_sequence_sr(self.schedule.n)

    """"
    =============================================================================
    Genetic operator: machine assignment problem
    =============================================================================
    """

    def ga_crossover_assignment(self, info):
        mac1 = deepcopy(self.mac)
        mac2 = deepcopy(info.mac)
        for i, (p, q) in enumerate(zip(mac1, mac2)):
            for j, (u, v) in enumerate(zip(p, q)):
                if np.random.random() < 0.5:
                    mac1[i][j], mac2[i][j] = v, u
        return mac1, mac2

    def ga_mutation_assignment(self, tech):
        mac = deepcopy(self.mac)
        for i in range(self.schedule.n):
            for j, k in enumerate(mac[i]):
                if np.random.random() < 0.5:
                    try:
                        a = [v for v in tech[i][j] if v != k]
                        mac[i][j] = np.random.choice(a, 1, replace=False)[0]
                    except ValueError:
                        pass
        return mac

    """"
    =============================================================================
    Genetic operator: route assignment problem
    =============================================================================
    """

    def ga_crossover_route(self, info):
        route1 = deepcopy(self.route)
        route2 = deepcopy(info.route)
        for i, (p, q) in enumerate(zip(route1, route2)):
            route1[i], route2[i] = q, p
        return route1, route2

    def ga_mutation_route(self):
        route = deepcopy(self.route)
        for i in range(self.schedule.n):
            try:
                a = list(range(self.schedule.job[i].nor))
                a.remove(route[i])
                route[i] = np.random.choice(a, 1, replace=False)[0]
            except ValueError:
                pass
        return route

    """"
    =============================================================================
    Genetic operator: match mac and route
    =============================================================================
    """

    def repair_mac_route(self, mac, route):
        for i, j in enumerate(mac):
            for u, v in enumerate(j):
                task = self.schedule.job[i].route[route[i]].task[u]
                if v not in task.machine:
                    mac[i][j] = task.machine[task.duration.index(min(task.duration))]

    """"
    =============================================================================
    Tabu search
    =============================================================================
    """

    @staticmethod
    def do_tabu_search(code, i, j, w):
        if i > j:
            i, j = j, i
        if w == 0:
            obj = np.delete(code, j)
            code = np.insert(obj, i, code[j])
        elif w == 1:
            obj = np.delete(code, i)
            code = np.insert(obj, j - 1, code[i])
            code[j], code[j - 1] = code[j - 1], code[j]
        else:
            code[i], code[j] = code[j], code[i]
        return code

    def ts_sequence_operation_based(self, tabu_list, max_tabu):
        self.std_code()
        code = deepcopy(self.code)
        n_try = 0
        while n_try < max_tabu:
            n_try += 1
            try:
                k = np.random.choice(list(self.schedule.machine.keys()), 1, replace=False)[0]
                i, j = np.random.choice(self.schedule.machine[k].index_list, 2, replace=False)
                w = np.random.choice(range(3), 1, replace=False)[0]
                tabu = {"machine-%s" % k, "way-%s" % w, i, j}
                if tabu not in tabu_list:
                    tabu_list.append(tabu)
                    code = self.do_tabu_search(code, i, j, w)
                    break
            except ValueError:
                pass
        return code

    def ts_sequence_permutation_based(self, tabu_list, max_tabu):
        code = deepcopy(self.code)
        n_try = 0
        while n_try < max_tabu:
            n_try += 1
            try:
                i, j = np.random.choice(self.schedule.n, 2, replace=False)
                w = np.random.choice(range(3), 1, replace=False)[0]
                tabu = {"way-%s" % w, i, j}
                if tabu not in tabu_list:
                    tabu_list.append(tabu)
                    code = self.do_tabu_search(code, i, j, w)
                    break
            except ValueError:
                pass
        return code

    """"
    =============================================================================
    Local operator
    =============================================================================
    """

    def key_block_move(self, block=None):
        self.std_code()
        code = deepcopy(self.code)
        if block is None:
            block = self.key_block()
        for i, j in block.items():
            if j.shape[0] >= 2:
                head, tail = j[0], j[-1]
                if np.random.random() < 0.5:
                    index = np.random.choice(j[1:], 1, replace=False)[0]
                    if np.random.random() < 0.8:
                        value = code[head]
                        obj = np.delete(code, head)
                        code = np.insert(obj, index - 1, value)
                        code[index], code[index - 1] = code[index - 1], code[index]
                    else:
                        code[head], code[index] = code[index], code[head]
                else:
                    index = np.random.choice(j[:-1], 1, replace=False)[0]
                    if np.random.random() < 0.8:
                        value = code[tail]
                        obj = np.delete(code, tail)
                        code = np.insert(obj, index, value)
                    else:
                        code[tail], code[index] = code[index], code[tail]
        return code

    def key_block_move_mac(self, block=None):
        mac = deepcopy(self.mac)
        if block is None:
            block = self.key_block()
        for i, j in block.items():
            for g in j:
                if np.random.random() < 0.5:
                    try:
                        a, b = self.schedule.sjike[2][g], self.schedule.sjike[1][g]
                        c = [v for v in self.schedule.job[a].task[b].machine if v != mac[a][b]]
                        mac[a][b] = np.random.choice(c, 1, replace=False)[0]
                    except ValueError:
                        pass
        return mac

    """"
    =============================================================================
    Other operator
    =============================================================================
    """

    def dislocation_operator(self):
        code = deepcopy(self.code)
        return np.hstack([code[1:], code[0]])
