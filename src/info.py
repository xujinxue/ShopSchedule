import copy
import datetime
import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly as py
import plotly.figure_factory as ff
from matplotlib import colors as mcolors

from .define import Crossover, Mutation
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
[COLORS.pop(j - i) for i, j in enumerate([6, 10, ])]
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
    def __init__(self, file=None, schedule=None, mac=None, wok=None):
        self.schedule = schedule
        self.mac = mac
        self.wok = wok
        if file is not None:
            from .shop.schedule import Schedule
            self.data = pd.read_csv(file)
            self.n = max(self.data.loc[:, "Job"])
            self.m = max(self.data.loc[:, "Machine"])
            self.w = max(self.data.loc[:, "Worker"])
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
            try:
                for i in range(self.w):
                    self.schedule.add_worker(name=i, index=i)
                self.wok = [[] for _ in range(self.n)]
            except TypeError:
                self.w = None
            for g, (start, operation, job, machine, end, duration, worker) in enumerate(zip(
                    self.data.loc[:, "Start"], self.data.loc[:, "Operation"], self.data.loc[:, "Job"],
                    self.data.loc[:, "Machine"], self.data.loc[:, "End"], self.data.loc[:, "Duration"],
                    self.data.loc[:, "Worker"])):
                job, operation, machine = job - 1, operation - 1, machine - 1
                if self.w is not None:
                    worker -= 1
                    self.wok[job].append(worker)
                    self.schedule.worker[worker].index_list.append(g)
                for i, val in enumerate([start, operation, job, machine, end, worker]):
                    self.schedule.sjikew[i] = np.append(self.schedule.sjikew[i], val)
                self.schedule.job[job].add_task(machine=machine, duration=duration, name=operation, index=operation)
                self.schedule.job[job].task[operation].start = start
                self.schedule.job[job].task[operation].end = end
                self.schedule.job[job].index_list.append(g)
                self.schedule.machine[machine].index_list.append(g)
                if end > self.schedule.machine[machine].end:
                    self.schedule.machine[machine].end = end

    def not_dummy(self, i):
        return True if self.schedule.sjikew[0][i] != self.schedule.sjikew[4][i] else False

    def get_job_end(self):
        job_end = {}
        for i in self.schedule.job.keys():
            job_end[i] = self.schedule.sjikew[4][self.schedule.job[i].index_list]
        return job_end

    def get_machine_end(self):
        machine_end = {}
        for i in self.schedule.machine.keys():
            machine_end[i] = self.schedule.sjikew[4][self.schedule.machine[i].index_list]
        return machine_end

    def get_worker_end(self):
        worker_end = {}
        for i in self.schedule.worker.keys():
            worker_end[i] = self.schedule.sjikew[4][self.schedule.worker[i].index_list]
        return worker_end

    def key_route(self):
        critical_path = []
        node_list = []
        job_end, machine_end = self.get_job_end(), self.get_machine_end()
        a = np.argwhere(self.schedule.sjikew[4] == self.schedule.makespan)[:, 0]
        if a.shape[0] > 1:
            for i in a:
                if self.not_dummy(i):
                    node_list.append(Node([i]))
        else:
            node_list.append(Node([a[0]]))
        while len(node_list):
            while True:
                index = node_list[0].index
                b = index[-1]
                c = self.schedule.sjikew[0][b]
                d = self.schedule.sjikew[2][b]
                e = self.schedule.sjikew[3][b]
                try:
                    f = self.schedule.job[d].index_list[np.argwhere(job_end[d] == c)[:, 0][0]]
                except IndexError:
                    f = None
                try:
                    g = self.schedule.machine[e].index_list[np.argwhere(machine_end[e] == c)[:, 0][0]]
                except IndexError:
                    g = None
                if f is not None and self.not_dummy(f) and g is not None and self.not_dummy(g):
                    z = {f, g}
                    for v in z:
                        index_a = deepcopy(index)
                        index_a.append(v)
                        node_list.append(Node(index_a))
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

    def key_route_worker(self):
        critical_path = []
        node_list = []
        job_end, machine_end, worker_end = self.get_job_end(), self.get_machine_end(), self.get_worker_end()
        a = np.argwhere(self.schedule.sjikew[4] == self.schedule.makespan)[:, 0]
        if a.shape[0] > 1:
            for i in a:
                if self.not_dummy(i):
                    node_list.append(Node([i]))
        else:
            node_list.append(Node([a[0]]))
        while len(node_list):
            while True:
                index = node_list[0].index
                b = index[-1]
                c = self.schedule.sjikew[0][b]
                d = self.schedule.sjikew[2][b]
                e = self.schedule.sjikew[3][b]
                w = self.schedule.sjikew[5][b]
                try:
                    f = self.schedule.job[d].index_list[np.argwhere(job_end[d] == c)[:, 0][0]]
                except IndexError:
                    f = None
                try:
                    g = self.schedule.machine[e].index_list[np.argwhere(machine_end[e] == c)[:, 0][0]]
                except IndexError:
                    g = None
                try:
                    h = self.schedule.worker[w].index_list[np.argwhere(worker_end[w] == c)[:, 0][0]]
                except IndexError:
                    h = None
                if f is not None and self.not_dummy(f) and g is not None and self.not_dummy(
                        g) and h is not None and self.not_dummy(h):
                    z = {f, g, h}
                    for v in z:
                        index_a = deepcopy(index)
                        index_a.append(v)
                        node_list.append(Node(index_a))
                    node_list.pop(0)
                elif f is not None and self.not_dummy(f) and g is not None and self.not_dummy(g) and h is None:
                    z = {f, g}
                    for v in z:
                        index_a = deepcopy(index)
                        index_a.append(v)
                        node_list.append(Node(index_a))
                    node_list.pop(0)
                elif f is not None and self.not_dummy(f) and h is not None and self.not_dummy(h) and g is None:
                    z = {f, h}
                    for v in z:
                        index_a = deepcopy(index)
                        index_a.append(v)
                        node_list.append(Node(index_a))
                    node_list.pop(0)
                elif g is not None and self.not_dummy(g) and h is not None and self.not_dummy(h) and f is None:
                    z = {g, h}
                    for v in z:
                        index_a = deepcopy(index)
                        index_a.append(v)
                        node_list.append(Node(index_a))
                    node_list.pop(0)
                elif f is not None and self.not_dummy(f) and g is None and h is None:
                    node_list[0].index.append(f)
                elif f is None and g is not None and self.not_dummy(g) and h is None:
                    node_list[0].index.append(g)
                elif f is None and g is None and h is not None and self.not_dummy(h):
                    node_list[0].index.append(h)
                else:
                    critical_path.extend(node_list[0].index)
                    node_list.pop(0)
                    break
        return critical_path

    def key_block(self, func=None):
        func = self.key_route if func is None else func
        index = list(set(func()))
        index_start = self.schedule.sjikew[0][index]
        index = [index[i] for i in np.argsort(-index_start)]
        a = self.schedule.sjikew[3][index]
        b = set(a)
        c = 0
        block = {}
        for i in b:
            block[c] = np.array([], dtype=int)
            d = np.argwhere(a == i)[:, 0]
            start = self.schedule.sjikew[0][[index[j] for j in d]].tolist()
            for cur, j in enumerate(d):
                g = index[j]
                try:
                    end = self.schedule.sjikew[4][g]
                    start.index(end)
                except ValueError:
                    if cur != 0:
                        c += 1
                        block[c] = np.array([], dtype=int)
                self.schedule.job[self.schedule.sjikew[2][g]].task[
                    self.schedule.sjikew[1][g]].block = c
                block[c] = np.append(block[c], g)
            c += 1
        for u, v in block.items():
            block[u] = v[np.argsort(v)]
        return block

    def gantt_chart_png(self, filename="GanttChart", fig_width=9, fig_height=5, random_colors=False, lang=1, dpi=200,
                        height=0.8, scale_more=None, x_step=None, y_based=0, text_rotation=0,
                        with_operation=True, with_start_end=False, key_block=False, show=False):
        if random_colors:
            random.shuffle(COLORS)
        if key_block:
            if self.wok is None:
                self.key_block()
            else:
                self.key_block(self.key_route_worker)
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
                    if self.wok is None:
                        y_values = [machine, job.index]
                    else:
                        y_values = [machine, self.wok[job.index][task.index]]
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
            if self.wok is None:
                for job in self.schedule.job.values():
                    plt.barh(0, 0, color=COLORS[job.index % LEN_COLORS], label=job.index + 1)
            else:
                for worker in self.schedule.worker.values():
                    plt.barh(0, 0, color=COLORS[worker.index % LEN_COLORS], label=worker.index + 1)
            plt.barh(y=0, width=self.schedule.makespan / scale_more, left=self.schedule.makespan, color="white")
            if lang == 0:
                title = r"${Job}$" if self.wok is None else r"${Worker}$"
            else:
                title = "工件" if self.wok is None else "工人"
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
    def __init__(self, schedule, code, mac=None, route=None, wok=None):
        self.schedule = deepcopy(schedule)
        self.code = code
        self.mac = mac
        self.route = route
        self.wok = wok
        GanttChart.__init__(self, schedule=self.schedule, mac=self.mac, wok=self.wok)

    def print(self):
        code = self.code.tolist() if type(self.code) is np.ndarray else self.code
        a = {"code": code, "mac": self.mac, "route": self.route, "wok": self.wok,
             "direction": self.schedule.direction, "makespan": self.schedule.makespan,
             "sjikew[2]": self.schedule.sjikew[2].tolist(), "sjikew[1]": self.schedule.sjikew[1].tolist(),
             "sjikew[3]": self.schedule.sjikew[3].tolist(), "sjikew[5]": self.schedule.sjikew[5].tolist(), "id": self,
             "schedule_id": self.schedule}
        for i, j in a.items():
            print("%s: %s" % (i, j))

    def trans_operation_based2machine_based(self):  # 转码：基于工序的编码->基于机器的编码
        code = [[] for _ in self.schedule.machine.keys()]
        for i, machine in self.schedule.machine.items():
            job = self.schedule.sjikew[2][machine.index_list]
            operation = self.schedule.sjikew[1][machine.index_list]
            for a, b in zip(job, operation):
                code[i].append((a, b))
        return code

    def std_code(self, std_direction=None):
        if std_direction not in [0, 1]:
            std_direction = Utils.direction()
        if self.schedule.direction == 0:
            if std_direction == 0:
                index = np.argsort(self.schedule.sjikew[0])
            else:
                index = np.argsort(-self.schedule.sjikew[0])[::-1]
        else:
            if std_direction == 0:
                index = np.argsort(self.schedule.sjikew[0])[::-1]
            else:
                index = np.argsort(-self.schedule.sjikew[0])
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
                    self.schedule.sjikew[u][g] = v
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
                    self.schedule.sjikew[u][g] = v
                self.schedule.job[i].nd += 1

    def std_code_machine_based(self):
        for k in self.schedule.machine.keys():
            start = []
            for i, j in self.code[k]:
                start.append(self.schedule.job[i].task[j].start)
            self.code[k] = [self.code[k][u] for u in np.argsort(start)]

    def save_code_to_txt(self, file):
        try:
            Utils.save_code_to_txt(file,
                                   {"code": self.code.tolist(), "route": self.route, "mac": self.mac, "wok": self.wok})
        except AttributeError:
            Utils.save_code_to_txt(file, {"code": self.code, "route": self.route, "mac": self.mac, "wok": self.wok})

    def save_gantt_chart_to_csv(self, file):
        if not file.endswith(".csv"):
            file = file + ".csv"
        with open(file, "w", encoding="utf-8") as f:
            f.writelines("Job,Operation,Machine,Start,Duration,End,Worker\n")
            for job in self.schedule.job.values():
                for task in job.task.values():
                    if self.wok is None:
                        worker = task.worker
                    else:
                        worker = self.wok[job.index][task.index]
                    if self.mac is None:
                        machine = task.machine
                        if self.wok is None:
                            duration = task.duration
                        else:
                            duration = task.duration[task.worker.index(worker)]
                    else:
                        machine = self.mac[job.index][task.index]
                        index_machine = task.machine.index(machine)
                        if self.wok is None:
                            duration = task.duration[index_machine]
                        else:
                            index_worker = task.worker[index_machine].index(worker)
                            duration = task.duration[index_machine][index_worker]
                    if self.wok is not None:
                        worker += 1
                    f.writelines("{},{},{},{},{},{},{}\n".format(
                        job.index + 1, task.index + 1, machine + 1, task.start, duration, task.end, worker))

    def ga_crossover_sequence(self, info):
        func_dict = {
            Crossover.default: self.ga_crossover_sequence_pox,
            Crossover.pox: self.ga_crossover_sequence_pox,
            Crossover.mox1: self.ga_crossover_sequence_mox1,
            Crossover.mox2: self.ga_crossover_sequence_mox2,
            Crossover.ipox: self.ga_crossover_sequence_ipox,
            Crossover.dpox: self.ga_crossover_sequence_dpox,
            Crossover.ox: self.ga_crossover_sequence_ox,
            Crossover.pmx: self.ga_crossover_sequence_pmx,
        }
        func = func_dict[self.schedule.ga_operator[Crossover.name]]
        return func(info)

    def ga_mutation_sequence(self):
        func_dict = {
            Mutation.default: self.ga_mutation_sequence_tpe,
            Mutation.tpe: self.ga_mutation_sequence_tpe,
            Mutation.insert: self.ga_mutation_sequence_insert,
            Mutation.sub_reverse: self.ga_mutation_sequence_sr,
        }
        func = func_dict[self.schedule.ga_operator[Mutation.name]]
        return func()

    def ga_crossover_sequence_permutation(self, info):
        func_dict = {
            Crossover.default: self.ga_crossover_sequence_permutation_pmx,
            Crossover.pmx: self.ga_crossover_sequence_permutation_pmx,
            Crossover.ox: self.ga_crossover_sequence_permutation_ox,
        }
        func = func_dict[self.schedule.ga_operator[Crossover.name]]
        return func(info)

    def ga_mutation_sequence_permutation(self):
        func_dict = {
            Mutation.default: self.ga_mutation_sequence_permutation_tpe,
            Mutation.tpe: self.ga_mutation_sequence_permutation_tpe,
            Mutation.insert: self.ga_mutation_sequence_permutation_insert,
            Mutation.sub_reverse: self.ga_mutation_sequence_permutation_sr,
        }
        func = func_dict[self.schedule.ga_operator[Mutation.name]]
        return func()

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

    def ga_crossover_sequence_mox1(self, info):  # 直接使用仅适用于JSP；对于FJSP，要进行合法性检查/修复
        code1 = deepcopy(self.code)
        code2 = deepcopy(info.code)
        a = np.random.choice(range(self.schedule.m), 1, replace=False)[0]
        b, c = self.schedule.machine[a].index_list, info.schedule.machine[a].index_list
        # code1[b], code2[c] = code2[c], code1[b]
        d, e = np.delete(range(self.schedule.length), b), np.delete(range(self.schedule.length), c)
        code1[d], code2[e] = code2[e], code1[d]
        return code1, code2

    def ga_crossover_sequence_mox2(self, info):  # 直接使用仅适用于JSP；对于FJSP，要进行合法性检查/修复
        code1 = deepcopy(self.code)
        code2 = deepcopy(info.code)
        a = np.random.choice(range(self.schedule.m), 2, replace=False)
        b, c = [], []
        for i in a:
            b.extend(self.schedule.machine[i].index_list)
            c.extend(info.schedule.machine[i].index_list)
        # code1[b], code2[c] = code2[c], code1[b]
        d, e = np.delete(range(self.schedule.length), b), np.delete(range(self.schedule.length), c)
        code1[d], code2[e] = code2[e], code1[d]
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
        number1, number2 = self.schedule.sjikew[1][r_a_b], info.schedule.sjikew[1][r_a_b]
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
        number1 = self.schedule.sjikew[1]
        number2 = info.schedule.sjikew[1]
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
        mac = deepcopy(mac)
        for i, j in enumerate(mac):
            for u, v in enumerate(j):
                task = self.schedule.job[i].route[route[i]].task[u]
                if v not in task.machine:
                    # mac[i][u] = task.machine[task.duration.index(min(task.duration))]
                    mac[i][u] = np.random.choice(task.machine, 1, replace=False)[0]
        return mac

    """"
    =============================================================================
    Genetic operator: worker assignment problem
    =============================================================================
    """

    def ga_crossover_worker(self, info):
        wok1 = deepcopy(self.wok)
        wok2 = deepcopy(info.wok)
        for i, (p, q) in enumerate(zip(wok1, wok2)):
            for j, (u, v) in enumerate(zip(p, q)):
                if np.random.random() < 0.5:
                    wok1[i][j], wok2[i][j] = v, u
        return wok1, wok2

    def ga_mutation_worker(self, mac):
        wok = deepcopy(self.wok)
        for i in range(self.schedule.n):
            for j, k in enumerate(wok[i]):
                if np.random.random() < 0.5:
                    try:
                        task = self.schedule.job[i].task[j]
                        index_k = task.machine.index(mac[i][j])
                        a = [v for v in task.worker[index_k] if v != k]
                        wok[i][j] = np.random.choice(a, 1, replace=False)[0]
                    except ValueError:
                        pass
        return wok

    def repair_mac_wok(self, mac, wok):
        wok = deepcopy(wok)
        for i, j in enumerate(mac):
            for u, v in enumerate(j):
                task = self.schedule.job[i].task[u]
                index_k = task.machine.index(v)
                if wok[i][u] not in task.worker[index_k]:
                    # index = task.duration[index_k].index(min(task.duration[index_k]))
                    # wok[i][u] = task.worker[index_k][index]
                    wok[i][u] = np.random.choice(task.worker[index_k], 1, replace=False)[0]
        return wok

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

    def ts_assignment_job_based(self, tabu_list, max_tabu):
        mac = deepcopy(self.mac)
        n_try = 0
        while n_try < max_tabu:
            n_try += 1
            try:
                i = np.random.randint(0, self.schedule.n, 1)[0]
                j = np.random.randint(0, self.schedule.job[i].nop, 1)[0]
                k = np.random.choice(self.schedule.job[i].task[j].machine, 1, replace=False)[0]
                tabu = {"j-%s" % i, "t-%s" % j, k}
                if mac[i][j] != k and tabu not in tabu_list:
                    tabu_list.append(tabu)
                    mac[i][j] = k
                    break
            except ValueError:
                pass
        return mac

    def ts_wok_job_based(self, mac, tabu_list, max_tabu):
        wok = deepcopy(self.wok)
        n_try = 0
        while n_try < max_tabu:
            n_try += 1
            try:
                i = np.random.randint(0, self.schedule.n, 1)[0]
                j = np.random.randint(0, self.schedule.job[i].nop, 1)[0]
                task = self.schedule.job[i].task[j]
                index_k = task.machine.index(mac[i][j])
                w = np.random.choice(task.worker[index_k], 1, replace=False)[0]
                tabu = {"j-%s" % i, "t-%s" % j, w}
                if wok[i][j] != w and tabu not in tabu_list:
                    tabu_list.append(tabu)
                    wok[i][j] = w
                    break
            except ValueError:
                pass
        return wok

    """"
    =============================================================================
    Local operator
    =============================================================================
    """

    def key_block_move_hybrid(self, block=None, func=None):
        if np.random.random() < 0.5:
            return self.key_block_move(block, func)
        return self.key_block_move_complete(block, func)

    def key_block_move(self, block=None, func=None):
        self.std_code()
        code = deepcopy(self.code)
        func = self.key_route if func is None else func
        if block is None:
            block = self.key_block(func)
        all_blocks = list(block.keys())
        n_blocks = len(all_blocks)
        i = 0
        while i < n_blocks:
            try:
                one_block = np.random.choice(all_blocks, 1, replace=False)[0]
                j = block[one_block]
                i += 1
                all_blocks.remove(one_block)
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
                    break
            except ValueError:
                break
        return code

    def key_block_move_mac(self, block=None, func=None):
        mac = deepcopy(self.mac)
        func = self.key_route if func is None else func
        if block is None:
            block = self.key_block(func)
        for i, j in block.items():
            if j.shape[0] >= 2:
                for g in j:
                    if np.random.random() < 0.5:
                        try:
                            a, b = self.schedule.sjikew[2][g], self.schedule.sjikew[1][g]
                            c = [v for v in self.schedule.job[a].task[b].machine if v != mac[a][b]]
                            mac[a][b] = np.random.choice(c, 1, replace=False)[0]
                        except ValueError:
                            pass
        return mac

    def key_block_move_wok(self, mac, block=None, func=None):
        wok = deepcopy(self.wok)
        func = self.key_route_worker if func is None else func
        if block is None:
            block = self.key_block(func)
        for i, j in block.items():
            if j.shape[0] >= 2:
                for g in j:
                    if np.random.random() < 0.5:
                        try:
                            a, b = self.schedule.sjikew[2][g], self.schedule.sjikew[1][g]
                            task = self.schedule.job[a].task[b]
                            index_k = task.machine.index(mac[a][b])
                            c = [v for v in task.worker[index_k] if v != wok[a][b]]
                            wok[a][b] = np.random.choice(c, 1, replace=False)[0]
                        except ValueError:
                            pass
        return wok

    """"
    =============================================================================
    Complete key block move with evaluate strategies
    =============================================================================
    """

    def get_duration(self, job_id, task):
        task_id = task.index
        if self.mac is None:
            if self.wok is None:
                a = task.duration
            else:
                wok_id = self.wok[job_id][task_id]
                a = task.duration[task.worker.index(wok_id)]
        else:
            machine = self.mac[job_id][task_id]
            if self.wok is None:
                a = task.duration[task.machine.index(machine)]
            else:
                wok_id = self.wok[job_id][task_id]
                a = task.duration[[task.machine.index(machine)]][task.worker.index(wok_id)]
        return a

    def get_duration_by_index(self, index):
        job_id = self.schedule.sjikew[2][index]
        task_id = self.schedule.sjikew[1][index]
        task = self.schedule.job[job_id].task[task_id]
        return self.get_duration(job_id, task)

    def evaluate_head_job(self, index):
        job_id = self.schedule.sjikew[2][index]
        task_id = self.schedule.sjikew[1][index]
        machine_id = self.schedule.sjikew[3][index]
        first = self.schedule.machine[machine_id].index_list[0]
        if self.schedule.sjikew[0][index] == self.schedule.sjikew[0][first]:
            a = 0
        else:
            try:
                a = self.schedule.job[job_id].task[task_id - 1].end
            except KeyError:
                a = 0
        b = self.get_duration(job_id, self.schedule.job[job_id].task[task_id])
        return a + b

    def evaluate_head_mac(self, index):
        machine_id = self.schedule.sjikew[3][index]
        mac_list = self.schedule.machine[machine_id].index_list
        index_cp = mac_list.index(index)
        index_mp = index_cp if index_cp == 0 else index_cp - 1
        index = self.schedule.machine[machine_id].index_list[index_mp]
        a = 0 if index_mp == 0 else self.schedule.sjikew[4][index]
        return a

    def evaluate_tail_job(self, index):
        job_id = self.schedule.sjikew[2][index]
        task_id = self.schedule.sjikew[1][index]
        return self.schedule.job[job_id].remain(task_id, include=False)

    def evaluate_tail_mac(self, index):
        a = 0
        machine_id = self.schedule.sjikew[3][index]
        mac_list = self.schedule.machine[machine_id].index_list
        if self.schedule.direction == 0:
            b = [i for i in mac_list if i > index]
        else:
            b = [i for i in mac_list if i < index]
        for i in b:
            job_id = self.schedule.sjikew[2][i]
            task_id = self.schedule.sjikew[1][i]
            task = self.schedule.job[job_id].task[task_id]
            a += self.get_duration(task_id, task)
        return a

    def evaluate(self, neg_complete):  # 领域结构的近似评价方法
        evaluate = []
        for neg in neg_complete:
            a, b = [], []  # 头部评价，尾部评价
            if neg[4] in [1, 2]:  # 块首向后插入
                index_stop = neg[3].index(neg[2]) + 1
                if neg[4] == 1:
                    """头部评价"""
                    c = self.evaluate_head_job(neg[3][1])
                    e = self.evaluate_head_mac(neg[1]) + self.get_duration_by_index(neg[3][1])
                    a.append(max([c, e]))  # 关键上的第2道工序的头部评价
                    for tmp, index in enumerate(range(2, index_stop)):
                        c = self.evaluate_head_job(neg[3][index])
                        e = self.get_duration_by_index(index)
                        a.append(max([a[tmp] + e, c]))  # 关键上的第3道工序至插入位置工序的头部评价
                else:
                    """头部评价"""
                    c = self.evaluate_head_job(neg[2])
                    e = self.evaluate_head_mac(neg[1]) + self.get_duration_by_index(neg[2])
                    a.append(max([c, e]))  # 插入位置工序的头部评价
                    for tmp, index in enumerate(range(1, index_stop - 1)):
                        c = self.evaluate_head_job(neg[3][index])
                        e = self.get_duration_by_index(index)
                        a.append(max([a[tmp] + e, c]))  # 关键上的第2道工序至插入位置起倒数第2道工序的头部评价
                    a[-1], a[:-1] = a[0], a[1:]
                e = self.get_duration_by_index(neg[1])
                a.append(max([a[-1] + e, self.evaluate_head_job(neg[1])]))  # 块首的头部评价
                """尾部评价"""
                for index in range(index_stop - 1, 0, -1):  # 插入位置工序至关键上的第2道工序的尾部评价
                    f = self.evaluate_tail_job(index)
                    g = self.evaluate_tail_mac(index) + self.get_duration_by_index(neg[1])
                    b.append(max([f, g]))
                f = self.evaluate_tail_job(neg[1])
                g = self.evaluate_tail_mac(neg[2])
                b.append(max([f, g]))  # 块首的尾部评价
            else:
                """头部评价"""
                c = self.evaluate_head_job(neg[1])
                e = self.evaluate_head_mac(neg[2]) + self.get_duration_by_index(neg[1])
                a.append(max([c, e]))  # 块尾的头部评价
                index_start = neg[3].index(neg[2])
                len_neg = len(neg[3])
                if neg[4] == 3:
                    for tmp, index in enumerate(range(index_start, len_neg - 1)):  # 插入位置起第1道工序至倒数第2道工序的头部评价
                        c = self.evaluate_head_job(neg[3][index])
                        e = self.get_duration_by_index(index)
                        a.append(max([a[tmp] + e, c]))
                else:
                    for tmp, index in enumerate(range(index_start + 1, len_neg - 1)):  # 插入位置起第2道工序至倒数第2道工序的头部评价
                        c = self.evaluate_head_job(neg[3][index])
                        e = self.get_duration_by_index(index)
                        a.append(max([a[tmp] + e, c]))
                    e = self.get_duration_by_index(neg[2])
                    a.append(max([a[-1] + e, self.evaluate_head_job(neg[2])]))  # 插入位置起第1道工序的头部评价
                    a[-1], a[:-1] = a[0], a[1:]
                """尾部评价"""
                f = self.evaluate_tail_job(neg[1])
                g = self.evaluate_tail_mac(neg[2]) - self.get_duration_by_index(neg[1])
                b.append(max([f, g]))  # 块尾的尾部评价
                for index in range(index_start, len_neg - 1):  # 插入位置起第1道工序至倒数第2道工序的尾部评价
                    f = self.evaluate_tail_job(index)
                    g = self.evaluate_tail_mac(index) - self.get_duration_by_index(neg[1])
                    b.append(max([f, g]))
            z = [i + j for i, j in zip(a, b)]
            evaluate.append(max(z))
        return evaluate

    def key_block_move_complete(self, block=None, func=None):
        self.std_code()
        func = self.key_route if func is None else func
        if block is None:
            block = self.key_block(func)
        neg_complete = []
        for i, j in block.items():
            if j.shape[0] >= 2:
                head, tail = j[0], j[-1]
                for index in j[1:]:
                    code = deepcopy(self.code)
                    value = code[head]
                    obj = np.delete(code, head)
                    code = np.insert(obj, index - 1, value)
                    code[index], code[index - 1] = code[index - 1], code[index]
                    neg_complete.append([code, head, index, j.tolist(), 1])
                for index in j[1:]:
                    code = deepcopy(self.code)
                    code[head], code[index] = code[index], code[head]
                    neg_complete.append([code, head, index, j.tolist(), 2])
                for index in j[:-1]:
                    code = deepcopy(self.code)
                    value = code[tail]
                    obj = np.delete(code, tail)
                    code = np.insert(obj, index, value)
                    neg_complete.append([code, tail, index, j.tolist(), 3])
                for index in j[:-1]:
                    code = deepcopy(self.code)
                    code[tail], code[index] = code[index], code[tail]
                    neg_complete.append([code, tail, index, j.tolist(), 4])
        evaluate = self.evaluate(neg_complete)
        threshold = min(evaluate)
        choice_list = [i for i, j in enumerate(evaluate) if j == threshold]
        choice = np.random.choice(choice_list, 1, replace=False)[0]
        return neg_complete[choice][0]
        # return neg_complete

    """"
    =============================================================================
    Other operator
    =============================================================================
    """

    def dislocation_operator(self, direction=0):
        code = deepcopy(self.code)
        return np.hstack([code[1:], code[0]]) if direction == 0 else np.hstack([code[-1], code[:-1]])
