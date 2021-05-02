__doc__ = """
Non-dominate sort genetic algorithm: 非支配排序遗传算法
class NSGA: 定义的基类
class NSGAJsp(NSGA): Jsp的NSGA, 重载了***
...
"""

import copy
import time

import numpy as np

from ..pareto import Pareto, SelectPareto
from ..utils import Utils

deepcopy = copy.deepcopy


class NSGA:
    def __init__(self, pop_size, rc, rm, max_generation, objective, schedule):
        self.pop_size = pop_size
        self.rc = rc
        self.rm = rm
        self.max_generation = max_generation
        self.objective = objective
        self.schedule = schedule
        self.p = [job.nop for job in self.schedule.job.values()]
        self.tech = [[task.machine for task in job.task.values()] for job in self.schedule.job.values()]
        self.num_obj = len(self.objective)
        self.pop = [[], [], []]  # (info, objective)
        self.pop_child = [[], []]  # (info, objective)
        # (start, end, pareto front rate)
        self.record = [[], [], []]
        self.max_tabu = 3 * self.schedule.n * self.schedule.m
        individual = range(self.pop_size)
        # (code, mac, tech)
        self.tabu_list = [[[] for _ in individual], [[] for _ in individual], [[] for _ in individual]]

    def clear(self):
        self.pop = [[], []]
        self.pop_child = [[], []]
        self.record = [[], [], []]
        individual = range(self.pop_size)
        self.tabu_list = [[[] for _ in individual], [[] for _ in individual], [[] for _ in individual]]

    def get_obj(self, info):
        obj = []
        for func in self.objective:
            obj.append(func(info))
        return obj

    def update_child(self, info):
        self.pop_child[0].append(info)
        self.pop_child[1].append(self.get_obj(info))

    def show_generation(self, g):
        Utils.print("Generation {:<4} Runtime {:<8.4f} Pareto rate: {:<.2f}".format(
            g, self.record[1][g] - self.record[0][g], self.record[2][g]))

    def do_selection(self):
        if len(self.pop_child[0]) != 0:
            info_new = []
            obj = np.vstack(self.pop_child[1])
            obj_new = np.vstack([self.pop[1], obj])
            for info in self.pop[0]:
                info_new.append(info)
            for info in self.pop_child[0]:
                info_new.append(info)
            scale = obj_new.shape[0]
            obj_new = obj_new.tolist()
        else:
            info_new = self.pop[0]
            obj_new = self.pop[1]
            scale = self.pop_size
        info_copy = copy.deepcopy(info_new)
        obj_copy = copy.deepcopy(obj_new)
        pareto = Pareto(scale, obj_copy, self.num_obj)
        pareto.fast_non_dominate_sort()  # 非支配排序
        pareto.crowd_distance()  # 计算拥挤度
        f = pareto.f
        rank = pareto.rank
        cd = pareto.cd
        select_pareto = SelectPareto(self.pop_size, scale, f, rank, cd)
        index = select_pareto.elite_strategy()  # 精英策略
        pareto_front = []
        self.pop = [[], [], []]
        for i in range(self.pop_size):
            b = index[i]
            self.pop[0].append(info_copy[b])
            self.pop[1].append(obj_copy[b])
            if b in f[0]:
                pareto_front.append(i)
        self.record[2].append(len(pareto_front) / self.pop_size)
        self.pop_child = [[], [], []]

    def do_init(self, pop=None):
        pass

    def do_crossover(self, i, j):
        pass

    def do_mutation(self, i):
        pass

    def do_tabu_search(self, i):
        pass

    def do_key_block_move(self, i):
        pass

    def do_evolution(self, pop=None, n_level=5, column=0):
        Utils.print("{}Evolution  start{}".format("=" * 48, "=" * 48), fore=Utils.fore().LIGHTYELLOW_EX)
        self.clear()
        self.do_init(pop)
        self.do_selection()
        self.show_generation(0)
        for g in range(1, self.max_generation + 1):
            self.record[0].append(time.perf_counter())
            for i in range(self.pop_size):
                if np.random.random() < self.rc:
                    j = np.random.choice(np.delete(np.arange(self.pop_size), i), 1, replace=False)[0]
                    self.do_crossover(i, j)
                if np.random.random() < self.rm:
                    self.do_mutation(i)
            self.do_selection()
            self.record[1].append(time.perf_counter())
            self.show_generation(g)
        Utils.print("{}Evolution finish{}".format("=" * 48, "=" * 48), fore=Utils.fore().LIGHTRED_EX)
        # 对结果进行处理
        pareto = Pareto(self.pop_size, self.pop[1], self.num_obj)
        pareto.fast_non_dominate_sort()
        all_res = []
        for level in range(len(pareto.f[:n_level])):
            if len(pareto.f[level]) == 1 and level > 0:
                break
            res, obj = [], []
            index = pareto.sort_obj_by(pareto.f[level], column)
            for i in [pareto.f[level][v] for v in index]:
                if self.pop[1][i] not in obj:
                    res.append((self.pop[0][i], self.pop[1][i]))
                obj.append(self.pop[1][i])
            all_res.append(res)
        return all_res


class NSGAJsp(NSGA):
    def __init__(self, pop_size, rc, rm, max_generation, objective, schedule):
        NSGA.__init__(self, pop_size, rc, rm, max_generation, objective, schedule)

    def decode_update(self, code, route=None):
        info = self.schedule.decode(code, route)
        self.update_child(info)

    def do_init(self, pop=None):
        self.record[0].append(time.perf_counter())
        for i in range(self.pop_size):
            if pop is None:
                code = self.schedule.sequence_operation_based(self.schedule.n, self.p)
            else:
                code = pop[0][i].code
            info = self.schedule.decode(code)
            self.pop[0].append(info)
            self.pop[1].append(self.get_obj(info))
        self.record[1].append(time.perf_counter())

    def do_crossover(self, i, j):
        code1, code2 = self.pop[0][i].ga_crossover_sequence_pox(self.pop[0][j])
        self.decode_update(code1)
        self.decode_update(code2)

    def do_mutation(self, i):
        code1 = self.pop[0][i].ga_mutation_sequence_operation_based_tpe()
        self.decode_update(code1)
