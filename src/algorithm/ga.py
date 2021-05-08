__doc__ = """
Genetic algorithm: 遗传算法
class Ga: 定义的基类
class GaJsp(De): Jsp的Ga, 重载了***
...
"""

import copy
import time

import numpy as np

from ..utils import Utils

deepcopy = copy.deepcopy


class Ga:
    def __init__(self, pop_size, rc, rm, max_generation, objective, schedule, max_stay_generation=None):
        """
        初始化参数。
        pop_size: 种群规模；rc: 交叉概率；rm: 变异概率；max_generation: 最大迭代次数；
        objective: 求解目标值的函数；schedule: 调度对象；max_stay_generation：最大滞留代数
        """
        self.pop_size = pop_size
        self.rc = rc
        self.rm = rm
        self.max_generation = max_generation
        self.objective = objective
        self.schedule = schedule
        self.max_stay_generation = max_stay_generation
        self.p = [job.nop for job in self.schedule.job.values()]
        self.tech = [[task.machine for task in job.task.values()] for job in self.schedule.job.values()]
        self.best = [None, None, None, [[], [], []]]  # (info, objective, fitness, tabu)
        self.pop = [[], [], []]  # (info, objective, fitness)
        # (start, end, best_objective, best_fitness, worst_fitness, mean_fitness)
        self.record = [[], [], [], [], [], []]
        # (code, mac, tech)
        self.max_tabu = Utils.len_tabu(self.schedule.m, self.schedule.n)
        self.individual = range(self.pop_size)
        self.tabu_list = [[[] for _ in self.individual], [[] for _ in self.individual], [[] for _ in self.individual]]

    def clear(self):
        self.best = [None, None, None, [[], [], []]]
        self.pop = [[], [], []]
        self.record = [[], [], [], [], [], []]
        self.tabu_list = [[[] for _ in self.individual], [[] for _ in self.individual], [[] for _ in self.individual]]

    def dislocation(self, i):
        pass

    def update_individual(self, i, obj_new, info_new):
        fit_new = Utils.calculate_fitness(obj_new)
        if Utils.similarity(self.pop[0][i].code, info_new.code) > 0.5:
            self.dislocation(i)
        else:
            self.pop[0].append(info_new)
            self.pop[1].append(obj_new)
            self.pop[2].append(fit_new)
            if Utils.update_info(self.pop[1][i], obj_new):
                for k in range(3):
                    self.tabu_list[k][i] = []
        for k in range(3):
            self.tabu_list[k].append([])
        if Utils.update_info(self.best[1], obj_new):
            self.best[0] = info_new
            self.best[1] = obj_new
            self.best[2] = fit_new
            for k in range(3):
                self.best[3][k].append([])

    def replace_individual(self, i, info_new):
        obj_new = self.objective(info_new)
        fit_new = Utils.calculate_fitness(obj_new)
        self.pop[0][i] = info_new
        self.pop[1][i] = obj_new
        self.pop[2][i] = fit_new
        for k in range(3):
            self.tabu_list[k][i] = []

    def init_best(self):
        self.best[2] = max(self.pop[2])
        index = self.pop[2].index(self.best[2])
        self.best[1] = self.pop[1][index]
        self.best[0] = deepcopy(self.pop[0][index])

    def show_generation(self, g):
        self.record[2].append(self.best[1])
        self.record[3].append(self.best[2])
        self.record[4].append(min(self.pop[2]))
        self.record[5].append(np.mean(self.pop[2]))
        Utils.print(
            "Generation {:<4} Runtime {:<8.4f} fBest: {:<.8f}, fWorst: {:<.8f}, fMean: {:<.8f}, gBest: {:<.2f} ".format(
                g, self.record[1][g] - self.record[0][g], self.record[3][g], self.record[4][g], self.record[5][g],
                self.record[2][g]))

    def do_selection(self):
        a = np.array(self.pop[2]) / sum(self.pop[2])
        b = np.array([])
        for i in range(a.shape[0]):
            b = np.append(b, sum(a[:i + 1]))
        pop = deepcopy(self.pop)
        tabu_list = deepcopy(self.tabu_list)
        self.pop = [[], [], []]
        self.tabu_list = [[[] for _ in self.individual], [[] for _ in self.individual], [[] for _ in self.individual]]
        for i in range(self.pop_size):
            j = np.argwhere(b > np.random.random())[0, 0]  # 轮盘赌选择
            self.pop[0].append(pop[0][j])
            self.pop[1].append(pop[1][j])
            self.pop[2].append(pop[2][j])
            for k in range(3):
                self.tabu_list[k][i] = tabu_list[k][j]
        self.pop[0][0] = self.best[0]
        self.pop[1][0] = self.best[1]
        self.pop[2][0] = self.best[2]
        for k in range(3):
            self.tabu_list[k][0] = self.best[3][k]

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

    def reach_max_stay_generation(self, g):
        if self.max_stay_generation is not None and g > self.max_stay_generation and self.record[2][g - 1] == \
                self.record[2][g - self.max_stay_generation]:
            return True
        return False

    def reach_best_known_solution(self):
        if self.schedule.best_known is not None and self.best[1] <= self.schedule.best_known:
            return True
        return False

    def do_evolution(self, tabu_search=True, key_block_move=False, pop=None):
        Utils.print("{}Evolution  start{}".format("=" * 48, "=" * 48), fore=Utils.fore().LIGHTYELLOW_EX)
        self.clear()
        self.do_init(pop)
        self.do_selection()
        for g in range(1, self.max_generation + 1):
            if self.reach_best_known_solution():
                break
            if self.reach_max_stay_generation(g):
                break
            self.record[0].append(time.perf_counter())
            for i in range(self.pop_size):
                if self.reach_best_known_solution():
                    break
                if np.random.random() < self.rc:
                    j = np.random.choice(np.delete(np.arange(self.pop_size), i), 1, replace=False)[0]
                    self.do_crossover(i, j)
                if np.random.random() < self.rm:
                    self.do_mutation(i)
                if tabu_search:
                    self.do_tabu_search(i)
                if key_block_move:
                    self.do_key_block_move(i)
            self.do_selection()
            self.record[1].append(time.perf_counter())
            self.show_generation(g)
        Utils.print("{}Evolution finish{}".format("=" * 48, "=" * 48), fore=Utils.fore().LIGHTRED_EX)


class GaJsp(Ga):
    def __init__(self, pop_size, rc, rm, max_generation, objective, schedule, max_stay_generation=None):
        Ga.__init__(self, pop_size, rc, rm, max_generation, objective, schedule, max_stay_generation)

    def decode_update(self, i, code):
        info = self.schedule.decode(code)
        self.update_individual(i, self.objective(info), info)

    def dislocation(self, i):
        code = self.pop[0][i].dislocation_operator()
        info = self.schedule.decode(code)
        self.replace_individual(i, info)

    def do_init(self, pop=None):
        self.record[0].append(time.perf_counter())
        for i in range(self.pop_size):
            if pop is None:
                code = self.schedule.sequence_operation_based(self.schedule.n, self.p)
            else:
                code = pop[0][i].code
            info = self.schedule.decode(code)
            self.pop[0].append(info)
            self.pop[1].append(self.objective(info))
            self.pop[2].append(Utils.calculate_fitness(self.pop[1][i]))
        self.init_best()
        self.record[1].append(time.perf_counter())
        self.show_generation(0)

    def do_crossover(self, i, j):
        code1, code2 = self.pop[0][i].ga_crossover_sequence_pox(self.pop[0][j])
        self.decode_update(i, code1)
        self.decode_update(j, code2)

    def do_mutation(self, i):
        code1 = self.pop[0][i].ga_mutation_sequence_tpe()
        self.decode_update(i, code1)

    def do_tabu_search(self, i):
        code1 = self.pop[0][i].ts_sequence_operation_based(self.tabu_list[0][i], self.max_tabu)
        self.decode_update(i, code1)
        if len(self.tabu_list[0][i]) >= self.max_tabu:
            self.tabu_list[0][i] = []

    def do_key_block_move(self, i):
        code1 = self.pop[0][i].key_block_move()
        self.decode_update(i, code1)
