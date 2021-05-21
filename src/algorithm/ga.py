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
        self.direction = Utils.direction0none(objective)
        self.r = [job.nor for job in self.schedule.job.values()]
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

    def update_p(self, route):
        self.schedule.clear(route)
        self.p = [job.nop for job in self.schedule.job.values()]

    def update_tech(self, mode=0):
        if mode == 0:
            self.tech = [[task.machine for task in job.task.values()] for job in self.schedule.job.values()]
        else:
            self.tech = [[[task.machine for task in route.task.values()] for route in job.route.values()] for job in
                         self.schedule.job.values()]

    def get_obj_fit(self, info):
        a = self.objective(info)
        b = Utils.calculate_fitness(a)
        return a, b

    def dislocation(self, i, mode=0):
        pass

    def init_best(self):
        self.best[2] = max(self.pop[2])
        index = self.pop[2].index(self.best[2])
        self.best[1] = self.pop[1][index]
        self.best[0] = deepcopy(self.pop[0][index])

    def append_individual(self, info_new):
        obj_new, fit_new = self.get_obj_fit(info_new)
        self.pop[0].append(info_new)
        self.pop[1].append(obj_new)
        self.pop[2].append(fit_new)
        for k in range(3):
            self.tabu_list[k].append([])
        self.update_best(obj_new, info_new, fit_new)

    def replace_individual(self, i, info_new):
        obj_new, fit_new = self.get_obj_fit(info_new)
        if Utils.update_info(self.pop[1][i], obj_new):
            self.pop[0][i] = info_new
            self.pop[1][i] = obj_new
            self.pop[2][i] = fit_new
            for k in range(3):
                self.tabu_list[k][i] = []
            self.update_best(obj_new, info_new, fit_new)

    def update_best(self, obj_new, info_new, fit_new):
        if Utils.update_info(self.best[1], obj_new):
            self.best[0] = info_new
            self.best[1] = obj_new
            self.best[2] = fit_new
            for k in range(3):
                self.best[3][k] = []

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

    def do_evolution(self, tabu_search=True, key_block_move=False, pop=None, exp_no=None):
        exp_no = "" if exp_no is None else exp_no
        Utils.print("{}Evolution {}  start{}".format("=" * 48, exp_no, "=" * 48), fore=Utils.fore().LIGHTYELLOW_EX)
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
                if tabu_search:
                    self.do_tabu_search(i)
                if key_block_move:
                    self.do_key_block_move(i)
                if np.random.random() < self.rc:
                    j = np.random.choice(np.delete(np.arange(self.pop_size), i), 1, replace=False)[0]
                    if Utils.similarity(self.pop[0][i].code, self.pop[0][j].code) < 0.5:
                        self.do_crossover(i, j)
                    else:
                        self.dislocation(i, mode=0)
                        self.dislocation(j, mode=1)
                if np.random.random() < self.rm:
                    self.do_mutation(i)
            self.do_selection()
            self.record[1].append(time.perf_counter())
            self.show_generation(g)
        Utils.print("{}Evolution {} finish{}".format("=" * 48, exp_no, "=" * 48), fore=Utils.fore().LIGHTRED_EX)


class GaJsp(Ga):
    def __init__(self, pop_size, rc, rm, max_generation, objective, schedule, max_stay_generation=None):
        Ga.__init__(self, pop_size, rc, rm, max_generation, objective, schedule, max_stay_generation)

    def decode(self, code):
        return self.schedule.decode(code, direction=self.direction)

    def do_init(self, pop=None):
        self.record[0].append(time.perf_counter())
        for i in range(self.pop_size):
            if pop is None:
                code = self.schedule.sequence_operation_based(self.schedule.n, self.p)
            else:
                code = pop[0][i].code
            info = self.decode(code)
            obj, fit = self.get_obj_fit(info)
            self.pop[0].append(info)
            self.pop[1].append(obj)
            self.pop[2].append(fit)
        self.init_best()
        self.record[1].append(time.perf_counter())
        self.show_generation(0)

    def do_crossover(self, i, j):
        code1, code2 = self.pop[0][i].ga_crossover_sequence_pox(self.pop[0][j])
        self.append_individual(self.decode(code1))
        self.append_individual(self.decode(code2))

    def dislocation(self, i, mode=0):
        code1 = self.pop[0][i].dislocation_operator(mode)
        self.append_individual(self.decode(code1))

    def do_mutation(self, i):
        code1 = self.pop[0][i].ga_mutation_sequence_tpe()
        self.append_individual(self.decode(code1))

    def do_tabu_search(self, i):
        code1 = self.pop[0][i].ts_sequence_operation_based(self.tabu_list[0][i], self.max_tabu)
        self.replace_individual(i, self.decode(code1))
        if len(self.tabu_list[0][i]) >= self.max_tabu:
            for k in range(3):
                self.tabu_list[k][i] = []

    def do_key_block_move(self, i):
        code1 = self.pop[0][i].key_block_move()
        self.replace_individual(i, self.decode(code1))


class GaLwJsp(GaJsp):
    def __init__(self, pop_size, rc, rm, max_generation, objective, schedule, max_stay_generation=None):
        GaJsp.__init__(self, pop_size, rc, rm, max_generation, objective, schedule, max_stay_generation)

    def decode(self, code):
        return self.schedule.decode_limited_wait(code, direction=self.direction)

    def do_init(self, pop=None):
        self.record[0].append(time.perf_counter())
        for i in range(self.pop_size):
            if pop is None:
                code = self.schedule.sequence_operation_based(self.schedule.n, self.p)
            else:
                code = pop[0][i].code
            info = self.decode(code)
            obj, fit = self.get_obj_fit(info)
            self.pop[0].append(info)
            self.pop[1].append(obj)
            self.pop[2].append(fit)
        self.init_best()
        self.record[1].append(time.perf_counter())
        self.show_generation(0)

    def do_crossover(self, i, j):
        code1, code2 = self.pop[0][i].ga_crossover_sequence_ox(self.pop[0][j])
        self.append_individual(self.decode(code1))
        self.append_individual(self.decode(code2))


class GaMrJsp(Ga):
    def __init__(self, pop_size, rc, rm, max_generation, objective, schedule, max_stay_generation=None):
        Ga.__init__(self, pop_size, rc, rm, max_generation, objective, schedule, max_stay_generation)

    def decode(self, code, route):
        return self.schedule.decode(code, route, direction=self.direction)

    def do_init(self, pop=None):
        self.record[0].append(time.perf_counter())
        for i in range(self.pop_size):
            if pop is None:
                route = self.schedule.assignment_route(self.schedule.n, self.r)
                self.update_p(route)
                code = self.schedule.sequence_operation_based(self.schedule.n, self.p)
            else:
                code = pop[0][i].code
                route = pop[0][i].route
            info = self.decode(code, route)
            obj, fit = self.get_obj_fit(info)
            self.pop[0].append(info)
            self.pop[1].append(obj)
            self.pop[2].append(fit)
        self.init_best()
        self.record[1].append(time.perf_counter())
        self.show_generation(0)

    def do_crossover(self, i, j):
        code1, code2 = self.pop[0][i].ga_crossover_sequence_pox(self.pop[0][j])
        route1, route2 = self.pop[0][i].ga_crossover_route(self.pop[0][j])
        self.append_individual(self.decode(code1, route1))
        self.append_individual(self.decode(code2, route2))

    def do_mutation(self, i):
        code1 = self.pop[0][i].ga_mutation_sequence_tpe()
        route1 = self.pop[0][i].ga_mutation_route()
        self.append_individual(self.decode(code1, route1))

    def do_tabu_search(self, i):
        code1 = self.pop[0][i].ts_sequence_operation_based(self.tabu_list[0][i], self.max_tabu)
        self.replace_individual(i, self.decode(code1, self.pop[0][i].route))
        if len(self.tabu_list[0][i]) >= self.max_tabu:
            for k in range(3):
                self.tabu_list[k][i] = []

    def do_key_block_move(self, i):
        code1 = self.pop[0][i].key_block_move()
        self.replace_individual(i, self.decode(code1, self.pop[0][i].route))


class GaFjsp(Ga):
    def __init__(self, pop_size, rc, rm, max_generation, objective, schedule, max_stay_generation=None):
        Ga.__init__(self, pop_size, rc, rm, max_generation, objective, schedule, max_stay_generation)

    def decode(self, code, mac):
        return self.schedule.decode(code, mac, direction=self.direction)

    def do_init(self, pop=None):
        self.record[0].append(time.perf_counter())
        for i in range(self.pop_size):
            if pop is None:
                mac = self.schedule.assignment_job_based(self.schedule.n, self.p, self.tech)
                code = self.schedule.sequence_operation_based(self.schedule.n, self.p)
            else:
                code = pop[0][i].code
                mac = pop[0][i].mac
            info = self.decode(code, mac)
            obj, fit = self.get_obj_fit(info)
            self.pop[0].append(info)
            self.pop[1].append(obj)
            self.pop[2].append(fit)
        self.init_best()
        self.record[1].append(time.perf_counter())
        self.show_generation(0)

    def do_crossover(self, i, j):
        code1, code2 = self.pop[0][i].ga_crossover_sequence_ipox(self.pop[0][j])
        mac1, mac2 = self.pop[0][i].ga_crossover_assignment(self.pop[0][j])
        self.append_individual(self.decode(code1, mac1))
        self.append_individual(self.decode(code2, mac2))

    def do_mutation(self, i):
        code1 = self.pop[0][i].ga_mutation_sequence_tpe()
        mac1 = self.pop[0][i].ga_mutation_assignment(self.tech)
        self.append_individual(self.decode(code1, mac1))

    def do_tabu_search(self, i):
        code1 = self.pop[0][i].ts_sequence_operation_based(self.tabu_list[0][i], self.max_tabu)
        mac1 = self.pop[0][i].ts_assignment_job_based(self.tabu_list[1][i], self.max_tabu)
        self.replace_individual(i, self.decode(code1, mac1))
        if len(self.tabu_list[0][i]) >= self.max_tabu:
            for k in range(3):
                self.tabu_list[k][i] = []

    def do_key_block_move(self, i):
        block = self.pop[0][i].key_block()
        code1 = self.pop[0][i].key_block_move(block)
        mac1 = self.pop[0][i].key_block_move_mac(block)
        self.replace_individual(i, self.decode(code1, mac1))


class GaMrFjsp(Ga):
    def __init__(self, pop_size, rc, rm, max_generation, objective, schedule, max_stay_generation=None):
        Ga.__init__(self, pop_size, rc, rm, max_generation, objective, schedule, max_stay_generation)

    def decode(self, code, mac, route):
        return self.schedule.decode(code, mac, route, direction=self.direction)

    def do_init(self, pop=None):
        self.record[0].append(time.perf_counter())
        self.update_tech(mode=1)
        for i in range(self.pop_size):
            if pop is None:
                route = self.schedule.assignment_route(self.schedule.n, self.r)
                self.update_p(route)
                mac = self.schedule.assignment_job_based_route(self.schedule.n, self.p, self.tech, route)
                code = self.schedule.sequence_operation_based(self.schedule.n, self.p)
            else:
                code = pop[0][i].code
                mac = pop[0][i].mac
                route = pop[0][i].route
            info = self.decode(code, mac, route)
            obj, fit = self.get_obj_fit(info)
            self.pop[0].append(info)
            self.pop[1].append(obj)
            self.pop[2].append(fit)
        self.init_best()
        self.record[1].append(time.perf_counter())
        self.show_generation(0)

    def do_crossover(self, i, j):
        code1, code2 = self.pop[0][i].ga_crossover_sequence_ipox(self.pop[0][j])
        mac1, mac2 = self.pop[0][i].ga_crossover_assignment(self.pop[0][j])
        route1, route2 = self.pop[0][i].ga_crossover_route(self.pop[0][j])
        mac1 = self.pop[0][i].repair_mac_route(mac1, route1)
        mac2 = self.pop[0][j].repair_mac_route(mac2, route2)
        self.append_individual(self.decode(code1, mac1, route1))
        self.append_individual(self.decode(code2, mac2, route2))

    def do_mutation(self, i):
        code1 = self.pop[0][i].ga_mutation_sequence_tpe()
        self.update_tech(mode=0)
        mac1 = self.pop[0][i].ga_mutation_assignment(self.tech)
        route1 = self.pop[0][i].ga_mutation_route()
        mac1 = self.pop[0][i].repair_mac_route(mac1, route1)
        self.append_individual(self.decode(code1, mac1, route1))

    def do_tabu_search(self, i):
        code1 = self.pop[0][i].ts_sequence_operation_based(self.tabu_list[0][i], self.max_tabu)
        mac1 = self.pop[0][i].ts_assignment_job_based(self.tabu_list[1][i], self.max_tabu)
        self.replace_individual(i, self.decode(code1, mac1, self.pop[0][i].route))
        if len(self.tabu_list[0][i]) >= self.max_tabu:
            for k in range(3):
                self.tabu_list[k][i] = []

    def do_key_block_move(self, i):
        block = self.pop[0][i].key_block()
        code1 = self.pop[0][i].key_block_move(block)
        mac1 = self.pop[0][i].key_block_move_mac(block)
        self.replace_individual(i, self.decode(code1, mac1, self.pop[0][i].route))


class GaFjspNew(Ga):
    def __init__(self, pop_size, rc, rm, max_generation, objective, schedule, max_stay_generation=None):
        Ga.__init__(self, pop_size, rc, rm, max_generation, objective, schedule, max_stay_generation)

    def decode(self, code):
        return self.schedule.decode_one(code, direction=self.direction)

    def do_init(self, pop=None):
        self.record[0].append(time.perf_counter())
        for i in range(self.pop_size):
            if pop is None:
                code = self.schedule.sequence_operation_based(self.schedule.n, self.p)
            else:
                code = pop[0][i].code
            info = self.decode(code)
            obj, fit = self.get_obj_fit(info)
            self.pop[0].append(info)
            self.pop[1].append(obj)
            self.pop[2].append(fit)
        self.init_best()
        self.record[1].append(time.perf_counter())
        self.show_generation(0)

    def do_crossover(self, i, j):
        code1, code2 = self.pop[0][i].ga_crossover_sequence_dpox(self.pop[0][j])
        self.append_individual(self.decode(code1))
        self.append_individual(self.decode(code2))

    def do_mutation(self, i):
        code1 = self.pop[0][i].ga_mutation_sequence_tpe()
        self.append_individual(self.decode(code1))

    def do_tabu_search(self, i):
        code1 = self.pop[0][i].ts_sequence_operation_based(self.tabu_list[0][i], self.max_tabu)
        self.replace_individual(i, self.decode(code1))
        if len(self.tabu_list[0][i]) >= self.max_tabu:
            for k in range(3):
                self.tabu_list[k][i] = []

    def do_key_block_move(self, i):
        code1 = self.pop[0][i].key_block_move()
        self.replace_individual(i, self.decode(code1))


class GaMrFjspNew(Ga):
    def __init__(self, pop_size, rc, rm, max_generation, objective, schedule, max_stay_generation=None):
        Ga.__init__(self, pop_size, rc, rm, max_generation, objective, schedule, max_stay_generation)

    def decode(self, code, route):
        return self.schedule.decode_one(code, route, direction=self.direction)

    def do_init(self, pop=None):
        self.record[0].append(time.perf_counter())
        for i in range(self.pop_size):
            if pop is None:
                route = self.schedule.assignment_route(self.schedule.n, self.r)
                self.update_p(route)
                code = self.schedule.sequence_operation_based(self.schedule.n, self.p)
            else:
                code = pop[0][i].code
                route = pop[0][i].route
            info = self.decode(code, route)
            obj, fit = self.get_obj_fit(info)
            self.pop[0].append(info)
            self.pop[1].append(obj)
            self.pop[2].append(fit)
        self.init_best()
        self.record[1].append(time.perf_counter())
        self.show_generation(0)

    def do_crossover(self, i, j):
        code1, code2 = self.pop[0][i].ga_crossover_sequence_dpox(self.pop[0][j])
        route1, route2 = self.pop[0][i].ga_crossover_route(self.pop[0][j])
        self.append_individual(self.decode(code1, route1))
        self.append_individual(self.decode(code2, route2))

    def do_mutation(self, i):
        code1 = self.pop[0][i].ga_mutation_sequence_tpe()
        route1 = self.pop[0][i].ga_mutation_route()
        self.append_individual(self.decode(code1, route1))

    def do_tabu_search(self, i):
        code1 = self.pop[0][i].ts_sequence_operation_based(self.tabu_list[0][i], self.max_tabu)
        self.replace_individual(i, self.decode(code1, self.pop[0][i].route))
        if len(self.tabu_list[0][i]) >= self.max_tabu:
            for k in range(3):
                self.tabu_list[k][i] = []

    def do_key_block_move(self, i):
        code1 = self.pop[0][i].key_block_move()
        self.replace_individual(i, self.decode(code1, self.pop[0][i].route))


class GaFspHfsp(Ga):
    def __init__(self, pop_size, rc, rm, max_generation, objective, schedule, max_stay_generation=None):
        Ga.__init__(self, pop_size, rc, rm, max_generation, objective, schedule, max_stay_generation)

    def decode(self, code):
        return self.schedule.decode(code)

    def do_init(self, pop=None):
        self.record[0].append(time.perf_counter())
        for i in range(self.pop_size):
            if pop is None:
                code = self.schedule.sequence_permutation(self.schedule.n)
            else:
                code = pop[0][i].code
            info = self.decode(code)
            obj, fit = self.get_obj_fit(info)
            self.pop[0].append(info)
            self.pop[1].append(obj)
            self.pop[2].append(fit)
        self.init_best()
        self.record[1].append(time.perf_counter())
        self.show_generation(0)

    def do_crossover(self, i, j):
        code1, code2 = self.pop[0][i].ga_crossover_sequence_permutation_pmx(self.pop[0][j])
        self.append_individual(self.decode(code1))
        self.append_individual(self.decode(code2))

    def do_mutation(self, i):
        code1 = self.pop[0][i].ga_mutation_sequence_permutation_tpe()
        self.append_individual(self.decode(code1))

    def do_tabu_search(self, i):
        code1 = self.pop[0][i].ts_sequence_permutation_based(self.tabu_list[0][i], self.max_tabu)
        self.replace_individual(i, self.decode(code1))
        if len(self.tabu_list[0][i]) >= self.max_tabu:
            for k in range(3):
                self.tabu_list[k][i] = []


class GaFspHfspWorkTimetable(GaFspHfsp):
    def __init__(self, pop_size, rc, rm, max_generation, objective, schedule, max_stay_generation=None):
        GaFspHfsp.__init__(self, pop_size, rc, rm, max_generation, objective, schedule, max_stay_generation)

    def decode(self, code):
        return self.schedule.decode_work_timetable(code)
