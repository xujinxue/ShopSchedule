__doc__ = """
Genetic algorithm: 遗传算法
class Ga: 定义的基类
class GaJsp(De): Jsp的Ga, 重载了***
...
"""

import copy
import time

import numpy as np

from ..define import Selection
from ..resource.code import Code
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
        self.worker = [[task.worker for task in job.task.values()] for job in self.schedule.job.values()]
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

    def decode(self, code, mac=None, route=None, wok=None):
        pass

    def dislocation(self, i, direction=0):
        code1 = self.pop[0][i].dislocation_operator(direction)
        self.append_individual(self.decode(code1, self.pop[0][i].mac, self.pop[0][i].route, self.pop[0][i].wok))

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

    def selection_roulette(self):
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

    def selection_champion2(self):
        pop = deepcopy(self.pop)
        tabu_list = deepcopy(self.tabu_list)
        self.pop = [[], [], []]
        self.tabu_list = [[[] for _ in self.individual], [[] for _ in self.individual], [[] for _ in self.individual]]
        for i in range(self.pop_size):
            a = np.random.choice(range(self.pop_size), 2, replace=False)
            j = a[0] if pop[2][a[0]] > pop[2][a[1]] else a[1]
            self.pop[0].append(pop[0][j])
            self.pop[1].append(pop[1][j])
            self.pop[2].append(pop[2][j])
            for k in range(3):
                self.tabu_list[k][i] = tabu_list[k][j]

    def save_best(self):
        self.pop[0][0] = self.best[0]
        self.pop[1][0] = self.best[1]
        self.pop[2][0] = self.best[2]
        for k in range(3):
            self.tabu_list[k][0] = self.best[3][k]
        # index = self.pop[2].index(max(self.pop[2]))
        # self.pop[0][index] = self.best[0]
        # self.pop[1][index] = self.best[1]
        # self.pop[2][index] = self.best[2]
        # for k in range(3):
        #     self.tabu_list[k][index] = self.best[3][k]

    def do_selection(self):
        func_dict = {
            Selection.default: self.selection_roulette,
            Selection.roulette: self.selection_roulette,
            Selection.champion2: self.selection_champion2,
        }
        func = func_dict[self.schedule.ga_operator[Selection.name]]
        func()
        self.save_best()

    def do_init(self, pop=None):
        pass

    def do_crossover(self, i, j, p):
        pass

    def do_mutation(self, i, p):
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

    def do_evolution(self, pop=None, exp_no=None):
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
                if self.schedule.para_key_block_move:
                    self.do_key_block_move(i)
                if self.schedule.para_tabu:
                    self.do_tabu_search(i)
                p, q = np.random.random(3), np.random.random(3)
                j = np.random.choice(np.delete(np.arange(self.pop_size), i), 1, replace=False)[0]
                if p[0] < self.rc:
                    j = np.random.choice(np.delete(np.arange(self.pop_size), i), 1, replace=False)[0]
                    code1, code2 = self.pop[0][i].code, self.pop[0][j].code
                    if self.schedule.para_dislocation and Utils.similarity(code1, code2) >= 0.5:
                        self.dislocation(i, direction=0)
                        self.dislocation(j, direction=1)
                        p[0] = 1
                self.do_crossover(i, j, p)
                self.do_mutation(i, q)
            self.do_selection()
            self.record[1].append(time.perf_counter())
            self.show_generation(g)
        Utils.print("{}Evolution {} finish{}".format("=" * 48, exp_no, "=" * 48), fore=Utils.fore().LIGHTRED_EX)


class GaJsp(Ga):
    def __init__(self, pop_size, rc, rm, max_generation, objective, schedule, max_stay_generation=None):
        Ga.__init__(self, pop_size, rc, rm, max_generation, objective, schedule, max_stay_generation)

    def decode(self, code, mac=None, route=None, wok=None):
        return self.schedule.decode(code, direction=self.direction)

    def do_init(self, pop=None):
        self.record[0].append(time.perf_counter())
        for i in range(self.pop_size):
            if pop is None:
                code = Code.sequence_operation_based(self.schedule.n, self.p)
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

    def do_crossover(self, i, j, p):
        if p[0] < self.rc:
            code1, code2 = self.pop[0][i].ga_crossover_sequence(self.pop[0][j])
            self.append_individual(self.decode(code1))
            self.append_individual(self.decode(code2))

    def do_mutation(self, i, p):
        if p[0] < self.rm:
            code1 = self.pop[0][i].ga_mutation_sequence()
            self.append_individual(self.decode(code1))

    def do_tabu_search(self, i):
        code1 = self.pop[0][i].ts_sequence_operation_based(self.tabu_list[0][i], self.max_tabu)
        self.replace_individual(i, self.decode(code1))
        if len(self.tabu_list[0][i]) >= self.max_tabu:
            for k in range(3):
                self.tabu_list[k][i] = []

    def do_key_block_move(self, i):
        # code1 = self.pop[0][i].key_block_move()
        code1 = self.pop[0][i].key_block_move_complete()
        self.replace_individual(i, self.decode(code1))
        # code1_complete = self.pop[0][i].key_block_move_complete()
        # for code1 in code1_complete:
        #     self.replace_individual(i, self.decode(code1[0]))


class GaLwJsp(GaJsp):
    def __init__(self, pop_size, rc, rm, max_generation, objective, schedule, max_stay_generation=None):
        GaJsp.__init__(self, pop_size, rc, rm, max_generation, objective, schedule, max_stay_generation)

    def decode(self, code, mac=None, route=None, wok=None):
        return self.schedule.decode_limited_wait(code, direction=self.direction)


class GaMrJsp(Ga):
    def __init__(self, pop_size, rc, rm, max_generation, objective, schedule, max_stay_generation=None):
        Ga.__init__(self, pop_size, rc, rm, max_generation, objective, schedule, max_stay_generation)

    def decode(self, code, mac=None, route=None, wok=None):
        return self.schedule.decode(code, route, direction=self.direction)

    def do_init(self, pop=None):
        self.record[0].append(time.perf_counter())
        for i in range(self.pop_size):
            if pop is None:
                route = Code.assignment_route(self.schedule.n, self.r)
                self.update_p(route)
                code = Code.sequence_operation_based(self.schedule.n, self.p)
            else:
                code = pop[0][i].code
                route = pop[0][i].route
            info = self.decode(code, route=route)
            obj, fit = self.get_obj_fit(info)
            self.pop[0].append(info)
            self.pop[1].append(obj)
            self.pop[2].append(fit)
        self.init_best()
        self.record[1].append(time.perf_counter())
        self.show_generation(0)

    def do_crossover(self, i, j, p):
        if p[0] < self.rc or p[1] < self.rc:
            if p[0] < self.rc:
                code1, code2 = self.pop[0][i].ga_crossover_sequence(self.pop[0][j])
            else:
                code1, code2 = self.pop[0][i].code, self.pop[0][j].code
            if p[1] < self.rc:
                route1, route2 = self.pop[0][i].ga_crossover_route(self.pop[0][j])
            else:
                route1, route2 = self.pop[0][i].route, self.pop[0][j].route
            self.append_individual(self.decode(code1, route=route1))
            self.append_individual(self.decode(code2, route=route2))

    def do_mutation(self, i, p):
        if p[0] < self.rm or p[1] < self.rm:
            code1 = self.pop[0][i].ga_mutation_sequence() if p[0] < self.rm else self.pop[0][i].code
            route1 = self.pop[0][i].ga_mutation_route() if p[1] < self.rm else self.pop[0][i].route
            self.append_individual(self.decode(code1, route=route1))

    def do_tabu_search(self, i):
        code1 = self.pop[0][i].ts_sequence_operation_based(self.tabu_list[0][i], self.max_tabu)
        self.replace_individual(i, self.decode(code1, route=self.pop[0][i].route))
        if len(self.tabu_list[0][i]) >= self.max_tabu:
            for k in range(3):
                self.tabu_list[k][i] = []

    def do_key_block_move(self, i):
        # code1 = self.pop[0][i].key_block_move()
        code1 = self.pop[0][i].key_block_move_complete()
        self.replace_individual(i, self.decode(code1, route=self.pop[0][i].route))


class GaJspNew(GaJsp):
    def __init__(self, pop_size, rc, rm, max_generation, objective, schedule, max_stay_generation=None):
        GaJsp.__init__(self, pop_size, rc, rm, max_generation, objective, schedule, max_stay_generation)

    def decode(self, code, mac=None, route=None, wok=None):
        return self.schedule.decode_new(code, direction=self.direction)


class GaLwJspNew(GaJsp):
    def __init__(self, pop_size, rc, rm, max_generation, objective, schedule, max_stay_generation=None):
        GaJsp.__init__(self, pop_size, rc, rm, max_generation, objective, schedule, max_stay_generation)

    def decode(self, code, mac=None, route=None, wok=None):
        return self.schedule.decode_new_twice(code, direction=self.direction)


class GaLwJspNew2(GaJsp):
    def __init__(self, pop_size, rc, rm, max_generation, objective, schedule, max_stay_generation=None):
        GaJsp.__init__(self, pop_size, rc, rm, max_generation, objective, schedule, max_stay_generation)

    def decode(self, code, mac=None, route=None, wok=None):
        return self.schedule.decode_limited_wait_new_twice(code, direction=self.direction)


class GaMrJspNew(GaMrJsp):
    def __init__(self, pop_size, rc, rm, max_generation, objective, schedule, max_stay_generation=None):
        GaMrJsp.__init__(self, pop_size, rc, rm, max_generation, objective, schedule, max_stay_generation)

    def decode(self, code, mac=None, route=None, wok=None):
        return self.schedule.decode_new(code, route=route, direction=self.direction)


class GaFjsp(Ga):
    def __init__(self, pop_size, rc, rm, max_generation, objective, schedule, max_stay_generation=None):
        Ga.__init__(self, pop_size, rc, rm, max_generation, objective, schedule, max_stay_generation)

    def decode(self, code, mac=None, route=None, wok=None):
        return self.schedule.decode(code, mac, direction=self.direction)

    def do_init(self, pop=None):
        self.record[0].append(time.perf_counter())
        for i in range(self.pop_size):
            if pop is None:
                code = Code.sequence_operation_based(self.schedule.n, self.p)
                # mac = Code.assignment_job_based(self.schedule.n, self.p, self.tech)
                # info = self.decode(code, mac=mac)
                info = self.schedule.decode_new(code)
            else:
                code = pop[0][i].code
                mac = pop[0][i].mac
                info = self.decode(code, mac=mac)
            obj, fit = self.get_obj_fit(info)
            self.pop[0].append(info)
            self.pop[1].append(obj)
            self.pop[2].append(fit)
        self.init_best()
        self.record[1].append(time.perf_counter())
        self.show_generation(0)

    def do_crossover(self, i, j, p):
        if p[0] < self.rc or p[1] < self.rc:
            if p[0] < self.rc:
                code1, code2 = self.pop[0][i].ga_crossover_sequence(self.pop[0][j])
            else:
                code1, code2 = self.pop[0][i].code, self.pop[0][j].code
            if p[1] < self.rc:
                mac1, mac2 = self.pop[0][i].ga_crossover_assignment(self.pop[0][j])
            else:
                mac1, mac2 = self.pop[0][i].mac, self.pop[0][j].mac
            self.append_individual(self.decode(code1, mac=mac1))
            self.append_individual(self.decode(code2, mac=mac2))

    def do_mutation(self, i, p):
        if p[0] < self.rm or p[1] < self.rm:
            code1 = self.pop[0][i].ga_mutation_sequence() if p[0] < self.rm else self.pop[0][i].code
            mac1 = self.pop[0][i].ga_mutation_assignment(self.tech) if p[1] < self.rm else self.pop[0][i].mac
            self.append_individual(self.decode(code1, mac=mac1))

    def do_tabu_search(self, i):
        code1 = self.pop[0][i].ts_sequence_operation_based(self.tabu_list[0][i], self.max_tabu)
        mac1 = self.pop[0][i].ts_assignment_job_based(self.tabu_list[1][i], self.max_tabu)
        self.replace_individual(i, self.decode(code1, mac=mac1))
        if len(self.tabu_list[0][i]) >= self.max_tabu:
            for k in range(3):
                self.tabu_list[k][i] = []

    def do_key_block_move(self, i):
        # block = self.pop[0][i].key_block()
        block = self.pop[0][i].key_block_complete()
        code1 = self.pop[0][i].key_block_move(block)
        mac1 = self.pop[0][i].key_block_move_mac(block)
        self.replace_individual(i, self.decode(code1, mac=mac1))


class GaLwFjsp(GaFjsp):
    def __init__(self, pop_size, rc, rm, max_generation, objective, schedule, max_stay_generation=None):
        GaFjsp.__init__(self, pop_size, rc, rm, max_generation, objective, schedule, max_stay_generation)

    def decode(self, code, mac=None, route=None, wok=None):
        return self.schedule.decode_limited_wait(code, mac, direction=self.direction)


class GaMrFjsp(Ga):
    def __init__(self, pop_size, rc, rm, max_generation, objective, schedule, max_stay_generation=None):
        Ga.__init__(self, pop_size, rc, rm, max_generation, objective, schedule, max_stay_generation)

    def decode(self, code, mac=None, route=None, wok=None):
        return self.schedule.decode(code, mac, route, direction=self.direction)

    def do_init(self, pop=None):
        self.record[0].append(time.perf_counter())
        # self.update_tech(mode=1)
        for i in range(self.pop_size):
            if pop is None:
                route = Code.assignment_route(self.schedule.n, self.r)
                self.update_p(route)
                code = Code.sequence_operation_based(self.schedule.n, self.p)
                # mac = Code.assignment_job_based_route(self.schedule.n, self.p, self.tech, route)
                # info = self.decode(code, mac, route)
                info = self.schedule.decode_new(code, route)
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

    def do_crossover(self, i, j, p):
        if p[0] < self.rc or p[1] < self.rc or p[2] < self.rc:
            if p[0] < self.rc:
                code1, code2 = self.pop[0][i].ga_crossover_sequence(self.pop[0][j])
            else:
                code1, code2 = self.pop[0][i].code, self.pop[0][j].code
            if p[1] < self.rc:
                mac1, mac2 = self.pop[0][i].ga_crossover_assignment(self.pop[0][j])
            else:
                mac1, mac2 = self.pop[0][i].mac, self.pop[0][j].mac
            if p[2] < self.rc:
                route1, route2 = self.pop[0][i].ga_crossover_route(self.pop[0][j])
            else:
                route1, route2 = self.pop[0][i].route, self.pop[0][i].route
            mac1 = self.pop[0][i].repair_mac_route(mac1, route1)
            mac2 = self.pop[0][j].repair_mac_route(mac2, route2)
            self.append_individual(self.decode(code1, mac1, route1))
            self.append_individual(self.decode(code2, mac2, route2))

    def do_mutation(self, i, p):
        if p[0] < self.rm or p[1] < self.rm or p[2] < self.rm:
            self.update_tech(mode=0)
            code1 = self.pop[0][i].ga_mutation_sequence() if p[0] < self.rm else self.pop[0][i].code
            mac1 = self.pop[0][i].ga_mutation_assignment(self.tech) if p[1] < self.rm else self.pop[0][i].mac
            route1 = self.pop[0][i].ga_mutation_route() if p[2] < self.rm else self.pop[0][i].route
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
        # block = self.pop[0][i].key_block()
        block = self.pop[0][i].key_block_complete()
        code1 = self.pop[0][i].key_block_move(block)
        mac1 = self.pop[0][i].key_block_move_mac(block)
        self.replace_individual(i, self.decode(code1, mac1, self.pop[0][i].route))


class GaDrcFjsp(Ga):
    def __init__(self, pop_size, rc, rm, max_generation, objective, schedule, max_stay_generation=None):
        Ga.__init__(self, pop_size, rc, rm, max_generation, objective, schedule, max_stay_generation)

    def decode(self, code, mac=None, route=None, wok=None):
        return self.schedule.decode_worker(code, mac, wok, direction=self.direction)

    def do_init(self, pop=None):
        self.record[0].append(time.perf_counter())
        for i in range(self.pop_size):
            if pop is None:
                code = Code.sequence_operation_based(self.schedule.n, self.p)
                # mac = Code.assignment_job_based(self.schedule.n, self.p, self.tech)
                # wok = Code.assignment_worker(self.schedule.n, self.p, self.tech, self.worker, mac)
                # info = self.decode(code, mac, wok=wok)
                info = self.schedule.decode_worker_new(code)
            else:
                code = pop[0][i].code
                mac = pop[0][i].mac
                wok = pop[0][i].wok
                info = self.decode(code, mac, wok=wok)
            obj, fit = self.get_obj_fit(info)
            self.pop[0].append(info)
            self.pop[1].append(obj)
            self.pop[2].append(fit)
        self.init_best()
        self.record[1].append(time.perf_counter())
        self.show_generation(0)

    def do_crossover(self, i, j, p):
        if p[0] < self.rc or p[1] < self.rc or p[2] < self.rc:
            if p[0] < self.rc:
                code1, code2 = self.pop[0][i].ga_crossover_sequence(self.pop[0][j])
            else:
                code1, code2 = self.pop[0][i].code, self.pop[0][j].code
            if p[1] < self.rc:
                mac1, mac2 = self.pop[0][i].ga_crossover_assignment(self.pop[0][j])
            else:
                mac1, mac2 = self.pop[0][i].mac, self.pop[0][j].mac
            if p[2] < self.rc:
                wok1, wok2 = self.pop[0][i].ga_crossover_worker(self.pop[0][j])
            else:
                wok1, wok2 = self.pop[0][i].wok, self.pop[0][j].wok
            wok1 = self.pop[0][i].repair_mac_wok(mac1, wok1)
            wok2 = self.pop[0][j].repair_mac_wok(mac2, wok2)
            self.append_individual(self.decode(code1, mac1, wok=wok1))
            self.append_individual(self.decode(code2, mac2, wok=wok2))

    def do_mutation(self, i, p):
        if p[0] < self.rm or p[1] < self.rm or p[2] < self.rm:
            code1 = self.pop[0][i].ga_mutation_sequence() if p[0] < self.rm else self.pop[0][i].code
            mac1 = self.pop[0][i].ga_mutation_assignment(self.tech) if p[1] < self.rm else self.pop[0][i].mac
            wok1 = self.pop[0][i].ga_mutation_worker(self.pop[0][i].mac) if p[2] < self.rm else self.pop[0][i].wok
            wok1 = self.pop[0][i].repair_mac_wok(mac1, wok1)
            self.append_individual(self.decode(code1, mac1, wok=wok1))

    def do_tabu_search(self, i):
        code1 = self.pop[0][i].ts_sequence_operation_based(self.tabu_list[0][i], self.max_tabu)
        mac1 = self.pop[0][i].ts_assignment_job_based(self.tabu_list[1][i], self.max_tabu)
        wok1 = self.pop[0][i].ts_wok_job_based(self.pop[0][i].mac, self.tabu_list[2][i], self.max_tabu)
        wok1 = self.pop[0][i].repair_mac_wok(mac1, wok1)
        self.replace_individual(i, self.decode(code1, mac1, wok=wok1))
        if len(self.tabu_list[0][i]) >= self.max_tabu:
            for k in range(3):
                self.tabu_list[k][i] = []

    def do_key_block_move(self, i):
        block = self.pop[0][i].key_block(self.pop[0][i].key_route_worker)
        # code1 = self.pop[0][i].key_block_move(block)
        code1 = self.pop[0][i].key_block_move_complete(block)
        mac1 = self.pop[0][i].key_block_move_mac(block)
        wok1 = self.pop[0][i].key_block_move_wok(self.pop[0][i].mac, block)
        wok1 = self.pop[0][i].repair_mac_wok(mac1, wok1)
        self.replace_individual(i, self.decode(code1, mac1, wok=wok1))


class GaFjspNew(Ga):
    def __init__(self, pop_size, rc, rm, max_generation, objective, schedule, max_stay_generation=None):
        Ga.__init__(self, pop_size, rc, rm, max_generation, objective, schedule, max_stay_generation)

    def decode(self, code, mac=None, route=None, wok=None):
        return self.schedule.decode_new(code, direction=self.direction)

    def do_init(self, pop=None):
        self.record[0].append(time.perf_counter())
        for i in range(self.pop_size):
            if pop is None:
                code = Code.sequence_operation_based(self.schedule.n, self.p)
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

    def do_crossover(self, i, j, p):
        if p[0] < self.rc:
            code1, code2 = self.pop[0][i].ga_crossover_sequence(self.pop[0][j])
            self.append_individual(self.decode(code1))
            self.append_individual(self.decode(code2))

    def do_mutation(self, i, p):
        if p[0] < self.rm:
            code1 = self.pop[0][i].ga_mutation_sequence()
            self.append_individual(self.decode(code1))

    def do_tabu_search(self, i):
        code1 = self.pop[0][i].ts_sequence_operation_based(self.tabu_list[0][i], self.max_tabu)
        self.replace_individual(i, self.decode(code1))
        if len(self.tabu_list[0][i]) >= self.max_tabu:
            for k in range(3):
                self.tabu_list[k][i] = []

    def do_key_block_move(self, i):
        # code1 = self.pop[0][i].key_block_move()
        code1 = self.pop[0][i].key_block_move_complete()
        self.replace_individual(i, self.decode(code1))


class GaLwFjspNew(GaFjspNew):
    def __init__(self, pop_size, rc, rm, max_generation, objective, schedule, max_stay_generation=None):
        GaFjspNew.__init__(self, pop_size, rc, rm, max_generation, objective, schedule, max_stay_generation)

    def decode(self, code, mac=None, route=None, wok=None):
        return self.schedule.decode_limited_wait_new(code, direction=self.direction)


class GaDrcFjspNew(GaFjspNew):
    def __init__(self, pop_size, rc, rm, max_generation, objective, schedule, max_stay_generation=None):
        GaFjspNew.__init__(self, pop_size, rc, rm, max_generation, objective, schedule, max_stay_generation)

    def decode(self, code, mac=None, route=None, wok=None):
        return self.schedule.decode_worker_new(code, direction=self.direction)


class GaMrFjspNew(Ga):
    def __init__(self, pop_size, rc, rm, max_generation, objective, schedule, max_stay_generation=None):
        Ga.__init__(self, pop_size, rc, rm, max_generation, objective, schedule, max_stay_generation)

    def decode(self, code, mac=None, route=None, wok=None):
        return self.schedule.decode_new(code, route, direction=self.direction)

    def do_init(self, pop=None):
        self.record[0].append(time.perf_counter())
        for i in range(self.pop_size):
            if pop is None:
                route = Code.assignment_route(self.schedule.n, self.r)
                self.update_p(route)
                code = Code.sequence_operation_based(self.schedule.n, self.p)
            else:
                code = pop[0][i].code
                route = pop[0][i].route
            info = self.decode(code, route=route)
            obj, fit = self.get_obj_fit(info)
            self.pop[0].append(info)
            self.pop[1].append(obj)
            self.pop[2].append(fit)
        self.init_best()
        self.record[1].append(time.perf_counter())
        self.show_generation(0)

    def do_crossover(self, i, j, p):
        if p[0] < self.rc or p[1] < self.rc:
            if p[0] < self.rc:
                code1, code2 = self.pop[0][i].ga_crossover_sequence(self.pop[0][j])
            else:
                code1, code2 = self.pop[0][i].code, self.pop[0][j].code
            if p[1] < self.rc:
                route1, route2 = self.pop[0][i].ga_crossover_route(self.pop[0][j])
            else:
                route1, route2 = self.pop[0][i].route, self.pop[0][j].route
            self.append_individual(self.decode(code1, route=route1))
            self.append_individual(self.decode(code2, route=route2))

    def do_mutation(self, i, p):
        if p[0] < self.rm or p[1] < self.rm:
            code1 = self.pop[0][i].ga_mutation_sequence() if p[0] < self.rm else self.pop[0][i].code
            route1 = self.pop[0][i].ga_mutation_route() if p[1] < self.rm else self.pop[0][i].route
            self.append_individual(self.decode(code1, route=route1))

    def do_tabu_search(self, i):
        code1 = self.pop[0][i].ts_sequence_operation_based(self.tabu_list[0][i], self.max_tabu)
        self.replace_individual(i, self.decode(code1, route=self.pop[0][i].route))
        if len(self.tabu_list[0][i]) >= self.max_tabu:
            for k in range(3):
                self.tabu_list[k][i] = []

    def do_key_block_move(self, i):
        # code1 = self.pop[0][i].key_block_move()
        code1 = self.pop[0][i].key_block_move_complete()
        self.replace_individual(i, self.decode(code1, route=self.pop[0][i].route))


class GaFspHfsp(Ga):
    def __init__(self, pop_size, rc, rm, max_generation, objective, schedule, max_stay_generation=None):
        Ga.__init__(self, pop_size, rc, rm, max_generation, objective, schedule, max_stay_generation)

    def decode(self, code, mac=None, route=None, wok=None):
        return self.schedule.decode(code)

    def do_init(self, pop=None):
        self.record[0].append(time.perf_counter())
        for i in range(self.pop_size):
            if pop is None:
                code = Code.sequence_permutation(self.schedule.n)
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

    def do_crossover(self, i, j, p):
        if p[0] < self.rc:
            code1, code2 = self.pop[0][i].ga_crossover_sequence_permutation(self.pop[0][j])
            self.append_individual(self.decode(code1))
            self.append_individual(self.decode(code2))

    def do_mutation(self, i, p):
        if p[0] < self.rm:
            code1 = self.pop[0][i].ga_mutation_sequence_permutation()
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

    def decode(self, code, mac=None, route=None, wok=None):
        return self.schedule.decode_work_timetable(code)
