__doc__ = """
Genetic algorithm: 遗传算法
class Ga: 定义的基类
class GaJsp(De): Jsp的Ga, 重载了***
...
"""

import copy
import time

import numpy as np

from ..define import k1, k2, k3, k4, RATE_ACCEPT_WORSE
from ..utils import Utils

deepcopy = copy.deepcopy


class Ga:
    def __init__(self, pop_size, rc, rm, max_generation, objective, schedule):
        """
        初始化参数
        :param pop_size: 种群规模
        :param rc: 交叉概率
        :param rm: 变异概率
        :param max_generation: 最大迭代次数
        :param objective: 求解目标值的函数
        :param schedule: 调度对象
        """
        self.pop_size = pop_size
        self.rc = rc
        self.rm = rm
        self.t0 = 1000
        self.t = 1000
        self.alpha = 0.95
        self.max_generation = max_generation
        self.objective = objective
        self.schedule = schedule
        self.p = [job.nop for job in self.schedule.job.values()]
        self.tech = [[task.machine for task in job.task.values()] for job in self.schedule.job.values()]
        self.best = [None, None, None, [[], [], []]]  # (info, objective, fitness, tabu)
        self.pop = [[], [], []]  # (info, objective, fitness)
        # (start, end, best_objective, best_fitness, worst_fitness, mean_fitness)
        self.record = [[], [], [], [], [], []]
        # (code, mac, tech)
        self.max_tabu = Utils.len_tabu(self.schedule.m, self.schedule.n)
        individual = range(self.pop_size)
        self.tabu_list = [[[] for _ in individual], [[] for _ in individual], [[] for _ in individual]]

    def clear(self):
        self.t = self.t0
        self.best = [None, None, None, [[], [], []]]
        self.pop = [[], [], []]
        self.record = [[], [], [], [], [], []]
        individual = range(self.pop_size)
        self.tabu_list = [[[] for _ in individual], [[] for _ in individual], [[] for _ in individual]]

    def update_t(self):
        self.t *= self.alpha

    def update_info(self, i, fit_new):
        p = np.exp(-np.abs(fit_new - self.pop[2][i]) / self.t)
        return True if np.random.random() < p else False

    def update_individual(self, i, obj_new, info_new):
        fit_new = Utils.calculate_fitness(obj_new)
        if Utils.update_info(self.pop[1][i], obj_new) or np.random.random() < RATE_ACCEPT_WORSE:
            self.pop[0][i] = info_new
            self.pop[1][i] = obj_new
            self.pop[2][i] = fit_new
            for k in range(3):
                self.tabu_list[k][i] = []
        if Utils.update_info(self.best[1], obj_new):
            self.best[0] = info_new
            self.best[1] = obj_new
            self.best[2] = fit_new
            for k in range(3):
                self.best[3][k].append(self.tabu_list[k][i])

    def adaptive_rc_rm_s(self, i, j):
        f_max, f_avg = max(self.pop[2]), np.mean(self.pop[2])
        f = max([self.pop[2][i], self.pop[2][j]])
        rc, rm = k2, k4
        if f > f_avg:
            rc = k1 * np.sin(np.pi * (f_max - f) / (2 * (f_max - f_avg)))
        if self.pop[2][i] > f_avg:
            rm = k3 * np.sin(np.pi * (f_max - self.pop[2][i]) / (f_max - f_avg))
        return rc, rm

    def adaptive_rc_rm_c(self, i, j):
        f_max, f_avg = max(self.pop[2]), np.mean(self.pop[2])
        f = max([self.pop[2][i], self.pop[2][j]])
        rc, rm = k2, k4
        if f > f_avg:
            rc = 1 - k1 * np.cos(np.pi * (f_max - f) / (2 * (f_max - f_avg)))
        if self.pop[2][i] > f_avg:
            rm = 1 - k3 * np.cos(np.pi * (f_max - self.pop[2][i]) / (f_max - f_avg))
        return rc, rm

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

    def reach_best_known_solution(self):
        if self.schedule.best_known is not None and self.best[1] <= self.schedule.best_known:
            return True
        return False

    def do_selection(self):
        a = np.array(self.pop[2]) / sum(self.pop[2])
        b = np.array([])
        for i in range(self.pop_size):
            b = np.append(b, sum(a[:i + 1]))
        pop = deepcopy(self.pop)
        tabu_list = deepcopy(self.tabu_list)
        for i in range(self.pop_size):
            j = np.argwhere(b > np.random.random())[0, 0]  # 轮盘赌选择
            self.pop[0][i] = pop[0][j]
            self.pop[1][i] = pop[1][j]
            self.pop[2][i] = pop[2][j]
            for k in range(3):
                self.tabu_list[k][i] = tabu_list[k][j]
        self.pop[0][0] = deepcopy(self.best[0])
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

    def do_evolution(self, tabu_search=True, key_block_move=False, pop=None):
        Utils.print("{}Evolution  start{}".format("=" * 48, "=" * 48), fore=Utils.fore().LIGHTYELLOW_EX)
        self.clear()
        self.do_init(pop)
        self.do_selection()
        for g in range(1, self.max_generation + 1):
            if self.reach_best_known_solution():
                break
            self.record[0].append(time.perf_counter())
            for i in range(self.pop_size):
                if self.reach_best_known_solution():
                    break
                # j = np.random.choice(np.delete(np.arange(self.pop_size), i), 1, replace=False)[0]
                # rc, rm = self.adaptive_rc_rm_s(i, j)
                # if np.random.random() < rc:
                if np.random.random() < self.rc:
                    j = np.random.choice(np.delete(np.arange(self.pop_size), i), 1, replace=False)[0]
                    self.do_crossover(i, j)
                # if np.random.random() < rm:
                if np.random.random() < self.rm:
                    self.do_mutation(i)
                if tabu_search:
                    self.do_tabu_search(i)
                if key_block_move:
                    self.do_key_block_move(i)
            self.do_selection()
            self.update_t()
            self.record[1].append(time.perf_counter())
            self.show_generation(g)
        Utils.print("{}Evolution finish{}".format("=" * 48, "=" * 48), fore=Utils.fore().LIGHTRED_EX)


class GaJsp(Ga):
    def __init__(self, pop_size, rc, rm, max_generation, objective, schedule):
        Ga.__init__(self, pop_size, rc, rm, max_generation, objective, schedule)

    def decode_update(self, i, code, route=None):
        info = self.schedule.decode_operation_based_active(code, route)
        self.update_individual(i, self.objective(info), info)

    def do_init(self, pop=None):
        self.record[0].append(time.perf_counter())
        for i in range(self.pop_size):
            if pop is None:
                code = self.schedule.sequence_operation_based(self.schedule.n, self.p)
            else:
                code = pop[0][i].code
            info = self.schedule.decode_operation_based_active(code)
            self.pop[0].append(info)
            self.pop[1].append(self.objective(info))
            self.pop[2].append(Utils.calculate_fitness(self.pop[1][i]))
        self.init_best()
        self.record[1].append(time.perf_counter())
        self.show_generation(0)

    def do_crossover(self, i, j):
        code1, code2 = self.pop[0][i].ga_crossover_sequence_hybrid(self.pop[0][j])
        self.decode_update(i, code1)
        self.decode_update(j, code2)

    def do_mutation(self, i):
        code1 = self.pop[0][i].ga_mutation_sequence_operation_based_hybrid()
        self.decode_update(i, code1)

    def do_tabu_search(self, i):
        code1 = self.pop[0][i].ts_sequence_operation_based(self.tabu_list[0][i], self.max_tabu)
        self.decode_update(i, code1)
        if len(self.tabu_list[0][i]) >= self.max_tabu:
            self.tabu_list[0][i] = []

    def do_key_block_move(self, i):
        code1 = self.pop[0][i].method_sequence_key_block_move()
        self.decode_update(i, code1, self.pop[0][i].route)


class GaFrJsp(GaJsp):
    def __init__(self, pop_size, rc, rm, max_generation, objective, schedule):
        GaJsp.__init__(self, pop_size, rc, rm, max_generation, objective, schedule)

    def do_init(self, pop=None):
        self.record[0].append(time.perf_counter())
        for i in range(self.pop_size):
            if pop is None:
                code = self.schedule.sequence_operation_based(self.schedule.n, self.p)
                route = self.schedule.route_job_based(self.schedule.n, self.p)
            else:
                code = pop[0][i].code
                route = pop[0][i].route
            info = self.schedule.decode_operation_based_active(code, route)
            self.pop[0].append(info)
            self.pop[1].append(self.objective(info))
            self.pop[2].append(Utils.calculate_fitness(self.pop[1][i]))
        self.init_best()
        self.record[1].append(time.perf_counter())
        self.show_generation(0)

    def do_crossover(self, i, j):
        code1, code2 = self.pop[0][i].ga_crossover_sequence_hybrid(self.pop[0][j])
        route1, route2 = self.pop[0][i].ga_crossover_route_pmx(self.pop[0][j])
        self.decode_update(i, code1, route1)
        self.decode_update(j, code1, route2)

    def do_mutation(self, i):
        code1 = self.pop[0][i].ga_mutation_sequence_operation_based_hybrid()
        route1 = self.pop[0][i].ga_mutation_route_tpe()
        self.decode_update(i, code1, route1)

    def do_tabu_search(self, i):
        code1 = self.pop[0][i].ts_sequence_operation_based(self.tabu_list[0][i], self.max_tabu)
        route1 = self.pop[0][i].ts_route_job_based(self.tabu_list[2][i], self.max_tabu)
        self.decode_update(i, code1, route1)
        for u in [0, 2]:
            if len(self.tabu_list[u][i]) >= self.max_tabu:
                self.tabu_list[u][i] = []


class GaNwJsp(Ga):
    def __init__(self, pop_size, rc, rm, max_generation, objective, schedule):
        Ga.__init__(self, pop_size, rc, rm, max_generation, objective, schedule)

    def decode_update(self, i, code, route=None):
        info = self.schedule.decode_no_wait_active(code, self.p, route)
        self.update_individual(i, self.objective(info), info)

    def do_init(self, pop=None):
        self.record[0].append(time.perf_counter())
        for i in range(self.pop_size):
            if pop is None:
                code = self.schedule.sequence_permutation(self.schedule.n)
            else:
                code = pop[0][i].code
            info = self.schedule.decode_no_wait_active(code, self.p)
            self.pop[0].append(info)
            self.pop[1].append(self.objective(info))
            self.pop[2].append(Utils.calculate_fitness(self.pop[1][i]))
        self.init_best()
        self.record[1].append(time.perf_counter())
        self.show_generation(0)

    def do_crossover(self, i, j):
        code1, code2 = self.pop[0][i].ga_crossover_sequence_permutation_ox(self.pop[0][j])
        self.decode_update(i, code1)
        self.decode_update(j, code2)

    def do_mutation(self, i):
        code1 = self.pop[0][i].ga_mutation_sequence_permutation_hybrid()
        self.decode_update(i, code1)

    def do_tabu_search(self, i):
        code1 = self.pop[0][i].ts_sequence_permutation(self.tabu_list[0][i], self.max_tabu)
        self.decode_update(i, code1)
        if len(self.tabu_list[0][i]) >= self.max_tabu:
            self.tabu_list[0][i] = []


class GaLwJsp(GaJsp):
    def __init__(self, pop_size, rc, rm, max_generation, objective, schedule):
        GaJsp.__init__(self, pop_size, rc, rm, max_generation, objective, schedule)

    def do_crossover(self, i, j):
        code1, code2 = self.pop[0][i].ga_crossover_sequence_ox(self.pop[0][j])
        self.decode_update(i, code1)
        self.decode_update(j, code2)

    def do_mutation(self, i):
        code1 = self.pop[0][i].ga_mutation_sequence_operation_based_hybrid()
        self.decode_update(i, code1)

    def do_tabu_search(self, i):
        code1 = self.pop[0][i].ts_sequence_operation_based_hybrid(self.tabu_list[0][i], self.max_tabu)
        self.decode_update(i, code1)
        if len(self.tabu_list[0][i]) >= self.max_tabu:
            self.tabu_list[0][i] = []


class GaNwFrJsp(GaNwJsp):
    def __init__(self, pop_size, rc, rm, max_generation, objective, schedule):
        Ga.__init__(self, pop_size, rc, rm, max_generation, objective, schedule)

    def do_init(self, pop=None):
        self.record[0].append(time.perf_counter())
        for i in range(self.pop_size):
            if pop is None:
                code = self.schedule.sequence_permutation(self.schedule.n)
                route = self.schedule.route_job_based(self.schedule.n, self.p)
            else:
                code = pop[0][i].code
                route = pop[0][i].route
            info = self.schedule.decode_no_wait_active(code, self.p, route)
            self.pop[0].append(info)
            self.pop[1].append(self.objective(info))
            self.pop[2].append(Utils.calculate_fitness(self.pop[1][i]))
        self.init_best()
        self.record[1].append(time.perf_counter())
        self.show_generation(0)

    def do_crossover(self, i, j):
        code1, code2 = self.pop[0][i].ga_crossover_sequence_permutation_pmx(self.pop[0][j])
        route1, route2 = self.pop[0][i].ga_crossover_route_pmx(self.pop[0][j])
        self.decode_update(i, code1, route1)
        self.decode_update(j, code2, route2)

    def do_mutation(self, i):
        code1 = self.pop[0][i].ga_mutation_sequence_permutation_tpe()
        route1 = self.pop[0][i].ga_mutation_route_tpe()
        self.decode_update(i, code1, route1)

    def do_tabu_search(self, i):
        code1 = self.pop[0][i].ts_sequence_permutation(self.tabu_list[0][i], self.max_tabu)
        route1 = self.pop[0][i].ts_route_job_based(self.tabu_list[2][i], self.max_tabu)
        self.decode_update(i, code1, route1)
        for u in [0, 2]:
            if len(self.tabu_list[u][i]) >= self.max_tabu:
                self.tabu_list[u][i] = []


class GaLwFrJsp(GaFrJsp):
    def __init__(self, pop_size, rc, rm, max_generation, objective, schedule):
        GaFrJsp.__init__(self, pop_size, rc, rm, max_generation, objective, schedule)

    def do_init(self, pop=None):
        n, p = self.schedule.n, [job.nop for job in self.schedule.job.values()]
        self.record[0].append(time.perf_counter())
        for i in range(self.pop_size):
            if pop is None:
                code = self.schedule.sequence_operation_based(n, p)
                route = self.schedule.route_job_based(n, p)
            else:
                code = pop[0][i].code
                route = pop[0][i].route
            info = self.schedule.decode_operation_based_active(code, route)
            self.pop[0].append(info)
            self.pop[1].append(self.objective(info))
            self.pop[2].append(Utils.calculate_fitness(self.pop[1][i]))
        self.init_best()
        self.record[1].append(time.perf_counter())
        self.show_generation(0)


class GaFjsp(Ga):
    def __init__(self, pop_size, rc, rm, max_generation, objective, schedule):
        Ga.__init__(self, pop_size, rc, rm, max_generation, objective, schedule)

    def decode_update(self, i, code, mac, route=None):
        info = self.schedule.decode_operation_based_active(code, mac, route)
        self.update_individual(i, self.objective(info), info)

    def do_init(self, pop=None):
        self.record[0].append(time.perf_counter())
        for i in range(self.pop_size):
            if pop is None:
                code = self.schedule.sequence_operation_based(self.schedule.n, self.p)
                # mac = self.schedule.assignment_job_based(self.schedule.n, self.p, self.tech)
                # info = self.schedule.decode_operation_based_active(code, mac)
                info = self.schedule.decode_only_operation_based_active(code)
            else:
                code = pop[0][i].code
                mac = pop[0][i].mac
                info = self.schedule.decode_operation_based_active(code, mac)
            self.pop[0].append(info)
            self.pop[1].append(self.objective(info))
            self.pop[2].append(Utils.calculate_fitness(self.pop[1][i]))
        self.init_best()
        self.record[1].append(time.perf_counter())
        self.show_generation(0)

    def do_crossover(self, i, j):
        code1, code2 = self.pop[0][i].ga_crossover_sequence_hybrid(self.pop[0][j])
        mac1, mac2 = self.pop[0][i].ga_crossover_assignment_job_based_random(self.pop[0][j])
        self.decode_update(i, code1, mac1)
        self.decode_update(j, code2, mac2)

    def do_mutation(self, i):
        code1 = self.pop[0][i].ga_mutation_sequence_operation_based_hybrid()
        mac1 = self.pop[0][i].ga_mutation_assignment_job_based_random_replace(self.tech)
        self.decode_update(i, code1, mac1)

    def do_tabu_search(self, i):
        code1 = self.pop[0][i].ts_sequence_operation_based(self.tabu_list[0][i], self.max_tabu)
        mac1 = self.pop[0][i].ts_assignment_job_based(self.tabu_list[1][i], self.max_tabu)
        self.decode_update(i, code1, mac1)
        if len(self.tabu_list[0][i]) >= self.max_tabu:
            self.tabu_list[0][i] = []
        if len(self.tabu_list[1][i]) >= self.max_tabu:
            self.tabu_list[1][i] = []

    def do_key_block_move(self, i):
        block = self.pop[0][i].key_block()
        code1 = self.pop[0][i].method_sequence_key_block_move(block)
        mac1 = self.pop[0][i].method_sequence_key_block_move_mac(block)
        self.decode_update(i, code1, mac1)


class GaFrFjsp(GaFjsp):
    def __init__(self, pop_size, rc, rm, max_generation, objective, schedule):
        GaFjsp.__init__(self, pop_size, rc, rm, max_generation, objective, schedule)

    def do_init(self, pop=None):
        self.record[0].append(time.perf_counter())
        for i in range(self.pop_size):
            if pop is None:
                code = self.schedule.sequence_operation_based(self.schedule.n, self.p)
                # mac = self.schedule.assignment_job_based(self.schedule.n, self.p, self.tech)
                route = self.schedule.route_job_based(self.schedule.n, self.p)
                # info = self.schedule.decode_operation_based_active(code, mac, route)
                info = self.schedule.decode_only_operation_based_active(code, route)
            else:
                code = pop[0][i].code
                mac = pop[0][i].mac
                route = pop[0][i].route
                info = self.schedule.decode_operation_based_active(code, mac, route)
            self.pop[0].append(info)
            self.pop[1].append(self.objective(info))
            self.pop[2].append(Utils.calculate_fitness(self.pop[1][i]))
        self.init_best()
        self.record[1].append(time.perf_counter())
        self.show_generation(0)

    def do_crossover(self, i, j):
        code1, code2 = self.pop[0][i].ga_crossover_sequence_hybrid(self.pop[0][j])
        mac1, mac2 = self.pop[0][i].ga_crossover_assignment_job_based_random(self.pop[0][j])
        route1, route2 = self.pop[0][i].ga_crossover_route_pmx(self.pop[0][j])
        self.decode_update(i, code1, mac1, route1)
        self.decode_update(j, code2, mac2, route2)

    def do_mutation(self, i):
        code1 = self.pop[0][i].ga_mutation_sequence_operation_based_hybrid()
        mac1 = self.pop[0][i].ga_mutation_assignment_job_based_random_replace(self.tech)
        route1 = self.pop[0][i].ga_mutation_route_tpe()
        self.decode_update(i, code1, mac1, route1)

    def do_tabu_search(self, i):
        code1 = self.pop[0][i].ts_sequence_operation_based(self.tabu_list[0][i], self.max_tabu)
        mac1 = self.pop[0][i].ts_assignment_job_based(self.tabu_list[1][i], self.max_tabu)
        route1 = self.pop[0][i].ts_route_job_based(self.tabu_list[2][i], self.max_tabu)
        self.decode_update(i, code1, mac1, route1)
        for u in range(3):
            if len(self.tabu_list[u][i]) >= self.max_tabu:
                self.tabu_list[u][i] = []


class GaNwFjsp(Ga):
    def __init__(self, pop_size, rc, rm, max_generation, objective, schedule):
        Ga.__init__(self, pop_size, rc, rm, max_generation, objective, schedule)

    def decode_update(self, i, code, mac, route=None):
        info = self.schedule.decode_no_wait_active(code, mac, self.p, route)
        self.update_individual(i, self.objective(info), info)

    def do_init(self, pop=None):
        self.record[0].append(time.perf_counter())
        for i in range(self.pop_size):
            if pop is None:
                code = self.schedule.sequence_permutation(self.schedule.n)
                mac = self.schedule.assignment_job_based(self.schedule.n, self.p, self.tech)
            else:
                code = pop[0][i].code
                mac = pop[0][i].mac
            info = self.schedule.decode_no_wait_active(code, mac, self.p)
            self.pop[0].append(info)
            self.pop[1].append(self.objective(info))
            self.pop[2].append(Utils.calculate_fitness(self.pop[1][i]))
        self.init_best()
        self.record[1].append(time.perf_counter())
        self.show_generation(0)

    def do_crossover(self, i, j):
        code1, code2 = self.pop[0][i].ga_crossover_sequence_permutation_ox(self.pop[0][j])
        mac1, mac2 = self.pop[0][i].ga_crossover_assignment_job_based_random(self.pop[0][j])
        self.decode_update(i, code1, mac1)
        self.decode_update(j, code2, mac2)

    def do_mutation(self, i):
        code1 = self.pop[0][i].ga_mutation_sequence_permutation_tpe()
        mac1 = self.pop[0][i].ga_mutation_assignment_job_based_random_replace(self.tech)
        self.decode_update(i, code1, mac1)

    def do_tabu_search(self, i):
        code1 = self.pop[0][i].ts_sequence_permutation(self.tabu_list[0][i], self.max_tabu)
        mac1 = self.pop[0][i].ts_assignment_job_based(self.tabu_list[1][i], self.max_tabu)
        self.decode_update(i, code1, mac1)
        if len(self.tabu_list[0][i]) >= self.max_tabu:
            self.tabu_list[0][i] = []
        if len(self.tabu_list[1][i]) >= self.max_tabu:
            self.tabu_list[1][i] = []


class GaLwFjsp(GaFjsp):
    def __init__(self, pop_size, rc, rm, max_generation, objective, schedule):
        GaFjsp.__init__(self, pop_size, rc, rm, max_generation, objective, schedule)

    def do_crossover(self, i, j):
        code1, code2 = self.pop[0][i].ga_crossover_sequence_ox(self.pop[0][j])
        mac1, mac2 = self.pop[0][i].ga_crossover_assignment_job_based_random(self.pop[0][j])
        self.decode_update(i, code1, mac1)
        self.decode_update(j, code2, mac2)


class GaNwFrFjsp(GaNwFjsp):
    def __init__(self, pop_size, rc, rm, max_generation, objective, schedule):
        GaNwFjsp.__init__(self, pop_size, rc, rm, max_generation, objective, schedule)

    def do_init(self, pop=None):
        self.record[0].append(time.perf_counter())
        for i in range(self.pop_size):
            if pop is None:
                code = self.schedule.sequence_permutation(self.schedule.n)
                mac = self.schedule.assignment_job_based(self.schedule.n, self.p, self.tech)
                route = self.schedule.route_job_based(self.schedule.n, self.p)
            else:
                code = pop[0][i].code
                mac = pop[0][i].mac
                route = pop[0][i].route
            info = self.schedule.decode_no_wait_active(code, mac, self.p, route)
            self.pop[0].append(info)
            self.pop[1].append(self.objective(info))
            self.pop[2].append(Utils.calculate_fitness(self.pop[1][i]))
        self.init_best()
        self.record[1].append(time.perf_counter())
        self.show_generation(0)

    def do_crossover(self, i, j):
        code1, code2 = self.pop[0][i].ga_crossover_sequence_permutation_ox(self.pop[0][j])
        mac1, mac2 = self.pop[0][i].ga_crossover_assignment_job_based_random(self.pop[0][j])
        route1, route2 = self.pop[0][i].ga_crossover_route_pmx(self.pop[0][j])
        self.decode_update(i, code1, mac1, route1)
        self.decode_update(j, code2, mac2, route2)

    def do_mutation(self, i):
        code1 = self.pop[0][i].ga_mutation_sequence_permutation_tpe()
        mac1 = self.pop[0][i].ga_mutation_assignment_job_based_random_replace(self.tech)
        route1 = self.pop[0][i].ga_mutation_route_tpe()
        self.decode_update(i, code1, mac1, route1)

    def do_tabu_search(self, i):
        code1 = self.pop[0][i].ts_sequence_permutation(self.tabu_list[0][i], self.max_tabu)
        mac1 = self.pop[0][i].ts_assignment_job_based(self.tabu_list[1][i], self.max_tabu)
        route1 = self.pop[0][i].ts_route_job_based(self.tabu_list[2][i], self.max_tabu)
        self.decode_update(i, code1, mac1, route1)
        for u in range(3):
            if len(self.tabu_list[u][i]) >= self.max_tabu:
                self.tabu_list[u][i] = []


class GaLwFrFjsp(GaFrFjsp):
    def __init__(self, pop_size, rc, rm, max_generation, objective, schedule):
        GaFjsp.__init__(self, pop_size, rc, rm, max_generation, objective, schedule)

    def do_crossover(self, i, j):
        code1, code2 = self.pop[0][i].ga_crossover_sequence_ox(self.pop[0][j])
        mac1, mac2 = self.pop[0][i].ga_crossover_assignment_job_based_random(self.pop[0][j])
        route1, route2 = self.pop[0][i].ga_crossover_route_pmx(self.pop[0][j])
        self.decode_update(i, code1, mac1, route1)
        self.decode_update(j, code2, mac2, route2)


class GaFjsp1(Ga):
    def __init__(self, pop_size, rc, rm, max_generation, objective, schedule):
        Ga.__init__(self, pop_size, rc, rm, max_generation, objective, schedule)

    def decode_update(self, i, code, route=None):
        info = self.schedule.decode_only_operation_based_active(code, route)
        self.update_individual(i, self.objective(info), info)

    def do_init(self, pop=None):
        self.record[0].append(time.perf_counter())
        for i in range(self.pop_size):
            if pop is None:
                code = self.schedule.sequence_operation_based(self.schedule.n, self.p)
            else:
                code = pop[0][i].code
            info = self.schedule.decode_only_operation_based_active(code)
            self.pop[0].append(info)
            self.pop[1].append(self.objective(info))
            self.pop[2].append(Utils.calculate_fitness(self.pop[1][i]))
        self.init_best()
        self.record[1].append(time.perf_counter())
        self.show_generation(0)

    def do_crossover(self, i, j):
        code1, code2 = self.pop[0][i].ga_crossover_sequence_dpox(self.pop[0][j])
        self.decode_update(i, code1)
        self.decode_update(j, code2)

    def do_mutation(self, i):
        code1 = self.pop[0][i].ga_mutation_sequence_operation_based_tpe()
        self.decode_update(i, code1)

    def do_tabu_search(self, i):
        code1 = self.pop[0][i].ts_sequence_operation_based(self.tabu_list[0][i], self.max_tabu)
        self.decode_update(i, code1)
        if len(self.tabu_list[0][i]) >= self.max_tabu:
            self.tabu_list[0][i] = []

    def do_key_block_move(self, i):
        code1 = self.pop[0][i].method_sequence_key_block_move()
        self.decode_update(i, code1)


class GaFrFjsp1(GaFjsp1):
    def __init__(self, pop_size, rc, rm, max_generation, objective, schedule):
        GaFjsp1.__init__(self, pop_size, rc, rm, max_generation, objective, schedule)

    def do_init(self, pop=None):
        self.record[0].append(time.perf_counter())
        for i in range(self.pop_size):
            if pop is None:
                code = self.schedule.sequence_operation_based(self.schedule.n, self.p)
                route = self.schedule.route_job_based(self.schedule.n, self.p)
            else:
                code = pop[0][i].code
                route = pop[0][i].route
            info = self.schedule.decode_only_operation_based_active(code, route)
            self.pop[0].append(info)
            self.pop[1].append(self.objective(info))
            self.pop[2].append(Utils.calculate_fitness(self.pop[1][i]))
        self.init_best()
        self.record[1].append(time.perf_counter())
        self.show_generation(0)

    def do_crossover(self, i, j):
        code1, code2 = self.pop[0][i].ga_crossover_sequence_dpox(self.pop[0][j])
        route1, route2 = self.pop[0][i].ga_crossover_route_pmx(self.pop[0][j])
        self.decode_update(i, code1, route1)
        self.decode_update(j, code2, route2)

    def do_mutation(self, i):
        code1 = self.pop[0][i].ga_mutation_sequence_operation_based_tpe()
        route1 = self.pop[0][i].ga_mutation_route_tpe()
        self.decode_update(i, code1, route1)

    def do_tabu_search(self, i):
        code1 = self.pop[0][i].ts_sequence_operation_based(self.tabu_list[0][i], self.max_tabu)
        route1 = self.pop[0][i].ts_route_job_based(self.tabu_list[2][i], self.max_tabu)
        self.decode_update(i, code1, route1)
        for u in [0, 2]:
            if len(self.tabu_list[u][i]) >= self.max_tabu:
                self.tabu_list[u][i] = []


class GaNwFjsp1(Ga):
    def __init__(self, pop_size, rc, rm, max_generation, objective, schedule):
        Ga.__init__(self, pop_size, rc, rm, max_generation, objective, schedule)

    def decode_update(self, i, code, route=None):
        info = self.schedule.decode_no_wait_only_job_active(code, self.p, route)
        self.update_individual(i, self.objective(info), info)

    def do_init(self, pop=None):
        self.record[0].append(time.perf_counter())
        for i in range(self.pop_size):
            if pop is None:
                code = self.schedule.sequence_permutation(self.schedule.n)
            else:
                code = pop[0][i].code
            info = self.schedule.decode_no_wait_only_job_active(code, self.p)
            self.pop[0].append(info)
            self.pop[1].append(self.objective(info))
            self.pop[2].append(Utils.calculate_fitness(self.pop[1][i]))
        self.init_best()
        self.record[1].append(time.perf_counter())
        self.show_generation(0)

    def do_crossover(self, i, j):
        code1, code2 = self.pop[0][i].ga_crossover_sequence_pmx(self.pop[0][j])
        self.decode_update(i, code1)
        self.decode_update(j, code2)

    def do_mutation(self, i):
        code1 = self.pop[0][i].ga_mutation_sequence_permutation_tpe()
        self.decode_update(i, code1)


class GaLwFjsp1(GaFjsp1):
    def __init__(self, pop_size, rc, rm, max_generation, objective, schedule):
        GaFjsp1.__init__(self, pop_size, rc, rm, max_generation, objective, schedule)

    def do_crossover(self, i, j):
        code1, code2 = self.pop[0][i].ga_crossover_sequence_ox(self.pop[0][j])
        self.decode_update(i, code1)
        self.decode_update(j, code2)


class GaNwFrFjsp1(GaNwFjsp1):
    def __init__(self, pop_size, rc, rm, max_generation, objective, schedule):
        GaNwFjsp1.__init__(self, pop_size, rc, rm, max_generation, objective, schedule)

    def do_init(self, pop=None):
        self.record[0].append(time.perf_counter())
        for i in range(self.pop_size):
            if pop is None:
                code = self.schedule.sequence_permutation(self.schedule.n)
                route = self.schedule.route_job_based(self.schedule.n, self.p)
            else:
                code = pop[0][i].code
                route = pop[0][i].route
            info = self.schedule.decode_no_wait_only_job_active(code, self.p, route)
            self.pop[0].append(info)
            self.pop[1].append(self.objective(info))
            self.pop[2].append(Utils.calculate_fitness(self.pop[1][i]))
        self.init_best()
        self.record[1].append(time.perf_counter())
        self.show_generation(0)

    def do_crossover(self, i, j):
        code1, code2 = self.pop[0][i].ga_crossover_sequence_permutation_ox(self.pop[0][j])
        route1, route2 = self.pop[0][i].ga_crossover_route_pmx(self.pop[0][j])
        self.decode_update(i, code1, route1)
        self.decode_update(j, code2, route2)

    def do_mutation(self, i):
        code1 = self.pop[0][i].ga_mutation_sequence_permutation_tpe()
        route1 = self.pop[0][i].ga_mutation_route_tpe()
        self.decode_update(i, code1, route1)

    def do_tabu_search(self, i):
        code1 = self.pop[0][i].ts_sequence_permutation(self.tabu_list[0][i], self.max_tabu)
        route1 = self.pop[0][i].ts_route_job_based(self.tabu_list[2][i], self.max_tabu)
        self.decode_update(i, code1, route1)
        for u in [0, 2]:
            if len(self.tabu_list[u][i]) >= self.max_tabu:
                self.tabu_list[u][i] = []


class GaLwFrFjsp1(GaFrFjsp1):
    def __init__(self, pop_size, rc, rm, max_generation, objective, schedule):
        GaFrFjsp1.__init__(self, pop_size, rc, rm, max_generation, objective, schedule)

    def do_crossover(self, i, j):
        code1, code2 = self.pop[0][i].ga_crossover_sequence_ox(self.pop[0][j])
        route1, route2 = self.pop[0][i].ga_crossover_route_pmx(self.pop[0][j])
        self.decode_update(i, code1, route1)
        self.decode_update(j, code2, route2)


class GaFspHfsp(Ga):
    def __init__(self, pop_size, rc, rm, max_generation, objective, schedule):
        Ga.__init__(self, pop_size, rc, rm, max_generation, objective, schedule)

    def decode_update(self, i, code):
        info = self.schedule.decode_permutation(code)
        self.update_individual(i, self.objective(info), info)

    def do_init(self, pop=None):
        self.record[0].append(time.perf_counter())
        for i in range(self.pop_size):
            if pop is None:
                code = self.schedule.sequence_permutation(self.schedule.n)
            else:
                code = pop[0][i].code
            info = self.schedule.decode_permutation(code)
            self.pop[0].append(info)
            self.pop[1].append(self.objective(info))
            self.pop[2].append(Utils.calculate_fitness(self.pop[1][i]))
        self.init_best()
        self.record[1].append(time.perf_counter())
        self.show_generation(0)

    def do_crossover(self, i, j):
        code1, code2 = self.pop[0][i].ga_crossover_sequence_permutation_hybrid(self.pop[0][j])
        self.decode_update(i, code1)
        self.decode_update(j, code2)

    def do_mutation(self, i):
        code1 = self.pop[0][i].ga_mutation_sequence_permutation_hybrid()
        self.decode_update(i, code1)

    def do_tabu_search(self, i):
        code1 = self.pop[0][i].ts_sequence_permutation(self.tabu_list[0][i], self.max_tabu)
        self.decode_update(i, code1)
        if len(self.tabu_list[0][i]) >= self.max_tabu:
            self.tabu_list[0][i] = []


class GaFspHfspTimetable(GaFspHfsp):
    def __init__(self, pop_size, rc, rm, max_generation, objective, schedule):
        GaFspHfsp.__init__(self, pop_size, rc, rm, max_generation, objective, schedule)

    def decode_update(self, i, code):
        info = self.schedule.decode_permutation_timetable(code)
        self.update_individual(i, self.objective(info), info)

    def do_init(self, pop=None):
        self.record[0].append(time.perf_counter())
        for i in range(self.pop_size):
            if pop is None:
                code = self.schedule.sequence_permutation(self.schedule.n)
            else:
                code = pop[0][i].code
            info = self.schedule.decode_permutation_timetable(code)
            self.pop[0].append(info)
            self.pop[1].append(self.objective(info))
            self.pop[2].append(Utils.calculate_fitness(self.pop[1][i]))
        self.init_best()
        self.record[1].append(time.perf_counter())
        self.show_generation(0)
