__doc__ = """
工具包
"""

import datetime
import os

import chardet
import numpy as np
from colorama import init, Fore

from .objective import Objective

dt = datetime.datetime
init(autoreset=True)


class Utils:
    @staticmethod
    def create_schedule(shop, n, m, p, tech, proc, multi_route=False, limited_wait=None, rest_start_end=None,
                        resumable=None, w=None, worker=None, due_date=None, best_known=None, time_unit=1):  # 创建一个车间调度对象
        schedule = shop()  # shop是车间类, 在shop包里面, 如Jsp, Fjsp, Fsp, Hfsp
        schedule.best_known = best_known  # 已知最优目标值
        schedule.time_unit = time_unit  # 加工时间单位
        for i in range(m):  # 添加机器, 方法add_machine定义在resource包的schedule模块的Schedule类里面
            if rest_start_end is not None:  # rest_start_end[0][i]是机器i的停工开始时刻, rest_start_end[1][i]是对应的开工开始时刻
                schedule.add_machine(name=i, timetable={0: rest_start_end[0][i], 1: rest_start_end[1][i]})
            else:
                schedule.add_machine(name=i)
        if w is not None:
            for i in range(w):
                schedule.add_worker(name=i)
        for i in range(n):  # 添加工件, 方法add_job也定义在resource包的schedule模块的Schedule类里面
            try:
                val_due_date = time_unit * due_date[i]  # 工件的交货期数据, due_date是一个包含n个元素的列表, 对应n个工件的交货期
            except TypeError:
                val_due_date = None
            schedule.add_job(due_date=val_due_date, name=i)
            if multi_route is False:
                for j in range(p[i]):  # 添加工序, p是一个包含n个元素的列表, 对应n个工件的工序数量
                    val_limited_wait = None
                    if limited_wait is not None:  # 等待时间有限数据
                        val_limited_wait = limited_wait[i][j]
                        if val_limited_wait == -1:  # 允许等待时间为无穷大或不存在允许等待时间
                            val_limited_wait = np.inf
                    try:  # 加工是否可恢复数据
                        val_resumable = resumable[i][j]
                    except TypeError:
                        val_resumable = None
                    if worker is None:
                        val_wok = None
                    else:
                        val_wok = worker[i][j]
                    schedule.job[i].add_task(tech[i][j], proc[i][j], name=j, limited_wait=val_limited_wait,
                                             resumable=val_resumable, worker=val_wok)
            else:
                for r, (route, duration) in enumerate(zip(tech[i], proc[i])):
                    schedule.job[i].add_route(name=r)
                    for j in range(p[i]):
                        val_limited_wait = None
                        if limited_wait is not None:
                            val_limited_wait = limited_wait[r][i][j]
                            if val_limited_wait == -1:
                                val_limited_wait = np.inf
                        try:
                            val_resumable = resumable[r][i][j]
                        except TypeError:
                            val_resumable = None
                        if worker is None:
                            val_wok = None
                        else:
                            val_wok = worker[r][i][j]
                        schedule.job[i].route[r].add_task(route[j], duration[j], name=j,
                                                          limited_wait=val_limited_wait,
                                                          resumable=val_resumable, worker=val_wok)
        try:
            schedule.rule_init_task_on_machine(m)
        except AttributeError:
            pass
        return schedule

    @staticmethod
    def direction():  # 正向时间表或反向时间表；反向时间表对拖期、完工时间之和、流程时间之和等目标有影响
        return 1 if np.random.random() < 0.5 else 0

    @staticmethod
    def direction0none(objective):
        return None if objective in [Objective.makespan, ] else 0

    @staticmethod
    def direction0none_multi(objective_list):
        return 0 if Objective.tardiness in objective_list else None

    @staticmethod
    def calculate_fitness(obj):  # 适应度函数
        return 1 / (1 + obj)

    @staticmethod
    def update_info_accept_equal(old_obj, new_obj):  # 更新个体的条件
        return True if new_obj <= old_obj else False

    @staticmethod
    def update_info(old_obj, new_obj):  # 更新个体的条件
        return True if new_obj < old_obj else False

    @staticmethod
    def similarity(a, b):
        return 1 - np.count_nonzero(a - b) / a.shape[0]

    @staticmethod
    def len_tabu(m, n):  # 禁忌搜索表的长度
        a = m * n
        if a < 250:
            return 250
        elif a < 500:
            return 500
        return a

    @staticmethod
    def is_dominate(obj_a, obj_b, num_obj):  # 支配关系, 若个体a支配个体b, 则返回True
        if type(obj_a) is not np.ndarray:
            obj_a, obj_b = np.array(obj_a), np.array(obj_b)
        res = np.array([np.sign(k) for k in obj_a - obj_b])
        res_ngt0, res_eqf1 = np.argwhere(res <= 0), np.argwhere(res == -1)
        if res_ngt0.shape[0] == num_obj and res_eqf1.shape[0] > 0:
            return True
        return False

    @staticmethod
    def fore():
        return Fore

    @staticmethod
    def print(msg, fore=Fore.LIGHTCYAN_EX):
        print(fore + msg)

    @staticmethod
    def make_dir(*args, **kw):
        try:
            os.makedirs(*args, **kw)
        except FileExistsError:
            pass

    @staticmethod
    def clear_dir(dir_name):
        try:
            for i in os.listdir(dir_name):
                os.remove("%s/%s" % (dir_name, i))
        except IsADirectoryError:
            pass

    @staticmethod
    def make_dir_save(save, instance, stage2=None):
        Utils.make_dir("./%s" % save)
        Utils.make_dir("./%s/%s" % (save, instance))
        a = ["./%s/%s" % (save, instance), "./%s/%s/Code" % (save, instance), "./%s/%s/GanttChart" % (save, instance),
             "./%s/%s/GanttChartPngHtml" % (save, instance), "./%s/%s/Record" % (save, instance)]
        [Utils.make_dir(i) for i in a]
        try:
            [Utils.clear_dir(i) for i in a]
        except PermissionError:
            pass
        if stage2 is not None:
            a = ["./%s/%s/Code2" % (save, instance), "./%s/%s/GanttChart2" % (save, instance),
                 "./%s/%s/GanttChartPngHtml2" % (save, instance), "./%s/%s/Record2" % (save, instance)]
            [Utils.make_dir(i) for i in a]
            try:
                [Utils.clear_dir(i) for i in a]
            except PermissionError:
                pass

    @staticmethod
    def load_text(file_name):
        try:
            with open(file_name, "rb") as f:
                f_read = f.read()
                f_cha_info = chardet.detect(f_read)
                final_data = f_read.decode(f_cha_info['encoding'])
                return final_data
        except FileNotFoundError:
            return None

    @staticmethod
    def crt_resumable(n, p, val=None):
        a = []
        for i in range(n):
            b = []
            for j in range(p[i]):
                if val in [0, 1]:
                    b.append(val)
                else:
                    b.append(np.random.choice(val, 1, replace=False)[0])
            a.append(b)
        return a

    @staticmethod
    def crt_limited_wait(n, p, low, high, non):
        a = np.random.uniform(low, high, sum(p) - n).astype(int)
        b = np.random.choice(range(a.shape[0]), non, replace=False)
        a[b] = -1
        c = []
        d = 0
        for i in range(n):
            c.append([])
            for j in range(p[i] - 1):
                c[i].append(a[d])
                d += 1
            c[i].append(-1)
        return c

    @staticmethod
    def crt_limited_wait_cof(p, proc, y, dtype=int):
        a = []
        for i, n in enumerate(p):
            a.append([])
            for j in range(n - 1):
                a[-1].append(dtype(y * np.mean(proc[i])))
            a[i].append(-1)
        return a

    @staticmethod
    def string2data_jsp_fsp(string, dtype=int, time_unit=1):
        try:
            to_data = list(map(dtype, string.split()))
            n, m = int(to_data[0]), int(to_data[1])
            p = [m] * n
            tech = [[] for _ in range(n)]
            proc = [[] for _ in range(n)]
            job, index = 0, 2
            for i in range(n):
                for j in range(m):
                    tech[i].append(int(to_data[index]))
                    proc[i].append(time_unit * to_data[index + 1])
                    index += 2
            return n, m, p, tech, proc
        except ValueError:
            return None, None, None, None, None

    @staticmethod
    def string2data_fjsp_hfsp(string, dtype=int, time_unit=1, minus_one=True):
        try:
            to_data = list(map(dtype, string.split()))
            job, p, tech, prt = 0, [], [], []
            n, m = int(to_data[0]), int(to_data[1])
            index_no, index_nm, index_m, index_t = 2, 3, 4, 5
            while job < n:
                p.append(int(to_data[index_no]))
                tech.append([])
                prt.append([])
                for i in range(p[job]):
                    tech[job].append([])
                    prt[job].append([])
                    int_index_nm = int(to_data[index_nm])
                    for j in range(int_index_nm):
                        int_index_m = int(to_data[index_m])
                        if minus_one is True:
                            tech[job][i].append(int_index_m - 1)
                        else:
                            tech[job][i].append(int_index_m)
                        prt[job][i].append(time_unit * to_data[index_t])
                        index_m += 2
                        index_t += 2
                    index_nm = index_nm + 2 * int_index_nm + 1
                    index_m = index_nm + 1
                    index_t = index_nm + 2
                job += 1
                index_nm = index_nm + 1
                index_m = index_m + 1
                index_t = index_t + 1
                index_no = index_t - 3
            return n, m, p, tech, prt
        except ValueError:
            return None, None, None, None, None

    @staticmethod
    def string2data_wait(string, nop, dtype=int, time_unit=1):
        to_int = list(map(dtype, string.split()))
        wait = []
        value = []
        for i, j in enumerate(to_int):
            value.append(j)
            if i + 1 == sum(nop[:len(wait) + 1]):
                wait.append(time_unit * value)
                value = []
        return wait

    @staticmethod
    def string2data_mrjsp(string, dtype=int, time_unit=1, minus_one=False):
        try:
            a = string.split("\n")
            n, m = list(map(int, a[0].split()))
            r = [m] * n  # 虚设加工路线数量
            p = [m] * n  # 虚设工序
            tech = [[] for _ in range(n)]
            proc = [[] for _ in range(n)]
            job, row = 0, 0
            while job < n:
                row += 1
                b = list(map(int, a[row].split()))
                r[job], p[job] = b[0], max(b[1:])  # 实际加工路径数量，所有加工路径中的最大工序数量
                for i in range(r[job]):
                    row += 1
                    tech_proc_r = list(map(dtype, a[row].split()))
                    tech_r = [int(i) for i in tech_proc_r[::2]]
                    proc_r = [time_unit * i for i in tech_proc_r[1::2]]
                    if len(tech_r) < p[job]:  # 虚工序
                        dummy = np.random.choice(range(m), p[job] - len(tech_r), replace=False)
                        tech_r.extend(dummy.tolist())
                        proc_r.extend([0] * len(dummy))
                    if minus_one:
                        tech_r = [i - 1 for i in tech_r]
                    tech[job].append(tech_r)
                    proc[job].append(proc_r)
                job += 1
            return n, m, p, tech, proc
        except ValueError:
            return None, None, None, None, None

    @staticmethod
    def string2data_mrfjsp(string, dtype=int, time_unit=1, minus_one=True):
        try:
            a = string.split("\n")
            n, m = list(map(int, a[0].split()))
            r = [m] * n  # 虚设加工路线数量
            p = [m] * n  # 虚设工序
            tech = [[] for _ in range(n)]
            proc = [[] for _ in range(n)]
            job, row = 0, 0
            while job < n:
                row += 1
                b = list(map(int, a[row].split()))
                r[job], p[job] = b[0], max(b[1:])
                for i in range(r[job]):
                    row += 1
                    tech_proc_r = list(map(dtype, a[row].split()))
                    index_nm, index_m, index_p, = 0, 1, 2
                    tech_r = []
                    proc_r = []
                    for j in range(b[1:][i]):
                        tech_r_j = []
                        proc_r_j = []
                        for k in range(tech_proc_r[index_nm]):
                            tech_r_j.append(tech_proc_r[index_m])
                            proc_r_j.append(time_unit * tech_proc_r[index_p])
                            index_m, index_p = index_m + 2, index_p + 2
                        if minus_one:
                            tech_r_j = [i - 1 for i in tech_r_j]
                        tech_r.append(tech_r_j)
                        proc_r.append(proc_r_j)
                        index_nm = index_p - 1
                        index_m, index_p = index_nm + 1, index_nm + 2
                    if b[1:][i] < p[job]:  # 虚工序
                        dummy = np.random.choice(range(m), p[job] - b[1:][i], replace=False)
                        for w in dummy:
                            tech_r.append([w])
                            proc_r.append([0])
                    tech[job].append(tech_r)
                    proc[job].append(proc_r)
                job += 1
            return n, m, p, tech, proc
        except ValueError:
            return None, None, None, None, None

    @staticmethod
    def string2data_drcfjsp(string, dtype=int, time_unit=1, minus_one=True):
        try:
            a = string.split("\n")
            n, m, w = list(map(int, a[0].split()))
            p = list(map(int, a[1].split()))
            n_job = 0
            tech, worker, proc = [], [], []
            b = [[], [], []]
            try:
                for val in a[2:]:
                    c, d, e = [], [], []
                    data = list(map(dtype, val.split()))
                    index_m, index_mn, index_nw, index_wn, index_t = 0, 1, 2, 3, 4
                    n_machine = int(data[index_m])
                    for k in range(n_machine):
                        c.append(int(data[index_mn]))
                        n_worker = int(data[index_nw])
                        f, g = [], []
                        for cur in range(n_worker):
                            f.append(int(data[index_wn]))
                            g.append(time_unit * data[index_t])
                            index_wn += 2
                            index_t += 2
                        if minus_one is True:
                            f = [v - 1 for v in f]
                        d.append(f)
                        e.append(g)
                        index_mn = index_t - 1
                        index_nw, index_wn, index_t = index_mn + 1, index_mn + 2, index_mn + 3
                    if minus_one is True:
                        c = [v - 1 for v in c]
                    b[0].append(c)
                    b[1].append(d)
                    b[2].append(e)
                    if len(b[0]) == p[n_job]:
                        tech.append(b[0])
                        worker.append(b[1])
                        proc.append(b[2])
                        n_job += 1
                        b = [[], [], []]
            except IndexError:
                pass
            return n, m, w, p, tech, worker, proc
        except ValueError:
            return None, None, None, None, None, None, None

    @staticmethod
    def save_code_to_txt(file, data):
        if not file.endswith(".txt"):
            file = file + ".txt"
        with open(file, "w", encoding="utf-8") as f:
            for i, j in enumerate(str(data)):
                f.writelines(j)
                if (i + 1) % 100 == 0:
                    f.writelines("\n")
            f.writelines("\n")

    @staticmethod
    def save_obj_to_csv(file, data):
        if not file.endswith(".csv"):
            file = file + ".csv"
        with open(file, "w", encoding="utf-8") as f:
            obj, n_iter, direction = [], [], []
            f.writelines("{},{},{},{}\n".format("Test", "Objective", "IterationReachBest", "Direction"))
            for k, v in enumerate(data):
                f.writelines("{},{},{},{}\n".format(k + 1, v[0], v[1], v[2]))
                obj.append(v[0])
                n_iter.append(v[1])
                direction.append(v[2])
            f.writelines("{},{}\n".format("MinObj", min(obj)))
            f.writelines("{},{}\n".format("MaxObj", max(obj)))
            f.writelines("{},{:.2f}\n".format("MeanObj", sum(obj) / len(obj)))
            f.writelines("{},{}\n".format("MinIter", min(n_iter)))
            f.writelines("{},{}\n".format("MaxIter", max(n_iter)))
            f.writelines("{},{:.2f}\n".format("MeanIter", sum(n_iter) / len(n_iter)))
            try:
                f.writelines("{},{}\n".format("Direction#0", len(direction) - sum(direction)))
                f.writelines("{},{}\n".format("Direction#1", sum(direction)))
            except TypeError:
                pass

    @staticmethod
    def save_record_to_csv(file, data):
        if not file.endswith(".csv"):
            file = file + ".csv"
        n_row, n_column = len(data[0]), len(data)
        with open(file, "w", encoding="utf-8") as f:
            for i in range(n_row):
                a = ""
                for j in range(n_column):
                    a += "%s," % data[j][i]
                f.writelines(a[:-1] + "\n")

    @staticmethod
    def data_mrjsp(file, n, m, r, p, low, high, reenter=False, dtype=int):
        tech, proc = [], []
        for i in range(n):
            tech.append([])
            proc.append([])
            for j in range(r[i]):
                tech[i].append(np.random.choice(range(m), p[i][j], replace=reenter).tolist())
                proc[i].append(np.random.uniform(low, high, p[i][j]).astype(dtype).tolist())
        Utils.save_mrjsp(file, n, m, r, p, tech, proc)

    @staticmethod
    def save_mrjsp(file, n, m, r, p, tech, proc):
        with open(file, "w", encoding="utf-8") as f:
            f.writelines("%s %s\n" % (n, m))
            for i in range(n):
                a = "%s " % r[i]
                for j in range(r[i]):
                    a += "%s " % p[i][j]
                f.writelines("%s\n" % a)
                for j in range(r[i]):
                    b = ""
                    for u, v in zip(tech[i][j], proc[i][j]):
                        b += "%s %s " % (u, v)
                    f.writelines("%s\n" % b)

    @staticmethod
    def data_mrfjsp(file, n, m, r, p, q, low, high, reenter=False, dtype=int):
        tech, proc = [], []
        for i in range(n):
            tech.append([])
            proc.append([])
            for j in range(r[i]):
                tech[i].append([])
                proc[i].append([])
                for k in range(p[i][j]):
                    tech[i][j].append((np.random.choice(range(m), q[i][j][k], replace=reenter) + 1).tolist())
                    proc[i][j].append(np.random.uniform(low, high, q[i][j][k]).astype(dtype).tolist())
        Utils.save_mrfjsp(file, n, m, r, p, q, tech, proc)

    @staticmethod
    def save_mrfjsp(file, n, m, r, p, q, tech, proc):
        with open(file, "w", encoding="utf-8") as f:
            f.writelines("%s %s\n" % (n, m))
            for i in range(n):
                a = "%s " % r[i]
                for j in range(r[i]):
                    a += "%s " % p[i][j]
                f.writelines("%s\n" % a)
                for j in range(r[i]):
                    b = ""
                    for k in range(p[i][j]):
                        b += "%s " % q[i][j][k]
                        for u, v in zip(tech[i][j][k], proc[i][j][k]):
                            b += "%s %s " % (u, v)
                    f.writelines("%s\n" % b)

    @staticmethod
    def route_list(n, r_min, r_max):
        return np.random.randint(r_min, r_max + 1, n).tolist()

    @staticmethod
    def p_list(n, r, p_min, p_max):
        a = []
        for i in range(n):
            a.append(Utils.route_list(r[i], p_min, p_max))
        return a

    @staticmethod
    def q_list(n, r, p, q_min, q_max):
        a = []
        for i in range(n):
            a.append([])
            for j in range(r[i]):
                a[i].append(Utils.route_list(p[i][j], q_min, q_max))
        return a
