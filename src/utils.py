__doc__ = """
工具包
"""

import datetime
import os

import chardet
import numpy as np
from colorama import init, Fore

dt = datetime.datetime
init(autoreset=True)


class Utils:
    @staticmethod
    def create_schedule(shop, n, m, p, tech, proc, limited_wait=None, rest_start_end=None,
                        resumable=None, due_date=None, best_known=None, time_unit=1):  # 创建一个车间调度对象
        schedule = shop()  # shop是车间类, 在shop包里面, 如Jsp, Fjsp, Fsp, Hfsp
        schedule.best_known = best_known  # 已知最优目标值
        schedule.time_unit = time_unit  # 加工时间单位
        for i in range(m):  # 添加机器, 方法add_machine定义在resource包的schedule模块的Schedule类里面
            if rest_start_end is not None:  # rest_start_end[0][i]是机器i的停工开始时刻, rest_start_end[1][i]是对应的开工开始时刻
                schedule.add_machine(name=i, timetable={0: rest_start_end[0][i], 1: rest_start_end[1][i]})
            else:
                schedule.add_machine(name=i)
        for i in range(n):  # 添加工件, 方法add_job也定义在resource包的schedule模块的Schedule类里面
            try:
                val_due_date = time_unit * due_date[i]  # 工件的交货期数据, due_date是一个包含n个元素的列表, 对应n个工件的交货期
            except TypeError:
                val_due_date = None
            schedule.add_job(due_date=val_due_date, name=i)
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
                schedule.job[i].add_task(tech[i][j], proc[i][j], name=j, limited_wait=val_limited_wait,
                                         resumable=val_resumable)  # 方法add_task定义在resource包的job模块的Job类里面
        schedule.rule_init_task_on_machine(m)
        return schedule

    @staticmethod
    def direction():  # 正向时间表或反向时间表
        return 0 if np.random.random() < 0.5 else 1

    @staticmethod
    def calculate_fitness(obj):  # 适应度函数
        return 1 / (1 + obj)

    @staticmethod
    def update_info(old_obj, new_obj):  # 更新个体的条件
        return True if new_obj < old_obj else False

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
    def save_code_to_txt(file, data):
        if not file.endswith(".txt"):
            file = file + ".txt"
        with open(file, "w", encoding="utf-8") as f:
            f.writelines(str(data))

    @staticmethod
    def save_obj_to_csv(file, data):
        if not file.endswith(".csv"):
            file = file + ".csv"
        with open(file, "w", encoding="utf-8") as f:
            obj, n_iter, direction = [], [], []
            f.writelines("{},{},{},{}\n".format("Test", "Objective", "Iteration", "Direction"))
            for k, v in enumerate(data):
                f.writelines("{},{},{},{}\n".format(k + 1, v[0], v[1] - 1, v[2]))
                obj.append(v[0])
                n_iter.append(v[1] - 1)
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
