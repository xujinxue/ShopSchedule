import numpy as np

from src import *


def main_jsp(instance, max_solve_time, index_limited_wait=0, index_due_date=0, n_exp=10, index=-1):
    # 第1步, 准备数据
    string = jsp_benchmark.instance[instance]
    if index < 0:
        string_limited_wait = Utils.load_text("./src/data/limited_wait_jsp/%s.txt" % instance)
    else:
        string_limited_wait = Utils.load_text("./src/data/limited_wait_jsp_c%s/%s.txt" % (index, instance))
    n, m, p, tech, proc = Utils.string2data_jsp_fsp(string)  # 工件数量, 机器数量, 工序数量, 加工机器, 加工时间
    all_machines = range(m)  # 机器列表
    jobs_data = []  # 加工数据jobs_data
    for i in range(n):
        jobs_data.append([])
        for j in range(p[i]):
            jobs_data[i].append((tech[i][j], int(proc[i][j])))  # 添加加工机器, 加工时间数据
    no_wait = []  # 无等待数据, -1表示无穷大或不存在
    for i in range(n):
        no_wait.append([])
        for j in range(p[i] - 1):
            no_wait[i].append(0)
        no_wait[i].append(-1)
    # 等待时间有限数据
    limited_wait = [None, no_wait, Utils.string2data_wait(string_limited_wait, p)][index_limited_wait]
    # 交货期数据
    try:
        due_date = [None, list(map(int, jsp_benchmark.due_date[instance].split()))][index_due_date]
    except KeyError:
        due_date = None
    # print(jobs_data, "# jobs_data")
    # print(limited_wait, "# limited_wait")
    # print(due_date, "# due_date")
    log = ["C", "NWJSP", "LWJSP"][index_limited_wait]
    if index_due_date == 1:
        log += "-DD"
    if index < 0:
        log_dir = "./ORToolsResult_%s/%s" % (log, instance)
    else:
        log_dir = "./ORToolsResult_%s/%s_%s" % (log, instance, index)
    Utils.make_dir("./ORToolsResult_%s" % log)
    Utils.make_dir(log_dir)
    record = [[], [], []]
    # 第2~8步：见~/src/algorithm/ot.py
    solver = OrToolsJspSat(instance, all_machines, jobs_data, max_solve_time, log_dir,
                           limited_wait=limited_wait, due_date=due_date)
    for cur_exp in range(n_exp):
        Utils.print("{}".format("=" * 100), fore=Utils.fore().LIGHTYELLOW_EX)
        Utils.print("Solving %s begin ..." % (cur_exp + 1), fore=Utils.fore().LIGHTYELLOW_EX)
        res = solver(cur_exp)
        [record[i].append(res[i]) for i in range(3)]
        Utils.print("Solving %s  done ..." % (cur_exp + 1), fore=Utils.fore().LIGHTRED_EX)
        Utils.print("Status: %s, Objective: %s, Runtime: %.4f" % (res[2], res[1], res[0]))
        Utils.print("{}".format("=" * 100), fore=Utils.fore().LIGHTRED_EX)
    Utils.print("Average runtime: %.4f" % np.mean(record[0]))
    Utils.print("Average objective value: %.4f" % np.mean(record[1]))
    Utils.print("Status#4: {}\n".format(record[2].count(4)))
    if index < 0:
        file_dir = "./ORToolsResult_%s/%s.csv" % (log, instance)
    else:
        file_dir = "./ORToolsResult_%s/%s_%s.csv" % (log, instance, index)
    with open(file_dir, "w") as f:
        f.writelines("Objective,Status,Runtime\n")
        for i, j, k in zip(record[1], record[2], record[0]):
            f.writelines("{},{},{}\n".format(i, j, k))
        f.writelines("Min obj: {}\n".format(np.min(record[1])))
        f.writelines("Max obj: {}\n".format(np.max(record[1])))
        f.writelines("Mean obj: {}\n".format(np.mean(record[1])))
        f.writelines("STD: {}\n".format(np.std(record[1])))
        f.writelines("Status#4: {} ({:.2f}%)\n".format(record[2].count(4), 100 * record[2].count(4) / n_exp))
        f.writelines("Mean Runtime: {}\n".format(np.mean(record[0])))


def run():
    index_limited_wait = 2
    index_due_date = 0
    max_solve_time = 3600
    for instance in INSTANCE_LIST_JSP.split():
        main_jsp(instance, max_solve_time, index_limited_wait, index_due_date)


def run2():
    index_limited_wait = 2
    index_due_date = 0
    max_solve_time = 3600
    for instance in "la06 la07 la08".split():
        Utils.print(instance)
        cof = [0.5, 1, 2, 10]
        for index, c in enumerate(cof):
            Utils.print("%s %s" % (index, c))
            main_jsp(instance, max_solve_time, index_limited_wait, index_due_date, index=index)


if __name__ == '__main__':
    # run()
    run2()
