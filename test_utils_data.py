from src import *

LOW, HIGH = 10, 100


def data_mrjsp():
    Utils.make_dir("./src/data/mrjsp")
    n = 10  # 工件数量
    m = 10  # 机器数量
    r_min, r_max = 2, 4  # 一个工件的最少加工路径数量、最多加工路径数量
    p_min, p_max = 3, 5  # 一条加工路径的最少工序数量、最多工序数量
    # n = 3  # 工件数量
    # m = 4  # 机器数量
    # r = [2, 3, 2]  # 工件的加工路径数量
    # p = [[2, 3], [2, 3, 3], [3, 3]]  # 工件加工路径的工序数量
    for instance in range(1, 6):
        r = Utils.route_list(n, r_min, r_max)
        p = Utils.p_list(n, r, p_min, p_max)
        save = "./src/data/mrjsp/n%sm%s-%s.txt" % (n, m, instance)
        Utils.data_mrjsp(save, n, m, r, p, LOW, HIGH)
        Utils.print("Create %s" % save)


def data_mrfjsp():
    Utils.make_dir("./src/data/mrfjsp")
    n = 10  # 工件数量
    m = 10  # 机器数量
    r_min, r_max = 2, 4  # 一个工件的最少加工路径数量、最多加工路径数量
    p_min, p_max = 3, 5  # 一条加工路径的最少工序数量、最多工序数量
    q_min, q_max = 2, 4  # 一道工序的最少加工机器数量、最多加工机器数量
    # n = 3  # 工件数量
    # m = 4  # 机器数量
    # r = [2, 3, 2]  # 工件的加工路径数量
    # p = [[2, 3], [2, 3, 3], [3, 3]]  # 工件加工路径的工序数量
    # # 工件加工路径工序的机器数量
    # q = [[[2, 2], [2, 3, 2]],
    #      [[2, 2], [2, 3, 3], [3, 2, 3]],
    #      [[2, 2, 2], [2, 3, 2]]]
    for instance in range(1, 6):
        r = Utils.route_list(n, r_min, r_max)
        p = Utils.p_list(n, r, p_min, p_max)
        q = Utils.q_list(n, r, p, q_min, q_max)
        save = "./src/data/mrfjsp/n%sm%s-%s.txt" % (n, m, instance)
        Utils.data_mrfjsp(save, n, m, r, p, q, LOW, HIGH)
        Utils.print("Create %s" % save, fore=Utils.fore().LIGHTMAGENTA_EX)


if __name__ == '__main__':
    data_mrjsp()
    data_mrfjsp()
