from src import *

LOW, HIGH = 10, 100


def data_mrjsp():
    n = 3  # 工件数量
    m = 4  # 机器数量
    r = [2, 3, 2]  # 工件的加工路径数量
    p = [[2, 3], [2, 3, 3], [3, 3]]  # 工件加工路径的工序数量
    for instance in range(1, 6):
        save = "./src/data/mrjsp/n%sm%s-%s.txt" % (n, m, instance)
        Utils.data_mrjsp(save, n, m, r, p, LOW, HIGH)
        Utils.print("Create %s" % save)


def data_mrfjsp():
    n = 3  # 工件数量
    m = 4  # 机器数量
    r = [2, 3, 2]  # 工件的加工路径数量
    p = [[2, 3], [2, 3, 3], [3, 3]]  # 工件加工路径的工序数量
    # 工件加工路径工序的机器数量
    q = [[[2, 2], [2, 3, 2]],
         [[2, 2], [2, 3, 3], [3, 2, 3]],
         [[2, 2, 2], [2, 3, 2]]]
    for instance in range(1, 6):
        save = "./src/data/mrfjsp/n%sm%s-%s.txt" % (n, m, instance)
        Utils.data_mrfjsp(save, n, m, r, p, q, LOW, HIGH)
        Utils.print("Create %s" % save, fore=Utils.fore().LIGHTMAGENTA_EX)


if __name__ == '__main__':
    Utils.make_dir("./src/data/mrjsp")
    Utils.make_dir("./src/data/mrfjsp")
    data_mrjsp()
    data_mrfjsp()
