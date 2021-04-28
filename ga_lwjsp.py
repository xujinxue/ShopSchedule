__doc__ = """
基于遗传算法求解等待时间有限的作业车间调度问题；

"""

from src import *


def run(instance="ft06", index=-1):
    a = jsp_benchmark.instance[instance]
    if index < 0:
        b = Utils.load_text("./src/data/limited_wait_jsp/%s.txt" % instance)
    else:
        b = Utils.load_text("./src/data/limited_wait_jsp_c%s/%s.txt" % (index, instance))
    n, m, p, tech, proc = Utils.string2data_jsp_fsp(a)
    lw = Utils.string2data_wait(b, p)
    best_known = jsp_benchmark.best_known_limited_wait[instance]
    ga = GaTemplateJsp(pop_size=50, rc=0.85, rm=0.15, max_generation=500, objective=Objective.makespan,
                       n=n, m=m, p=p, tech=tech, proc=proc, limited_wait=lw,
                       index_template=3, best_known=best_known)
    ga.do_exp(exp_log="./GA_LWJSP", instance=instance, n_exp=10, tabu_search=True, key_block_move=False,
              gc=False, y_based=1)


def main():
    for instance in INSTANCE_LIST_JSP.split():
        run(instance=instance)


def main2():
    for instance in "la06 la07 la08".split():
        Utils.print(instance)
        cof = [0.5, 1, 2, 10]
        for index, c in enumerate(cof):
            Utils.print("%s %s" % (index, c))
            run(instance, index=index)


if __name__ == '__main__':
    # main()
    main2()
