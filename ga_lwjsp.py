__doc__ = """
基于遗传算法求解等待时间有限的作业车间调度问题；

"""

from src import *


def run(instance="ft06", y=-1):
    a = jsp_benchmark.instance[instance]
    n, m, p, tech, proc = Utils.string2data_jsp_fsp(a)
    if y < 0:
        b = Utils.load_text("./src/data/limited_wait_jsp/%s.txt" % instance)
        lw = Utils.string2data_wait(b, p)
        instance = instance
        best_known = jsp_benchmark.best_known_limited_wait[instance]
    else:
        best_known = jsp_benchmark.best_known_time_lag[str(y)][instance]
        lw = Utils.crt_limited_wait_cof(p, proc, y, float)
        instance = "%s_0_%s" % (instance, y)
        # best_known = None
    ga = GaTemplateJsp(pop_size=100, rc=0.85, rm=0.15, max_generation=5000, objective=Objective.makespan,
                       n=n, m=m, p=p, tech=tech, proc=proc, limited_wait=lw,
                       index_template=3, best_known=best_known)
    ga.do_exp(exp_log="./HGA_JSPTL", instance=instance, n_exp=10, tabu_search=True, key_block_move=False,
              gc=False, y_based=1)


def main():
    for instance in INSTANCE_LIST_JSP.split():
        run(instance=instance)


def main2():
    y_set = [0.5, 1, 2]  # la01 la02 la03 la04 la05
    # y_set = [0.5, 1, 3, 10]  # la06 la07 la08
    for instance in "la01 la02 la03 la04 la05".split():
        for y in y_set:
            Utils.print("%s_0_%s" % (instance, y), fore=Utils.fore().LIGHTRED_EX)
            run(instance, y=y)


if __name__ == '__main__':
    # main()
    main2()
