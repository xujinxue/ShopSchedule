__doc__ = """
基于遗传算法求解柔性作业车间调度问题；

"""

from src import *


def run(instance="kacem1"):
    a = fjsp_benchmark.instance[instance]
    b = fjsp_benchmark.due_date[instance]
    n, m, p, tech, proc = Utils.string2data_fjsp_hfsp(a)
    due_date = list(map(int, b.split()))
    best_known = fjsp_benchmark.best_known[instance]
    ga = GaTemplateFjsp(pop_size=50, rc=0.85, rm=0.15, max_generation=500, objective=Objective.makespan,
                        n=n, m=m, p=p, tech=tech, proc=proc, due_date=due_date,
                        index_template=6, best_known=best_known)
    ga.do_exp(exp_log="./GA_FJSP", instance=instance, n_exp=10, tabu_search=True, key_block_move=False,
              gc=False, y_based=1)


def main():
    for instance in INSTANCE_LIST_FJSP.split():
        run(instance=instance)


if __name__ == '__main__':
    main()
