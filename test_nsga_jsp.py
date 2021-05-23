__doc__ = """
基于非支配排序遗传算法求解多目标作业车间调度问题；

"""

from src import *


def run(instance="ft06"):
    time_unit = 1
    a = jsp_benchmark.instance[instance]
    b = jsp_benchmark.due_date[instance]
    n, m, p, tech, proc = Utils.string2data_jsp_fsp(a, int, time_unit)
    due_date = list(map(int, b.split()))
    problem = Utils.create_schedule(Jsp, n, m, p, tech, proc, due_date=due_date, time_unit=time_unit)
    objective_list = [Objective.tardiness, Objective.makespan]
    nsga = NsgaJsp(pop_size=40, rc=0.85, rm=0.15, max_generation=50, objective=objective_list, schedule=problem)
    nsga.schedule.ga_operator[Crossover.name] = Crossover.pox
    nsga.schedule.ga_operator[Mutation.name] = Mutation.tpe
    nsga.schedule.ga_operator[Selection.name] = Selection.elite_strategy
    nsga.schedule.para_key_block_move = False
    NsgaTemplate(save="NSGA_JSP", instance=instance, nsga=nsga, n_exp=10, n_level=5, column=0)


def main():
    for instance in INSTANCE_LIST_JSP.split():
        run(instance=instance)


if __name__ == '__main__':
    main()
