__doc__ = """
基于非支配排序遗传算法求解多目标作业车间调度问题；

"""

import matplotlib.pyplot as plt

from src import *


def run(instance="ft06"):
    time_unit = 1
    a = jsp_benchmark.instance[instance]
    b = jsp_benchmark.due_date[instance]
    n, m, p, tech, proc = Utils.string2data_jsp_fsp(a, int, time_unit)
    due_date = list(map(int, b.split()))
    problem = Utils.create_schedule(Jsp, n, m, p, tech, proc, due_date=due_date, time_unit=time_unit)
    objective_list = [Objective.tardiness, Objective.makespan]
    nsga = NSGAJsp(pop_size=40, rc=0.85, rm=0.15, max_generation=50, objective=objective_list, schedule=problem)
    nsga.schedule.ga_operator[Crossover.name] = Crossover.pox
    nsga.schedule.ga_operator[Mutation.name] = Mutation.tpe
    nsga.schedule.ga_operator[Selection.name] = Selection.elite_strategy
    nsga.schedule.para_key_block_move = False
    c = nsga.do_evolution(pop=None, n_level=5, column=0)
    # 输出结果
    Utils.make_dir("./NSGA_JSP")
    Utils.make_dir("./NSGA_JSP/%s" % instance)
    Utils.make_dir("./NSGA_JSP/%s/GanttChart" % instance)
    # Utils.clear_dir("./NSGA_JSP/%s" % instance)
    # Utils.clear_dir("./NSGA_JSP/%s/GanttChart" % instance)
    plt.figure(figsize=[9, 5])
    res = ""
    for i, j in enumerate(c):
        d = [[], []]
        res += "帕累托等级-%s\n" % (i + 1)
        for k in j:
            res += "%s\n" % str(k)
            d[0].append(k[1][0])
            d[1].append(k[1][1])
        plt.plot(d[0], d[1], "--o", label="帕累托等级-%s" % (i + 1))
    with open("./NSGA_JSP/%s.txt" % instance, "w", encoding="utf-8") as f:
        f.writelines(res)
    print(res)
    plt.legend(loc="best")
    plt.margins()
    plt.tight_layout()
    plt.gcf().subplots_adjust(left=0.08, bottom=0.12)
    plt.xlabel("拖期")
    plt.ylabel("工期")
    plt.savefig("./NSGA_JSP/%s-ParetoRank.png" % instance, dpi=200)
    for i in range(len(c[0])):
        res = c[0][i][0]
        res.save_code_to_txt("./NSGA_JSP/%s/%s-code.txt" % (instance, i + 1))
        res.save_gantt_chart_to_csv("./NSGA_JSP/%s/%s-GanttChart.csv" % (instance, i + 1))
        # res.gantt_chart_png("./NSGA_JSP/%s/GanttChart/%s-GanttChart.png" % (instance, i + 1), key_block=True)


def main():
    for instance in INSTANCE_LIST_JSP.split():
        run(instance=instance)


if __name__ == '__main__':
    main()
