from src import *


def main(instance="example"):
    time_unit = 1
    a = jsp_benchmark.instance[instance]
    n, m, p, tech, proc = Utils.string2data_jsp_fsp(a, int, time_unit)
    best_known = jsp_benchmark.best_known[instance]
    problem = Utils.create_schedule(Jsp, n, m, p, tech, proc, best_known=best_known, time_unit=time_unit)
    ga = GaJsp(pop_size=20, rc=0.85, rm=0.15, max_generation=int(10e4), objective=Objective.makespan, schedule=problem,
               max_stay_generation=500)
    ga.do_evolution(tabu_search=False, key_block_move=False)
    Utils.make_dir("./GA_JSP")
    Utils.make_dir("./GA_JSP/%s" % instance)
    Utils.make_dir("./GA_JSP/%s/GanttChart" % instance)
    Utils.clear_dir("./GA_JSP/%s" % instance)
    Utils.clear_dir("./GA_JSP/%s/GanttChart" % instance)
    Utils.save_record_to_csv("./GA_JSP/%s/record.csv" % instance, ga.record)
    ga.best[0].save_code_to_txt("./GA_JSP/%s/code.txt" % instance)
    ga.best[0].save_gantt_chart_to_csv("./GA_JSP/%s/GanttChart.csv" % instance)
    # ga.best[0].gantt_chart_png("./GA_JSP/%s/GanttChart/GanttChart.png" % instance, key_block=True)


if __name__ == '__main__':
    main(instance="ft06")
