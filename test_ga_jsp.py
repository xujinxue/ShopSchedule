from src import *


def main(instance="example"):
    time_unit = 1
    a = jsp_benchmark.instance[instance]
    n, m, p, tech, proc = Utils.string2data_jsp_fsp(a, int, time_unit)
    best_known = [jsp_benchmark.best_known[instance], None][1]
    problem = Utils.create_schedule(Jsp, n, m, p, tech, proc, best_known=best_known, time_unit=time_unit)
    ga = GaJsp(pop_size=10, rc=0.85, rm=0.15, max_generation=int(10e4), objective=Objective.makespan, schedule=problem,
               max_stay_generation=50)
    ga.do_evolution(tabu_search=False, key_block_move=False)
    Utils.save_record_to_csv("./Result/Record/%s.csv" % instance, ga.record)
    ga.best[0].save_gantt_chart_to_csv("./Result/GanttChart/%s.csv" % instance)
    # ga.best[0].gantt_chart_png("./Result/GanttChart/%s.png" % instance)


if __name__ == '__main__':
    main(instance="ft06")
