from src import *


def main(instance="example"):
    time_unit = 1
    a = mrfjsp_benchmark.instance[instance]
    n, m, p, tech, proc = Utils.string2data_mrfjsp(a, int, time_unit)
    best_known = mrfjsp_benchmark.best_known[instance]
    problem = Utils.create_schedule(Fjsp, n, m, p, tech, proc, best_known=best_known, time_unit=time_unit,
                                    multi_route=True)
    ga = GaMrFjsp(pop_size=20, rc=0.85, rm=0.15, max_generation=int(10e4), objective=Objective.makespan,
                  schedule=problem, max_stay_generation=50)
    ga.do_evolution(tabu_search=False, key_block_move=False)
    Utils.make_dir("./GA_MRFJSP")
    Utils.make_dir("./GA_MRFJSP/%s" % instance)
    Utils.make_dir("./GA_MRFJSP/%s/GanttChart" % instance)
    # Utils.clear_dir("./GA_MRFJSP/%s" % instance)
    # Utils.clear_dir("./GA_MRFJSP/%s/GanttChart" % instance)
    Utils.save_record_to_csv("./GA_MRFJSP/%s/record.csv" % instance, ga.record)
    ga.best[0].save_code_to_txt("./GA_MRFJSP/%s/code.txt" % instance)
    ga.best[0].save_gantt_chart_to_csv("./GA_MRFJSP/%s/GanttChart.csv" % instance)
    # ga.best[0].gantt_chart_png("./GA_MRFJSP/%s/GanttChart/GanttChart.png" % instance, key_block=True)


if __name__ == '__main__':
    main(instance="example")
