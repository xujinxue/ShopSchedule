from src import *


def main(instance="example"):
    time_unit = 1
    a = jsp_benchmark.instance[instance]
    n, m, p, tech, proc = Utils.string2data_jsp_fsp(a, int, time_unit)
    best_known = jsp_benchmark.best_known[instance]
    problem = Utils.create_schedule(Jsp, n, m, p, tech, proc, best_known=best_known, time_unit=time_unit)
    ga = GaJsp(pop_size=20, rc=0.85, rm=0.15, max_generation=int(10e4), objective=Objective.makespan,
               schedule=problem, max_stay_generation=50)
    obj_list = []
    for i in range(1, N_EXP + 1):
        ga.do_evolution(tabu_search=True, key_block_move=False, exp_no=i)
        Utils.save_record_to_csv("./GA_JSP/%s/%s-record.csv" % (instance, i), ga.record)
        ga.best[0].save_code_to_txt("./GA_JSP/%s/%s-code.txt" % (instance, i))
        ga.best[0].save_gantt_chart_to_csv("./GA_JSP/%s/%s-GanttChart.csv" % (instance, i))
        obj_list.append([ga.best[1], ga.record[2].index(ga.best[1]), ga.best[0].schedule.direction])
    Utils.save_obj_to_csv("./GA_JSP/%s.csv" % instance, obj_list)


def exp():
    Utils.make_dir("./GA_JSP")
    for instance in INSTANCE_LIST_JSP.split():
        Utils.make_dir("./GA_JSP/%s" % instance)
        Utils.make_dir("./GA_JSP/%s/GanttChart" % instance)
        main(instance=instance)


if __name__ == '__main__':
    exp()
