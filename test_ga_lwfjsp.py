from src import *


def main(instance="example"):
    time_unit = 1
    a = fjsp_benchmark.instance[instance]
    n, m, p, tech, proc = Utils.string2data_fjsp_hfsp(a, int, time_unit)
    lw = Utils.string2data_wait(Utils.load_text("./src/data/limited_wait_fjsp/%s.txt" % instance), p, int, time_unit)
    best_known = fjsp_benchmark.best_known[instance]
    problem = Utils.create_schedule(Fjsp, n, m, p, tech, proc, limited_wait=lw, best_known=best_known,
                                    time_unit=time_unit)
    ga = GaLwFjsp(pop_size=20, rc=0.85, rm=0.15, max_generation=int(10e4), objective=Objective.makespan,
                  schedule=problem, max_stay_generation=50)
    ga.schedule.ga_operator[Crossover.name] = Crossover.ox
    ga.schedule.ga_operator[Mutation.name] = Mutation.tpe
    ga.schedule.ga_operator[Selection.name] = Selection.roulette
    ga.schedule.para_key_block_move = False
    ga.schedule.para_tabu = False
    ga.schedule.para_dislocation = False
    obj_list = []
    for i in range(1, N_EXP + 1):
        ga.do_evolution(exp_no=i)
        Utils.save_record_to_csv("./GA_LWFJSP/%s/%s-record.csv" % (instance, i), ga.record)
        ga.best[0].save_code_to_txt("./GA_LWFJSP/%s/%s-code.txt" % (instance, i))
        ga.best[0].save_gantt_chart_to_csv("./GA_LWFJSP/%s/%s-GanttChart.csv" % (instance, i))
        obj_list.append([ga.best[1], ga.record[2].index(ga.best[1]), ga.best[0].schedule.direction])
    Utils.save_obj_to_csv("./GA_LWFJSP/%s.csv" % instance, obj_list)


def exp():
    Utils.make_dir("./GA_LWFJSP")
    for instance in INSTANCE_LIST_LWFJSP.split():
        Utils.make_dir("./GA_LWFJSP/%s" % instance)
        Utils.make_dir("./GA_LWFJSP/%s/GanttChart" % instance)
        main(instance=instance)


if __name__ == '__main__':
    exp()
