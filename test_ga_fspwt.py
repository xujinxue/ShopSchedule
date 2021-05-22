from src import *


def main(instance="example"):
    time_unit = 60
    a = fsp_simulation.case[instance]
    n, m, p, tech, proc = Utils.string2data_jsp_fsp(a, int, time_unit)
    rest_start_end = TimeTable.consistent5d8h(m, 2020, 7, 6, 30, connect_internet=False, save=False, show=False)
    problem = Utils.create_schedule(Fsp, n, m, p, tech, proc, best_known=None, time_unit=time_unit)
    problem2 = Utils.create_schedule(Fsp, n, m, p, tech, proc, rest_start_end=rest_start_end,
                                     resumable=Utils.crt_resumable(n, p, 0), best_known=None, time_unit=time_unit)
    ga = GaFspHfsp(pop_size=20, rc=0.85, rm=0.15, max_generation=int(10e4), objective=Objective.makespan,
                   schedule=problem, max_stay_generation=50)
    ga2 = GaFspHfspWorkTimetable(pop_size=ga.pop_size, rc=ga.rc, rm=ga.rm, max_generation=int(10e2),
                                 objective=Objective.makespan, schedule=problem2, max_stay_generation=20)
    ga.schedule.ga_operator[Crossover.name] = Crossover.pmx
    ga.schedule.ga_operator[Mutation.name] = Mutation.sub_reverse
    ga.schedule.ga_operator[Selection.name] = Selection.roulette
    ga.schedule.para_tabu = False
    ga.schedule.para_dislocation = False
    ga2.schedule.ga_operator[Crossover.name] = Crossover.pmx
    ga2.schedule.ga_operator[Mutation.name] = Mutation.sub_reverse
    ga2.schedule.ga_operator[Selection.name] = Selection.roulette
    ga2.schedule.para_tabu = True
    ga2.schedule.para_dislocation = False
    obj_list = []
    obj_list2 = []
    for i in range(1, N_EXP + 1):
        ga.do_evolution(exp_no=i)
        Utils.save_record_to_csv("./GA_FSPWT/%s/%s-record.csv" % (instance, i), ga.record)
        ga.best[0].save_code_to_txt("./GA_FSPWT/%s/%s-code.txt" % (instance, i))
        ga.best[0].save_gantt_chart_to_csv("./GA_FSPWT/%s/%s-GanttChart.csv" % (instance, i))
        obj_list.append([ga.best[1], ga.record[2].index(ga.best[1]), ga.best[0].schedule.direction])
        ga2.do_evolution(pop=ga.pop, exp_no="%s-s2" % i)
        Utils.save_record_to_csv("./GA_FSPWT/%s/%s-record-s2.csv" % (instance, i), ga2.record)
        ga2.best[0].save_code_to_txt("./GA_FSPWT/%s/%s-code-s2.txt" % (instance, i))
        ga2.best[0].save_gantt_chart_to_csv("./GA_FSPWT/%s/%s-GanttChart-s2.csv" % (instance, i))
        obj_list2.append([ga2.best[1], ga2.record[2].index(ga2.best[1]), ga2.best[0].schedule.direction])
    Utils.save_obj_to_csv("./GA_FSPWT/%s.csv" % instance, obj_list)
    Utils.save_obj_to_csv("./GA_FSPWT/%s-s2.csv" % instance, obj_list2)


def exp():
    Utils.make_dir("./GA_FSPWT")
    for instance in CASES_LIST.split():
        Utils.make_dir("./GA_FSPWT/%s" % instance)
        Utils.make_dir("./GA_FSPWT/%s/GanttChart" % instance)
        main(instance=instance)


if __name__ == '__main__':
    exp()
