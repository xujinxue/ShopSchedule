from src import *


def run(case="case1"):
    a = fsp_simulation.case[case]
    n, m, p, tech, proc = Utils.string2data_jsp_fsp(a, time_unit=60)
    resumable = Utils.crt_resumable(n, p, 0)
    rest_start_end = TimeTable.consistent_timetable_b8w5t2(m, 2020, 7, 6, 30, show=True)
    ga = GaTemplateFspTimetable(pop_size=50, rc=0.75, rm=0.15, max_generation=500, objective=Objective.makespan,
                                n=n, m=m, p=p, tech=tech, proc=proc,
                                rest_start_end=rest_start_end, resumable=resumable, time_unit=60)
    ga.do_exp(exp_log="./GA_FSP-WT", instance="case1", n_exp=10, tabu_search=False,
              gc=False, y_based=0)


def main():
    for case in CASES_LIST.split():
        run(case=case)


if __name__ == '__main__':
    main()
