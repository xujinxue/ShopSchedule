from src import *


def main(instance="example"):
    time_unit = 1
    a = jsp_benchmark.instance[instance]
    n, m, p, tech, proc = Utils.string2data_jsp_fsp(a, int, time_unit)
    best_known = jsp_benchmark.best_known[instance]
    problem = Utils.create_schedule(Jsp, n, m, p, tech, proc, best_known=best_known, time_unit=time_unit)
    # code = Code.sequence_operation_based(n, p)
    # solution = problem.decode(code)
    # neg_complete, evaluate = solution.key_block_move_complete()
    # a = []
    # for i in neg_complete:
    #     a.append(problem.decode(i[0]).schedule.makespan)
    # print(evaluate, "# e")
    # print(a, "# a")
    ga = GaJsp(pop_size=20, rc=0.85, rm=0.15, max_generation=int(10e4), objective=Objective.makespan,
               schedule=problem, max_stay_generation=50)
    ga.schedule.ga_operator[Crossover.name] = Crossover.pox
    ga.schedule.ga_operator[Mutation.name] = Mutation.tpe
    ga.schedule.ga_operator[Selection.name] = Selection.roulette
    ga.schedule.para_key_block_move = True
    ga.schedule.para_tabu = False
    ga.schedule.para_dislocation = False
    GaTemplate(save="GA_JSP", instance=instance, ga=ga, n_exp=10)


def exp():
    for instance in INSTANCE_LIST_JSP.split():
        main(instance=instance)


if __name__ == '__main__':
    exp()
