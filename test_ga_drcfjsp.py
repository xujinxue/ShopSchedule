from src import *


def main(instance="DMFJS01"):
    time_unit = 1
    # a = drcfjsp_benchmark.instance[instance]
    # n, m, w, p, tech, worker, proc = Utils.string2data_drcfjsp(a, int, time_unit)
    # best_known = drcfjsp_benchmark.best_known[instance]
    a = Utils.load_text("./src/data/drcfjsp/%s.txt" % instance)
    n, m, w, p, tech, worker, proc = Utils.string2data_drcfjsp(a, int, time_unit)
    best_known = None
    problem = Utils.create_schedule(Fjsp, n, m, p, tech, proc, w=w, worker=worker, best_known=best_known,
                                    time_unit=time_unit)
    ga = GaDrcFjspNew(pop_size=20, rc=0.85, rm=0.15, max_generation=int(10e4), objective=Objective.makespan,
                      schedule=problem, max_stay_generation=50)
    ga.schedule.ga_operator[Crossover.name] = Crossover.ipox
    ga.schedule.ga_operator[Mutation.name] = Mutation.tpe
    ga.schedule.ga_operator[Selection.name] = Selection.roulette
    ga.schedule.para_key_block_move = False
    ga.schedule.para_tabu = False
    ga.schedule.para_dislocation = False
    GaTemplate(save="GA_DRCFJSP", instance=instance, ga=ga, n_exp=10)


def exp():
    for instance in INSTANCE_LIST_DRCFJSP.split():
        main(instance=instance)


if __name__ == '__main__':
    exp()
