from src import *


def main(instance="example"):
    time_unit = 1
    # a = mrjsp_benchmark.instance[instance]
    # n, m, p, tech, proc = Utils.string2data_mrjsp(a, int, time_unit)
    # best_known = mrjsp_benchmark.best_known[instance]
    a = Utils.load_text("./src/data/mrjsp/%s.txt" % instance)
    n, m, p, tech, proc = Utils.string2data_mrjsp(a, int, time_unit)
    best_known = None
    problem = Utils.create_schedule(Jsp, n, m, p, tech, proc, best_known=best_known, time_unit=time_unit,
                                    multi_route=True)
    ga = GaMrJsp(pop_size=20, rc=0.85, rm=0.15, max_generation=int(10e4), objective=Objective.makespan,
                 schedule=problem, max_stay_generation=5)
    ga.schedule.ga_operator[Crossover.name] = Crossover.pox
    ga.schedule.ga_operator[Mutation.name] = Mutation.tpe
    ga.schedule.ga_operator[Selection.name] = Selection.roulette
    ga.schedule.para_key_block_move = False
    ga.schedule.para_tabu = False
    ga.schedule.para_dislocation = False
    GaTemplate(save="GA_MRJSP", instance=instance, ga=ga, n_exp=10)


def exp():
    for instance in INSTANCE_LIST_MRJSP.split():
        main(instance=instance)


if __name__ == '__main__':
    exp()
