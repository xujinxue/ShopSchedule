from src import *


def main(instance="example"):
    time_unit = 1
    a = mrjsp_benchmark.instance[instance]
    n, m, p, tech, proc = Utils.string2data_mrjsp(a, int, time_unit)
    problem = Utils.create_schedule(Jsp, n, m, p, tech, proc, multi_route=True)
    r = [job.nor for job in problem.job.values()]
    code = problem.sequence_operation_based(n, p)
    route = problem.assignment_route(n, r)
    solution = problem.decode(code, route)
    print(solution.code, "# solution.code")
    print(solution.route, "# solution.route")
    print(solution.schedule.direction, "# solution.schedule.direction")
    print(solution.schedule.makespan, "# solution.schedule.makespan")
    print(solution.schedule.sjike[2], "# solution.schedule.sjike[2]")
    solution.save_code_to_txt("./Result/Code/%s.txt" % instance)
    solution.save_gantt_chart_to_csv("./Result/GanttChart/%s.csv" % instance)
    solution.gantt_chart_png("./Result/GanttChart/%s.png" % instance, key_block=True)


if __name__ == '__main__':
    main(instance="example")
