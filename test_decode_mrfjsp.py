from src import *


def main(instance="example"):
    time_unit = 1
    a = mrfjsp_benchmark.instance[instance]
    n, m, p, tech, proc = Utils.string2data_mrfjsp(a, int, time_unit)
    problem = Utils.create_schedule(Fjsp, n, m, p, tech, proc, multi_route=True)
    r = [job.nor for job in problem.job.values()]
    code = problem.sequence_operation_based(n, p)
    route = problem.assignment_route(n, r)
    # mac = problem.assignment_job_based_route(n, p, tech, route)
    # solution = problem.decode(code, mac, route)
    solution = problem.decode_one(code, route)
    print(solution.code, "# solution.code")
    print(solution.route, "# solution.route")
    print(solution.mac, "# solution.mac")
    print(solution.schedule.direction, "# solution.schedule.direction")
    print(solution.schedule.makespan, "# solution.schedule.makespan")
    print(solution.schedule.sjike[2], "# solution.schedule.sjike[2]")
    solution.save_gantt_chart_to_csv("./Result/GanttChart/%s.csv" % instance)
    # solution.gantt_chart_png("./Result/GanttChart/%s.png" % instance)


if __name__ == '__main__':
    main(instance="example")
