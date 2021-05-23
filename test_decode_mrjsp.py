from src import *


def main(instance="example2"):
    time_unit = 1
    a = mrjsp_benchmark.instance[instance]
    n, m, p, tech, proc = Utils.string2data_mrjsp(a, int, time_unit)
    problem = Utils.create_schedule(Jsp, n, m, p, tech, proc, multi_route=True)
    r = [job.nor for job in problem.job.values()]
    code = problem.sequence_operation_based(n, p)
    route = problem.assignment_route(n, r)
    solution = problem.decode(code, route, direction=0)
    solution.print()
    solution.save_code_to_txt("./Result/Code/%s.txt" % instance)
    solution.save_gantt_chart_to_csv("./Result/GanttChart/%s.csv" % instance)
    solution.gantt_chart_png("./Result/GanttChartPngHtml/%s.png" % instance, key_block=True)


if __name__ == '__main__':
    main(instance="example2")
