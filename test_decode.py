from src import *


def main(instance="example"):
    time_unit = 1
    a = jsp_benchmark.instance[instance]
    n, m, p, tech, proc = Utils.string2data_jsp_fsp(a, int, time_unit)
    b = Utils.load_text("./src/data/limited_wait_jsp/%s.txt" % instance)
    c = Utils.string2data_wait(b, p, int, time_unit)
    problem = Utils.create_schedule(Jsp, n, m, p, tech, proc, limited_wait=c, time_unit=time_unit)
    """基于工序的编码"""
    # code = problem.spt()
    # print(code, "# code")
    # solution = problem.decode(code)
    # solution = problem.decode_limited_wait(code)
    """基于机器的编码"""
    code = problem.sequence_machine_based(n, m, problem.job)
    # solution = problem.decode_machine_based(code)
    solution = problem.decode_machine_based_limited_wait(code)
    """解码结果"""
    print(solution.code, "# solution.code")
    print(solution.schedule.direction, "# solution.schedule.direction")
    print(solution.schedule.makespan, "# solution.schedule.makespan")
    print(solution.schedule.sjike, "# solution.schedule.sjike")
    solution.save_gantt_chart_to_csv("./Result/%s.csv" % instance)
    solution.gantt_chart_png("./Result/GanttChart/%s.png" % instance)


if __name__ == '__main__':
    main(instance="example")
