from src import *


def main(instance="example"):
    time_unit = 1
    a = jsp_benchmark.instance[instance]
    n, m, p, tech, proc = Utils.string2data_jsp_fsp(a, int, time_unit)
    b = Utils.load_text("./src/data/limited_wait_jsp/%s.txt" % instance)
    c = Utils.string2data_wait(b, p, int, time_unit)
    problem = Utils.create_schedule(Jsp, n, m, p, tech, proc, limited_wait=c, time_unit=time_unit)
    # problem.spt_lpt_new(spt_or_lpt=1)
    # for node in problem.node_list_complete:
    #     print(node.value + 1, problem.decode(node.value, direction=0).schedule.makespan)
    # """基于工序的编码"""
    problem.spt_lpt_new(spt_or_lpt=0)
    # code = problem.spt()
    code = problem.node_list_complete[0].value
    print(code, "# code")
    # solution = problem.decode(code)
    solution = problem.decode_limited_wait(code)
    # solution = problem.decode_limited_wait_new(code)
    # solution = problem.decode_limited_wait_new_twice(code)
    # code = solution.trans_operation_based2machine_based()
    # print(code, "# solution.trans_operation_based2machine_based()")
    """基于机器的编码"""
    # code = problem.sequence_machine_based(n, m, problem.job)
    # solution = problem.decode_machine_based(code)
    # solution = problem.decode_machine_based_limited_wait(code)
    """解码结果"""
    print(solution.code, "# solution.code")
    print(solution.schedule.direction, "# solution.schedule.direction")
    print(solution.schedule.makespan, "# solution.schedule.makespan")
    print(solution.schedule.sjike[2], "# solution.schedule.sjike[2]")
    solution.save_code_to_txt("./Result/Code/%s.txt" % instance)
    solution.save_gantt_chart_to_csv("./Result/GanttChart/%s.csv" % instance)
    solution.gantt_chart_png("./Result/GanttChart/%s.png" % instance, key_block=True)


if __name__ == '__main__':
    main(instance="example")
