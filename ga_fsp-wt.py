__doc__ = """
基于遗传算法求解考虑作息时间的流水车间调度问题；
加工不可恢复（加工时间不可拆分）；
改进解码算法；
"""

from src import *


def run(case="case1"):
    a = fsp_simulation.case[case]  # 获取case1的文本数据
    n, m, p, tech, proc = Utils.string2data_jsp_fsp(a, time_unit=60)  # 文本数据转换
    resumable = Utils.crt_resumable(n, p, 0)  # 生成加工不可恢复数据
    rest_start_end = TimeTable.consistent_timetable_b8w5t2(m, 2020, 7, 6, 30, show=True)  # 生成机器的开工停工时刻
    ga = GaTemplateFspTimetable(pop_size=50, rc=0.75, rm=0.15, max_generation=500, objective=Objective.makespan,
                                n=n, m=m, p=p, tech=tech, proc=proc,
                                rest_start_end=rest_start_end, resumable=resumable, time_unit=60)  # 调用写好的模板
    ga.do_exp(exp_log="./GA_FSP-WT", instance="case1", n_exp=10, tabu_search=True, gc=False, y_based=0)  # 求解
    # 结果保存在exp_log（例："./GA_FSP-WT"）及exp_log/instance（例："./GA_FSP-WT/case1"）目录下


def main():
    for case in CASES_LIST.split():
        run(case=case)


if __name__ == '__main__':
    main()
