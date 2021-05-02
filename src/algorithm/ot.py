__doc__ = """
ORTools求解器
def OrToolsJspSat: Jsp的ORTools模型及求解
"""

import collections

from ortools.sat.python import cp_model

"""
求解状态对应的status值(源程序cp_model_pb2.py中查看): cp_model -> cp_model_pb2
UNKNOWN = 0 # 未知状态
MODEL_INVALID = 1 # 模型不正确
FEASIBLE = 2 # 可行解
INFEASIBLE = 3 # 无可行解
OPTIMAL = 4 # 最优解
"""


def OrToolsJspSat(instance, all_machines, jobs_data, max_solve_time, log_dir, limited_wait=None, due_date=None):
    # 第1步, 准备数据（做为参数进行传递）
    # 第2步, 建立CPModel, 即Constraint program model
    model = cp_model.CpModel()
    horizon = sum(task[1] for job in jobs_data for task in job)
    # 第3步, 定义变量
    task_type = collections.namedtuple('task_type', 'start end interval')
    all_tasks = {}
    machine_to_intervals = collections.defaultdict(list)
    for i, job in enumerate(jobs_data):  # i: 工件
        for j, task in enumerate(job):  # j: 机器
            machine, duration = task[0], task[1]
            suffix = '_%i_%i' % (i, j)
            start_var = model.NewIntVar(0, horizon, 'start' + suffix)
            end_var = model.NewIntVar(0, horizon, 'end' + suffix)
            interval_var = model.NewIntervalVar(start_var, duration, end_var, 'interval' + suffix)
            all_tasks[i, j] = task_type(start=start_var, end=end_var, interval=interval_var)
            machine_to_intervals[machine].append(interval_var)
    # 第4步, 添加约束
    for machine in all_machines:  # 加工无重叠约束
        model.AddNoOverlap(machine_to_intervals[machine])
    for i, job in enumerate(jobs_data):
        for j in range(len(job) - 1):
            model.Add(all_tasks[i, j + 1].start >= all_tasks[i, j].end)  # 加工路径约束
            if limited_wait is not None:  # 等待时间有限约束
                if limited_wait[i][j] >= 0:
                    model.Add(all_tasks[i, j + 1].start - all_tasks[i, j].end <= limited_wait[i][j])
        if due_date is not None:  # 交货期约束
            model.Add(all_tasks[i, len(job) - 1].end <= due_date[i])
    # 第5步, 定义优化目标
    # 目标函数为最小化工期
    obj_var = model.NewIntVar(0, horizon, 'makespan')
    model.AddMaxEquality(obj_var, [all_tasks[i, len(job) - 1].end for i, job in enumerate(jobs_data)])
    # 目标函数为最小化工期之和
    # obj_var = model.NewIntVar(0, horizon, 'makespan')
    # model.AddMaxEquality(obj_var, [all_tasks[i, len(job) - 1].end for i, job in enumerate(jobs_data)])
    # obj_var_copy = obj_var
    # for i, job in enumerate(jobs_data):
    #     obj_var += all_tasks[i, len(job) - 1].end
    # obj_var -= obj_var_copy
    model.Minimize(obj_var)

    def solve(cur_exp):
        # 第6步, 创建CpSolver
        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = max_solve_time  # 设置最大求解时间
        # 第7步, 求解
        status = solver.Solve(model)  # 求解model并返回求解状态
        # 第8步, 记录结果
        if status in [2, 4]:  # 可行解或最优解
            assigned_jobs = collections.defaultdict(list)
            assigned_task_type = collections.namedtuple('assigned_task_type', 'start job index duration')
            for job_id, job in enumerate(jobs_data):
                for task_id, task in enumerate(job):
                    assigned_jobs[task[0]].append(
                        assigned_task_type(start=solver.Value(all_tasks[job_id, task_id].start), job=job_id,
                                           index=task_id, duration=task[1]))
            with open("%s/%s-%s-ganttChart.csv" % (log_dir, instance, cur_exp + 1), "w", encoding='utf-8') as f:
                f.writelines("Job,Operation,Machine,Start,Duration,End\n")
                for machine_id in all_machines:
                    for assigned_task in assigned_jobs[machine_id]:
                        f.writelines("{},{},{},{},{},{}\n".format(
                            assigned_task.job + 1, assigned_task.index + 1, machine_id + 1,
                            assigned_task.start, assigned_task.duration, assigned_task.start + assigned_task.duration))
        return solver.WallTime(), solver.ObjectiveValue(), status  # 求解用时, 求解得到的目标值, 求解状态

    return solve
