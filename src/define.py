class Define:
    default = "default"


class Crossover(Define):
    name = "crossover"
    pox = "pox"
    mox1 = "mox1"
    mox2 = "mox2"
    ipox = "ipox"
    dpox = "dpox"
    ox = "ox"
    pmx = "pmx"


class Mutation(Define):
    name = "mutation"
    tpe = "tpe"
    insert = "insert"
    sub_reverse = "sub_reverse"


class Selection(Define):
    name = "selection"
    roulette = "roulette"
    champion2 = "champion2"
    elite_strategy = "elite_strategy"
    champion = "champion"


class Name:
    # 车间类型
    jsp = "作业车间"
    fjsp = "柔性作业车间"
    fsp = "流水车间"
    hfsp = "混合流水车间"
    mrjsp = "多加工路径作业车间"
    mrfjsp = "多加工路径柔性作业车间"
    jspwcw = "考虑工人的作业车间"
    fjspwcw = "考虑工人的柔性作业车间"
    # 数据来源
    input_data = "输入数据"
    uniform_distribute = "均匀分布"
    benchmark = "标准算例"
    # 约束条件
    basic_constrain = "基本约束"
    work_timetable = "工作时间表"
    no_wait = "无等待"
    limited_wait = "等待时间有限"
    # 优化目标
    makespan = "工期"
    total_makespan = "工期之和"
    total_flow_time = "总流程时间"
    n_tardiness = "拖期工件数"
    tardiness = "总拖期"
    earliness = "总提前期"


class Para:
    # 调度类型
    shop_type = "shop_type"
    data_source = "data_source"
    constrain_condition = "constrain_condition"
    optimize_objective = "optimize_objective"
