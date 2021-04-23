class Objective:
    @staticmethod
    def makespan(info):
        return max([machine.end for machine in info.schedule.machine.values()]) / info.schedule.time_unit

    @staticmethod
    def total_makespan(info):
        return sum([job.end for job in info.schedule.job.values()]) / info.schedule.time_unit

    @staticmethod
    def total_flow_time(info):
        return sum([job.end - job.start for job in info.schedule.job.values()]) / info.schedule.time_unit

    @staticmethod
    def tardiness(info):
        time_unit = info.schedule.time_unit
        return sum([max([0, job.end - job.due_date]) for job in info.schedule.job.values()]) / time_unit

    @staticmethod
    def earliness(info):
        time_unit = info.schedule.time_unit
        return sum([max([0, job.due_date - job.end]) for job in info.schedule.job.values()]) / time_unit
