import copy

import numpy as np

deepcopy = copy.deepcopy


class Rule:
    def __init__(self, job):
        self.job = job
        self.job_copy = None
        self.wait = []
        self.wait_machine = []
        self.wait_duration = []
        self.wait_wait = []
        self.wait_wait_remain = []
        self.wait_wait_done = []
        self.wait_remain = []
        self.wait_done = []
        self.wait_remain_time = []
        self.wait_done_time = []
        self.done = np.array([], dtype=int)

    def reset_schedule_rule(self):
        self.job_copy = deepcopy(self.job)
        self.wait = [(i, 0) for i in self.job.keys()]
        self.wait_machine = []
        self.wait_duration = []
        self.wait_wait = []
        self.wait_wait_remain = []
        self.wait_wait_done = []
        self.wait_remain = []
        self.wait_done = []
        self.wait_remain_time = []
        self.wait_done_time = []
        self.done = np.array([], dtype=int)

    def do_update(self, choice, wait_data):
        decision = self.wait[choice]
        job, operation = decision
        self.job_copy[job].task.pop(operation)
        self.wait.pop(choice)
        self.wait_machine.pop(choice)
        wait_data.pop(choice)
        self.done = np.append(self.done, job)
        next_operation = operation + 1
        if next_operation < self.job[job].nop:
            self.wait.append((job, next_operation))

    def do_rule(self, func, get):
        pass

    def get_remain_from_wait(self):
        self.wait_remain = [self.job_copy[i].nop for i, j in self.wait]
        return self.wait_remain

    def get_done_from_wait(self):
        self.wait_done = [self.job[i].nop - self.job_copy[i].nop for i, j in self.wait]
        return self.wait_done

    def get_limited_wait_from_wait(self):
        self.wait_wait = [self.job[i].task[j].limited_wait for i, j in self.wait]
        return self.wait_wait

    def get_remain_limited_wait_from_wait(self):
        self.wait_wait_remain = [self.job[i].remain_limited_wait(j) for i, j in self.wait]
        return self.wait_wait_remain

    def get_done_limited_wait_from_wait(self):
        self.wait_wait_done = [self.job[i].done_limited_wait(j) for i, j in self.wait]
        return self.wait_wait_done

    def swt(self):
        return self.do_rule(np.argmin, self.get_limited_wait_from_wait)

    def lwt(self):
        return self.do_rule(np.argmax, self.get_limited_wait_from_wait)

    def mrwt(self):
        return self.do_rule(np.argmax, self.get_remain_limited_wait_from_wait)

    def lrwt(self):
        return self.do_rule(np.argmin, self.get_remain_limited_wait_from_wait)

    def mdwt(self):
        return self.do_rule(np.argmax, self.get_done_limited_wait_from_wait)

    def ldwt(self):
        return self.do_rule(np.argmin, self.get_done_limited_wait_from_wait)

    def mro(self):
        return self.do_rule(np.argmax, self.get_remain_from_wait)

    def lro(self):
        return self.do_rule(np.argmin, self.get_remain_from_wait)

    def mdo(self):
        return self.do_rule(np.argmax, self.get_done_from_wait)

    def ldo(self):
        return self.do_rule(np.argmin, self.get_done_from_wait)


class RuleJsp(Rule):
    def __init__(self, job):
        Rule.__init__(self, job)

    def get_machine_from_wait(self):
        self.wait_machine = [self.job[i].task[j].machine for i, j in self.wait]
        return self.wait_machine

    def get_duration_from_wait(self):
        self.wait_duration = [self.job[i].task[j].duration for i, j in self.wait]
        return self.wait_duration

    def get_done_time_from_wait(self):
        self.wait_done_time = [self.job_copy[i].done(j) for i, j in self.wait]
        return self.wait_done_time

    def get_remain_time_from_wait(self):
        self.wait_remain_time = [self.job_copy[i].remain(j) for i, j in self.wait]
        return self.wait_remain_time

    def do_rule(self, func, get):
        self.reset_schedule_rule()
        while any([job.nop for job in self.job_copy.values()]):
            self.get_machine_from_wait()
            wait_data = get()
            for i, j in enumerate(set(self.wait_machine)):
                n_mac_j = self.wait_machine.count(j)
                if n_mac_j == 1:
                    self.do_update(self.wait_machine.index(j), wait_data)
                elif n_mac_j > 1:
                    index = np.argwhere(np.array(self.wait_machine) == j)[:, 0]
                    proc = np.array(wait_data)[index]
                    index_value = func(proc)
                    choice = np.argwhere(proc == proc[index_value])[:, 0]
                    # choose one from choice
                    index_choice = np.random.choice(choice, 1, replace=False)[0]
                    self.do_update(index[index_choice], wait_data)
                    # choose all from choice
                    # for u, v in enumerate(choice):
                    #     self.do_update(index[v] - u, wait_data)
        return self.done

    def spt(self):
        return self.do_rule(np.argmin, self.get_duration_from_wait)

    def lpt(self):
        return self.do_rule(np.argmax, self.get_duration_from_wait)

    def mrt(self):
        return self.do_rule(np.argmax, self.get_remain_time_from_wait)

    def lrt(self):
        return self.do_rule(np.argmin, self.get_remain_time_from_wait)

    def mdt(self):
        return self.do_rule(np.argmax, self.get_done_time_from_wait)

    def ldt(self):
        return self.do_rule(np.argmin, self.get_done_time_from_wait)
