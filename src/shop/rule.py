import copy

import numpy as np

deepcopy = copy.deepcopy


class Node:
    def __init__(self, value, wait, job):
        self.value = value
        self.wait = wait
        self.job_copy = deepcopy(job)


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
        self.task_on_machine_operation = [[] for _ in range(3)]
        self.task_on_machine_duration = [[] for _ in range(3)]
        self.node_list = []
        self.node_list_complete = []

    def rule_init_task_on_machine(self, m):
        self.task_on_machine_operation = [[] for _ in range(m)]
        self.task_on_machine_duration = [[] for _ in range(m)]
        for i in range(m):
            for u, v in self.job.items():
                for p, q in v.task.items():
                    if q.machine == i:
                        self.task_on_machine_operation[i].append((u, p))
                        self.task_on_machine_duration[i].append(q.duration)

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

    def reset_schedule_rule_new(self):
        self.job_copy = deepcopy(self.job)
        self.wait = [(i, 0) for i in self.job.keys()]
        self.node_list = []
        self.node_list_complete = []

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

    def all_task_is_assignment(self):
        return any([job.nop for job in self.job_copy.values()])

    def do_rule(self, func, get):
        self.reset_schedule_rule()
        while self.all_task_is_assignment():
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
                    index_choice = np.random.choice(choice, 1, replace=False)[0]
                    self.do_update(index[index_choice], wait_data)
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

    def spt_lpt_new(self, spt_or_lpt=0):
        self.reset_schedule_rule_new()
        a = deepcopy(self.task_on_machine_operation)
        for i, j in enumerate(self.task_on_machine_duration):
            b = np.argsort(j)
            if spt_or_lpt == 1:
                b = b[::-1]
            for u, v in enumerate(b):
                a[i][u] = self.task_on_machine_operation[i][v]
        location = np.array([a[self.job[v[0]].task[v[1]].machine].index(v) for v in self.wait])
        index_location = np.argwhere(location == min(location))[:, 0]
        remain = np.array([self.job_copy[v[0]].nop for v in self.wait])[index_location]
        index_remain = np.argwhere(remain == max(remain))[:, 0]
        index = index_location[index_remain]
        for i in index:
            job_copy = deepcopy(self.job_copy)
            wait = deepcopy(self.wait)
            job, operation = wait[i]
            job_copy[job].task.pop(operation)
            wait.remove(wait[i])
            next_operation = operation + 1
            if next_operation < self.job[job].nop:
                wait.append((job, next_operation))
            node_new = Node(np.array([job], dtype=int), wait, job_copy)
            self.node_list.append(node_new)
        while len(self.node_list):
            node = deepcopy(self.node_list[0])
            location = np.array(
                [a[self.job[v[0]].task[v[1]].machine].index(v) for v in node.wait])
            index_location = np.argwhere(location == min(location))[:, 0]
            remain = np.array([node.job_copy[v[0]].nop for v in node.wait])[index_location]
            index_remain = np.argwhere(remain == max(remain))[:, 0]
            index = index_location[index_remain]
            if index.shape[0] > 1:
                self.node_list.pop(0)
                for i in index:
                    job_copy = deepcopy(node.job_copy)
                    wait = deepcopy(node.wait)
                    value = deepcopy(node.value)
                    job, operation = wait[i]
                    job_copy[job].task.pop(operation)
                    wait.remove(wait[i])
                    next_operation = operation + 1
                    if next_operation < self.job[job].nop:
                        wait.append((job, next_operation))
                    value = np.append(value, job)
                    node_new = Node(value, wait, job_copy)
                    if len(node_new.wait) == 0:
                        self.node_list_complete.append(node_new)
                    else:
                        self.node_list.append(node_new)
            elif index.shape[0] == 1:
                i = index[0]
                job, operation = node.wait[i]
                node.job_copy[job].task.pop(operation)
                node.wait.remove(node.wait[i])
                next_operation = operation + 1
                if next_operation < self.job[job].nop:
                    node.wait.append((job, next_operation))
                node.value = np.append(node.value, job)
                self.node_list[0] = node
                if len(node.wait) == 0:
                    self.node_list_complete.append(node)
                    self.node_list.pop(0)
