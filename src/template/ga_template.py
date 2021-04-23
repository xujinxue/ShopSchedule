from ..algorithm import GaFjsp, GaFrFjsp, GaNwFjsp, GaLwFjsp, GaNwFrFjsp, GaLwFrFjsp
from ..algorithm import GaFjsp1, GaFrFjsp1, GaNwFjsp1, GaLwFjsp1, GaNwFrFjsp1, GaLwFrFjsp1
from ..algorithm import GaFspHfsp, GaFspHfspTimetable
from ..algorithm import GaJsp, GaFrJsp, GaNwJsp, GaLwJsp, GaNwFrJsp, GaLwFrJsp
from ..shop import Jsp, Fjsp, Fsp, Hfsp
from ..utils import Utils


class GaTemplate:
    def __init__(self, ga):
        self.ga = ga

    def do_exp(self, exp_log="./Result", instance="Test", n_exp=1, tabu_search=False, key_block_move=False,
               gc=False, y_based=0):
        obj_log = "%s/%s" % (exp_log, instance)
        Utils.make_dir(exp_log)
        Utils.make_dir(obj_log)
        obj_list = []
        for i in range(1, n_exp + 1):
            Utils.print("Experiment {} begin ...".format(i))
            self.ga.do_evolution(tabu_search=tabu_search, key_block_move=key_block_move)
            Utils.print("Experiment {} done ...".format(i))
            self.ga.best[0].save_gantt_chart_to_csv("%s/%s-GanttChart" % (obj_log, i))
            Utils.save_record_to_csv("%s/%s-Record" % (obj_log, i), self.ga.record)
            Utils.save_code_to_txt("%s/%s-Code" % (obj_log, i), self.ga.best[0].code)
            Utils.save_code_to_txt("%s/%s-Mac" % (obj_log, i), self.ga.best[0].mac)
            Utils.save_code_to_txt("%s/%s-Route" % (obj_log, i), self.ga.best[0].route)
            obj_list.append([self.ga.best[1], len(self.ga.record[0]), self.ga.best[0].schedule.direction])
            if gc is True:
                Utils.make_dir("%s/GanttChart" % obj_log)
                self.ga.best[0].ganttChart_png(filename="%s/GanttChart/GanttChart-%s" % (obj_log, i), y_based=y_based)
                self.ga.best[0].ganttChart_html(filename="%s/GanttChart/GanttChart-%s" % (obj_log, i))
        Utils.save_obj_to_csv("%s" % obj_log, obj_list)


class GaTemplateJsp(GaTemplate):
    def __init__(self, pop_size, rc, rm, max_generation, objective, n, m, p, tech, proc, index_template=0,
                 limited_wait=None, rest_start_end=None, resumable=None, due_date=None, best_known=None, time_unit=1):
        jsp = Utils.create_schedule(Jsp, n, m, p, tech, proc,
                                    limited_wait, rest_start_end, resumable, due_date, best_known, time_unit)
        template = [GaJsp, GaFrJsp, GaNwJsp, GaLwJsp, GaNwFrJsp, GaLwFrJsp][index_template]
        ga = template(pop_size=pop_size, rc=rc, rm=rm, max_generation=max_generation,
                      objective=objective, schedule=jsp)
        GaTemplate.__init__(self, ga)


class GaTemplateFjsp(GaTemplate):
    def __init__(self, pop_size, rc, rm, max_generation, objective, n, m, p, tech, proc, index_template=0,
                 limited_wait=None, rest_start_end=None, resumable=None, due_date=None, best_known=None, time_unit=1):
        fjsp = Utils.create_schedule(Fjsp, n, m, p, tech, proc,
                                     limited_wait, rest_start_end, resumable, due_date, best_known, time_unit)
        template = [GaFjsp1, GaFrFjsp1, GaNwFjsp1, GaLwFjsp1, GaNwFrFjsp1, GaLwFrFjsp1,
                    GaFjsp, GaFrFjsp, GaNwFjsp, GaLwFjsp, GaNwFrFjsp, GaLwFrFjsp][index_template]
        ga = template(pop_size=pop_size, rc=rc, rm=rm, max_generation=max_generation,
                      objective=objective, schedule=fjsp)
        GaTemplate.__init__(self, ga)


class GaTemplateFsp(GaTemplate):
    def __init__(self, pop_size, rc, rm, max_generation, objective, n, m, p, tech, proc,
                 limited_wait=None, rest_start_end=None, resumable=None, due_date=None, best_known=None, time_unit=1):
        fsp = Utils.create_schedule(Fsp, n, m, p, tech, proc, limited_wait, rest_start_end, resumable, due_date,
                                    best_known, time_unit)
        ga = GaFspHfsp(pop_size=pop_size, rc=rc, rm=rm, max_generation=max_generation,
                       objective=objective, schedule=fsp)
        GaTemplate.__init__(self, ga)


class GaTemplateHfsp(GaTemplate):
    def __init__(self, pop_size, rc, rm, max_generation, objective, n, m, p, tech, proc,
                 limited_wait=None, rest_start_end=None, resumable=None, due_date=None, best_known=None, time_unit=1):
        hfsp = Utils.create_schedule(Hfsp, n, m, p, tech, proc,
                                     limited_wait, rest_start_end, resumable, due_date, best_known, time_unit)
        ga = GaFspHfsp(pop_size=pop_size, rc=rc, rm=rm, max_generation=max_generation,
                       objective=objective, schedule=hfsp)
        GaTemplate.__init__(self, ga)


class GaTemplateFspTimetable(GaTemplate):
    def __init__(self, pop_size, rc, rm, max_generation, objective, n, m, p, tech, proc,
                 limited_wait=None, rest_start_end=None, resumable=None, due_date=None, best_known=None, time_unit=1):
        hfsp = Utils.create_schedule(Fsp, n, m, p, tech, proc,
                                     limited_wait, rest_start_end, resumable, due_date, best_known, time_unit)
        ga = GaFspHfspTimetable(pop_size=pop_size, rc=rc, rm=rm, max_generation=max_generation,
                                objective=objective, schedule=hfsp)
        GaTemplate.__init__(self, ga)


class GaTemplateHfspTimetable(GaTemplate):
    def __init__(self, pop_size, rc, rm, max_generation, objective, n, m, p, tech, proc,
                 limited_wait=None, rest_start_end=None, resumable=None, due_date=None, best_known=None, time_unit=1):
        hfsp = Utils.create_schedule(Hfsp, n, m, p, tech, proc,
                                     limited_wait, rest_start_end, resumable, due_date, best_known, time_unit)
        ga = GaFspHfspTimetable(pop_size=pop_size, rc=rc, rm=rm, max_generation=max_generation,
                                objective=objective, schedule=hfsp)
        GaTemplate.__init__(self, ga)
