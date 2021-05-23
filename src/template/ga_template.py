from ..utils import Utils


def GaTemplate(save, instance, ga, n_exp=10, ga2=None):
    Utils.make_dir_save(save, instance, ga2)
    obj_list = []
    obj_list2 = []
    for exp in range(1, n_exp + 1):
        ga.do_evolution(exp_no=exp)
        ga.best[0].save_code_to_txt("./%s/%s/Code/%s.txt" % (save, instance, exp))
        ga.best[0].save_gantt_chart_to_csv("./%s/%s/GanttChart/%s.csv" % (save, instance, exp))
        Utils.save_record_to_csv("./%s/%s/Record/%s.csv" % (save, instance, exp), ga.record)
        obj_list.append([ga.best[1], ga.record[2].index(ga.best[1]), ga.best[0].schedule.direction])
        if ga2 is not None:
            ga2.do_evolution(pop=ga.pop, exp_no="%s*" % exp)
            ga2.best[0].save_code_to_txt("./%s/%s/Code2/%s.txt" % (save, instance, exp))
            ga2.best[0].save_gantt_chart_to_csv("./%s/%s/GanttChart2/%s.csv" % (save, instance, exp))
            Utils.save_record_to_csv("./%s/%s/Record2/%s.csv" % (save, instance, exp), ga2.record)
            obj_list2.append([ga2.best[1], ga2.record[2].index(ga2.best[1]), ga2.best[0].schedule.direction])
    Utils.save_obj_to_csv("./%s/%s.csv" % (save, instance), obj_list)
    if ga2 is not None:
        Utils.save_obj_to_csv("./%s/%s-2.csv" % (save, instance), obj_list2)
