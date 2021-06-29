from ..utils import Utils


def NsgaTemplate(save, instance, nsga, n_exp=10, nsga2=None, n_level=5, column=0):
    Utils.make_dir_save(save, instance, nsga2)
    all_res, all_res2 = "", ""
    for exp in range(1, n_exp + 1):
        cur_res = "实验%s\n" % exp
        c = nsga.do_evolution(n_level=n_level, column=column, exp_no=exp)
        for i in range(len(c[0])):
            res = c[0][i][0]
            j = i + 1
            res.save_code_to_txt("./%s/%s/Code/e%s_%s.txt" % (save, instance, exp, j))
            res.save_gantt_chart_to_csv("./%s/%s/GanttChart/e%s-%s.csv" % (save, instance, exp, j))
            res.trans_direction()
            res.save_gantt_chart_to_csv("./%s/%s/GanttChartReal/e%s-%s.csv" % (save, instance, exp, j))
        for i, j in enumerate(c):
            cur_res += "帕累托等级-%s\n" % (i + 1)
            for k in j:
                cur_res += "%s\n" % str(k)
        cur_res += "\n"
        with open("./%s/%s/Record/e%s.txt" % (save, instance, exp), "w", encoding="utf-8") as f:
            f.writelines(cur_res)
        all_res += cur_res
        if nsga2 is not None:
            cur_res2 = "实验%s\n" % exp
            c = nsga2.do_evolution(pop=nsga.pop, n_level=n_level, column=column, exp_no=exp)
            for i in range(len(c[0])):
                res = c[0][i][0]
                j = i + 1
                res.save_code_to_txt("./%s/%s/Code2/e%s_%s.txt" % (save, instance, exp, j))
                res.save_gantt_chart_to_csv("./%s/%s/GanttChart2/e%s-%s.csv" % (save, instance, exp, j))
                res.trans_direction()
                res.save_gantt_chart_to_csv("./%s/%s/GanttChartReal2/e%s-%s.csv" % (save, instance, exp, j))
            for i, j in enumerate(c):
                cur_res2 += "帕累托等级-%s\n" % (i + 1)
                for k in j:
                    cur_res2 += "%s\n" % str(k)
            cur_res2 += "\n"
            with open("./%s/%s/Record2/e%s.txt" % (save, instance, exp), "w", encoding="utf-8") as f:
                f.writelines(cur_res2)
            all_res2 += cur_res2
    with open("./%s/%s.txt" % (save, instance), "w", encoding="utf-8") as f:
        f.writelines(all_res)
    if nsga2 is not None:
        with open("./%s/%s-2.txt" % (save, instance), "w", encoding="utf-8") as f:
            f.writelines(all_res2)
