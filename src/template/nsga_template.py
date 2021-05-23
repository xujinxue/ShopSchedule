from ..utils import Utils


def NsgaTemplate(save, instance, nsga, n_exp=10, nsga2=None, n_level=5, column=0):
    Utils.make_dir_save(save, instance, nsga2)
    all_res = ""
    for exp in range(1, n_exp + 1):
        all_res += "实验%s\n" % exp
        c = nsga.do_evolution(n_level=n_level, column=column, exp_no=exp)
        for i in range(len(c[0])):
            res = c[0][i][0]
            j = i + 1
            res.save_code_to_txt("./%s/%s/Code/e%s_%s.txt" % (save, instance, exp, j))
            res.save_gantt_chart_to_csv("./%s/%s/GanttChart/e%s_%s.csv" % (save, instance, exp, j))
        res = ""
        for i, j in enumerate(c):
            res += "帕累托等级-%s\n" % (i + 1)
            for k in j:
                res += "%s\n" % str(k)
        with open("./%s/%s/Record/e%s.txt" % (save, instance, exp), "w", encoding="utf-8") as f:
            f.writelines(res)
        all_res += "%s\n" % res
    with open("./%s/%s.txt" % (save, instance), "w", encoding="utf-8") as f:
        f.writelines(all_res)
