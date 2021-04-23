from src import *

file_dir = "./GA_LWJSP/ft06"
file = "1-GanttChart.csv"
file_save = "%s/GanttChart/%s" % (file_dir, file[:-4])
Utils.make_dir("%s/GanttChart" % file_dir)
a = GanttChartFromCsv("%s/%s" % (file_dir, file))
a.schedule.time_unit = 1
a.ganttChart_png(fig_width=9, fig_height=5, filename=file_save, random_colors=False, lang=1, dpi=200,
                 height=0.8, scale_more=12, x_step=a.schedule.makespan // 10, y_based=0, text_rotation=0,
                 with_operation=False, with_start_end=False, show=False)
a.ganttChart_html(date="2020 7 6", filename=file_save, show=False, lang=1)
