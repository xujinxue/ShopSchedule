__doc__ = """
绘制甘特图
"""

from src import *

file_dir = "./NSGA_JSP/ft06"  # 甘特图数据文件所在目录
file = "1-GanttChart.csv"  # 甘特图数据文件
file_save = "%s/GanttChart/%s" % (file_dir, file[:-4])  # 保存的甘特图名称
Utils.make_dir("%s/GanttChart" % file_dir)  # 生成的甘特图保存至该目录
a = GanttChartFromCsv("%s/%s" % (file_dir, file))  # 调用甘特图生成类
a.schedule.time_unit = 1  # 设置加工时间单位
a.ganttChart_png(fig_width=9, fig_height=5, filename=file_save, random_colors=False, lang=1, dpi=200,
                 height=0.8, scale_more=12, x_step=a.schedule.makespan // 10, y_based=0, text_rotation=0,
                 with_operation=False, with_start_end=False, show=False)  # 绘制png格式的甘特图
a.ganttChart_html(date="2020 7 6", filename=file_save, show=False, lang=1)  # 绘制html格式的甘特图
