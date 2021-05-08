__doc__ = """
绘制甘特图
"""

from src import *

"""Result"""
file_dir = "./Result/GanttChart"  # 甘特图数据文件所在目录
file = "ft06.csv"  # 甘特图数据文件
file_save = "%s/%s" % (file_dir, file[:-4])  # 保存的甘特图名称
"""NSGA_JSP"""
# file_dir = "./NSGA_JSP/ft06"  # 甘特图数据文件所在目录
# file = "1-GanttChart.csv"  # 甘特图数据文件
# file_save = "%s/GanttChart/%s" % (file_dir, file[:-4])  # 保存的甘特图名称
"""===================================================================================="""
a = GanttChart("%s/%s" % (file_dir, file))  # 调用甘特图生成类
a.schedule.time_unit = 1  # 设置加工时间单位
a.gantt_chart_png(filename=file_save, fig_width=9, fig_height=5, random_colors=False, lang=1, dpi=200,
                  height=0.8, scale_more=12, x_step=a.schedule.makespan // 10, y_based=0, text_rotation=0,
                  with_operation=True, with_start_end=False, key_block=True, show=False)  # 绘制png格式的甘特图
# a.gantt_chart_html(date="2020 7 6", filename=file_save, show=False, lang=1)  # 绘制html格式的甘特图
