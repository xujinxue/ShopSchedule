from .algorithm import GaJsp, NSGAJsp
from .algorithm import OrToolsJspSat
from .data import fjsp_benchmark, fsp_benchmark, jsp_benchmark, hfsp_benchmark
from .data import fjsp_simulation, fsp_simulation, jsp_simulation
from .data import mrfjsp_benchmark
from .data import mrjsp_benchmark
from .info import GanttChart
from .objective import Objective
from .pareto import Pareto, SelectPareto
from .resource import Code, Job, Machine, Task, TimeTable
from .shop import Jsp, Fjsp, Fsp, Hfsp
from .utils import Utils

Utils.make_dir("./Result")
Utils.make_dir("./Result/Code")
Utils.make_dir("./Result/GanttChart")

INSTANCE_LIST_JSP = """
ft06
"""
INSTANCE_LIST_FJSP = """
mk1
"""
INSTANCE_LIST_FSP = """
car1
"""
INSTANCE_LIST_HFSP = """
real1
"""
CASES_LIST = """
case1
"""
