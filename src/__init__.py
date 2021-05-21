from .algorithm import GaJsp, GaLwJsp, GaMrJsp
from .algorithm import GaFjsp, GaMrFjsp, GaFjspNew, GaMrFjspNew
from .algorithm import NSGAJsp
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
kacem1
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
INSTANCE_LIST_LWJSP = """
ft06 ft10 ft20
la01 la02 la03 la04 la05 la06 la07 la08 la09 la10
la11 la12 la13 la14 la15 la16 la17 la19 la19 la20
"""
INSTANCE_LIST_MRJSP = """
example
"""
INSTANCE_LIST_MRFJSP = """
example
"""
N_EXP = 10
