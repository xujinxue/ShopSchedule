from PyQt5.QtWidgets import QMainWindow, QMessageBox

from .define import Para
from .ui import Ui_MainWindow


class Main(QMainWindow, Ui_MainWindow):
    def __init__(self, ):
        QMainWindow.__init__(self)
        self.para = {}
        self.setupUi(self)
        self.childQMessageBox = QMessageBox()
        self.comboBoxShopType.currentIndexChanged.connect(self.changeShopType)
        self.comboBoxDataSource.currentIndexChanged.connect(self.changeDataSource)
        self.checkBoxConstrain = [self.checkBoxBasic, self.checkBoxWorkTimetable,
                                  self.checkBoxNoWait, self.checkBoxLimitedWait, ]
        for checkBox in self.checkBoxConstrain:
            checkBox.stateChanged.connect(self.changeConstrainCondition)
        self.checkBoxObjective = [self.checkBoxMakespan, self.checkBoxTotalMakespan, self.checkBoxTotalFlowTime,
                                  self.checkBoxNTardiness, self.checkBoxTardiness, self.checkBoxEarliness]
        for checkBox in self.checkBoxObjective:
            checkBox.stateChanged.connect(self.changeOptimizeObjective)
        self.do_init()

    def closeEvent(self, event):
        reply = self.childQMessageBox.question(self, '消息', '退出？')
        if reply == QMessageBox.Yes:
            event.accept()
        else:
            event.ignore()

    def do_init(self):
        self.para[Para.shop_type] = self.comboBoxShopType.itemText(0)
        self.para[Para.data_source] = self.comboBoxDataSource.itemText(0)
        self.changeConstrainCondition()
        self.changeOptimizeObjective()

    def changeShopType(self, index):
        self.para[Para.shop_type] = self.comboBoxShopType.itemText(index)

    def changeDataSource(self, index):
        self.para[Para.data_source] = self.comboBoxDataSource.itemText(index)

    def changeConstrainCondition(self):
        constrain = {}
        for checkBox in self.checkBoxConstrain:
            if self.checkBoxNoWait.checkState():
                self.checkBoxLimitedWait.setCheckState(0)
            if self.checkBoxLimitedWait.checkState():
                self.checkBoxNoWait.setCheckState(0)
            constrain[checkBox.text()] = checkBox.checkState()
        self.para[Para.constrain_condition] = constrain

    def changeOptimizeObjective(self):
        objective = {}
        for checkBox in self.checkBoxObjective:
            objective[checkBox.text()] = checkBox.checkState()
        self.para[Para.optimize_objective] = objective
