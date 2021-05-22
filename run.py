import sys

from PyQt5.QtWidgets import QApplication

from src.main import Main

if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = Main()
    screen = QApplication.desktop().screenGeometry()
    size = win.frameGeometry()
    win.move(int((screen.width() - size.width()) / 2), int((screen.height() - size.height()) / 2))
    win.show()
    sys.exit(app.exec_())
