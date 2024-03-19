from PyQt5.QtWidgets import QApplication, QWidget, QGridLayout, QLabel,QTableWidget, QTableWidgetItem, QVBoxLayout, QHeaderView
from PyQt5.QtCore import QTimer
from PyQt5.QtGui import QFont

class App(QWidget):
    def __init__(self, drone_controller):
        super().__init__()
        self.drone_controller = drone_controller
        self.initUI()

    def initUI(self):
        self.layout = QVBoxLayout()
        self.table = QTableWidget(3, 10, self)  # 2 rows, 15 columns
        self.table.setFont(QFont('Arial', 16))
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.layout.addWidget(self.table)
        self.setLayout(self.layout)
        self.update_data()

        self.timer = QTimer()
        self.timer.timeout.connect(self.update_data)
        self.timer.start(50)
        
        self.resize(8000, 280) 

    def update_data(self):
        data = self.drone_controller.get_data()
        keys = list(data.keys())

        for i in range(3):
            for j in range(5):
                key = keys[i * 5 + j]
                value = data[key]
                self.table.setItem(i, j * 2, QTableWidgetItem(key.ljust(12)))
                self.table.setItem(i, j * 2 + 1, QTableWidgetItem(f"{value:+10.6f}"))