import sys
import pandas as pd
import matplotlib.pyplot as plt
from PyQt5.QtWidgets import QApplication, QTableView
from PyQt5.QtCore import QAbstractTableModel, Qt

df = pd.read_csv('tabela6896.csv', sep=';')


class pandasModel(QAbstractTableModel):

    # Método construtor
    def __init__(self, data):
        QAbstractTableModel.__init__(self)
        self._data = data

    # Retorna a quantidade de linhas do CSV
    def rowCount(self, parent=None):
        return self._data.shape[0]

    # Retorna a quantidade de colunas do CSV
    def columnCount(self, parnet=None):
        return self._data.shape[1]

    # Método responsável por inserir os dados na tabela
    def data(self, index, role=Qt.DisplayRole):
        if index.isValid():
            if role == Qt.DisplayRole:
                return str(self._data.iloc[index.row(), index.column()])
        return None

    # Método responsável pelo cabeçalho da tabela
    def headerData(self, col, orientation, role):
        if orientation == Qt.Horizontal and role == Qt.DisplayRole:
            return self._data.columns[col]
        return None


if __name__ == '__main__':
    app = QApplication(sys.argv)
    model = pandasModel(df)
    view = QTableView()
    view.setModel(model)
    view.resize(800, 600)
    view.show()
    sys.exit(app.exec_())
