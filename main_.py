import sys
from PyQt5.QtWidgets import QFileDialog, QApplication, QFileSystemModel, QMainWindow
from PyQt5.QtCore import QAbstractTableModel, Qt, QModelIndex
from PyQt5 import uic
from PyQt5.uic.properties import QtWidgets
from PyQt5.QtGui import QDoubleValidator
import os
import shutil
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Qt5Agg')
import seaborn as sns
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import pathlib
import scipy.stats as stats
from sklearn import preprocessing
from sklearn.metrics import (confusion_matrix, precision_score, 
                            recall_score, f1_score, silhouette_score,
                            davies_bouldin_score, calinski_harabasz_score)
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.cluster import KMeans, DBSCAN
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
import sip

pd.set_option('mode.chained_assignment', None)

PATH = os.path.dirname(os.path.abspath(__file__))


class MyApp(QMainWindow):
    '''Janela pricipal, para abrir, criar e inserir arquivos em um projeto '''

    # Método construtor
    def __init__(self):
        super().__init__()
        
        uic.loadUi('app.ui', self)
        self.createProject = CreateProject()
        self.addFiles = AddFiles()

        self.pushButton_project.clicked.connect(self.makeProject)
        self.pushButton_open.clicked.connect(self.openProject)
        self.pushButton_files.clicked.connect(self.selectFiles)

    #Abre nova janela para criar novo projeto
    def makeProject(self):
        self.createProject.show()
    
    #Abre nova janela para abrir projeto
    def openProject(self):
        self.open_project = QFileDialog.getExistingDirectory()

        txtfiles = []
        for file in pathlib.Path(self.open_project).glob("*.csv"):
            txtfiles.append(file)

        self.table_view = ViewTables(txtfiles)
        self.table_view.show()
    
    #Abre nova janela para adicionar arquivos no projeto
    def selectFiles(self):
        self.addFiles.show()
  

class CreateProject(QMainWindow):
    '''Cria novos projetos'''

    # Método construtor
    def __init__(self):
        super().__init__()
        self.main_path = PATH

        uic.loadUi('create_project.ui', self)

        self.pushButton_main_path.clicked.connect(self.selectMainPath)
        self.pushButton_new_project.clicked.connect(self.createProject)
        
        self.lineEdit_main_path.setText(self.main_path) 

    #Seleciona pasta principal do projeto
    def selectMainPath(self):
        self.main_path = QFileDialog.getExistingDirectory()
        self.lineEdit_main_path.setText(self.main_path) 

    #Cria a pasta do novo projeto
    def createProject(self):
        project_name = self.lineEdit_project.text()
        if project_name != "":
            os.mkdir(os.path.join(self.main_path, project_name))
            self.close()


class AddFiles(QMainWindow):
    '''Adiciona novos arquivos a um projeto'''

    # Método construtor
    def __init__(self):
        super().__init__()
        self.main_path = PATH

        uic.loadUi('add_files.ui', self)

        self.model = QFileSystemModel()

        self.pushButton_project_path.clicked.connect(self.selectProject)
        self.pushButton_add_files.clicked.connect(self.addFiles)
        self.lineEdit_project_path.setText(self.main_path) 

    #Seleciona pasta do projeto
    def selectProject(self):
        self.main_path = QFileDialog.getExistingDirectory()
        self.lineEdit_project_path.setText(self.main_path) 
        self.updateFiles()

    #Visualização dos arquivos do projeto
    def updateFiles(self):
        if self.main_path != "":
            self.model.setRootPath(self.main_path)
            self.treeView_project.setModel(self.model)
            self.treeView_project.setRootIndex(self.model.index(self.main_path))
            self.treeView_project.setColumnWidth(0,200)
            self.treeView_project.setAlternatingRowColors(True)
 
    #Adiciona novos arquivos no projeto
    def addFiles(self):
        self.files_path = QFileDialog.getOpenFileNames(self, 'Select csv File', "", "CSV files (*.csv)")[0]
        if self.files_path != []:

            for file in self.files_path:
                head, tail = os.path.split(file)
                join_main_path = os.path.join(self.main_path, tail)
                new_tail = tail.split(".")

                if os.path.isfile(join_main_path):
                    n = 0

                    while(os.path.isfile(join_main_path)):
                        n +=1
                        join_main_path = os.path.join(self.main_path, new_tail[0]+"_"+str(n)+"."+new_tail[1])
                        if not os.path.isfile(join_main_path):
                            shutil.copy2(file, join_main_path)
                            break
                else:
                    shutil.copy2(file, join_main_path)


class AddTable(QMainWindow):
    '''Cria uma nova tabela no projeto selecionado'''

    # Método construtor
    def __init__(self, txtfiles):
        super().__init__()
     
        uic.loadUi('new_table.ui', self)
        self.txtfiles = txtfiles
        self.path_parent = self.txtfiles[0].parent
        self.pushButton_save.clicked.connect(self.saveTable)
        
        for file in self.txtfiles:
            df = pd.read_csv(file, sep=';')
            for cl_name in  df.columns:
                self.listWidget.addItem(file.name + " : " + cl_name)

    #Salva a tabela
    def saveTable(self):
        self.new_name = self.lineEdit_name.text()+".csv"
        files_select = dict()

        for item in self.listWidget.selectedItems():
            file_column = item.text().split(" : ")
            file, column = file_column[0], file_column[1]

            if files_select.get(file):
                files_select[file].append(column)
            else:
                files_select[file] = [column]

        list_data = []

        for key, values in files_select.items():
            data = pd.read_csv(os.path.join(self.path_parent, key), sep=';')
            list_data.append(data[values])
        
        new_dataFrame = pd.concat(list_data, axis=1)
        new_dataFrame.to_csv(os.path.join(self.path_parent, self.new_name), sep=';', index=False)
        self.close()
    

class OptionsMachine(QMainWindow):
    '''Opções para alplicar algoritmos de aprendizagem de máquina
    no arquivo selecionado do projeto'''

    # Método construtor
    def __init__(self, file):
        super().__init__()
        uic.loadUi('machine.ui', self)
        self.pushButton_Kmeans.clicked.connect(self.execKmeans)
        self.pushButton_DBScan.clicked.connect(self.execDBScan)
        self.pushButton_Tree.clicked.connect(self.execTree)
        self.pushButton_Bayes.clicked.connect(self.execBayes)

        self.path_parent = file.parent
        self.file_name = file.stem
 
        self.df = pd.read_csv(file, sep=';')
        for cl_name in  self.df.columns:
            self.listWidget_x.addItem(cl_name)
            self.listWidget_y.addItem(cl_name)

        self.plotMachine = ViewMachine()

    #Selecionar entradas  X e Y
    def selectInputOutput(self, group=False):
        x_name = []

        for x in self.listWidget_x.selectedItems():
            x_name.append(x.text())

        if group:
            y = None
        else:
            y_name = self.listWidget_y.selectedItems()[0].text()
            y = self.df[[y_name]]
        x = self.df[x_name]

        return x, y
        
    #Aplicação do K-medias
    def execKmeans(self):
        X, Y = self.selectInputOutput(group=True)
        n_clusters = self.spinBox_Kmeans.value()
        kmeans = KMeans(n_clusters=n_clusters, random_state=0)
        result = kmeans.fit_predict(X)
        X["y_result"] = result

        X.to_csv(os.path.join(self.path_parent, self.file_name+ "_kmeans.csv"), sep=';', index=False)
        self.plotMachine.show()
        self.plotMachine.plotCluster(result)

        silhouette = round(silhouette_score(X, result, metric='sqeuclidean'),2)
        cohesion = round(davies_bouldin_score(X, result),2)
        acop = round(calinski_harabasz_score(X, result),2)
        self.lineEdit_silhouette.setText(str(silhouette)) 
        self.lineEdit_cohesion.setText(str(cohesion)) 
        self.lineEdit_acop.setText(str(acop)) 

    #Aplicação de DBScan
    def execDBScan(self):
        X, Y = self.selectInputOutput(group=True)
        esp = self.doubleSpinBox_BBScam_distance.value()
        min_samples = self.spinBox_BDScan_samples.value()
        clustering = DBSCAN(eps=esp, min_samples=min_samples)
        result = clustering.fit_predict(X)
        X["y_result"] = result

        X.to_csv(os.path.join(self.path_parent, self.file_name+ "_dbscan.csv"), sep=';', index=False)
        self.plotMachine.show()
        self.plotMachine.plotCluster(result)
        
        silhouette = round(silhouette_score(X, result, metric='sqeuclidean'),2)
        cohesion = round(davies_bouldin_score(X, result),2)
        acop = round(calinski_harabasz_score(X, result),2)
        self.lineEdit_silhouette.setText(str(silhouette)) 
        self.lineEdit_cohesion.setText(str(cohesion)) 
        self.lineEdit_acop.setText(str(acop)) 



    #Aplicação de Árvore de Decisão
    def execTree(self):
        X, Y = self.selectInputOutput()
        max_depth = self.spinBox_Tree.value()
        clf = DecisionTreeClassifier(max_depth=max_depth, random_state=0)
        Y = Y.astype('int').values
        result = clf.fit(X, Y).predict(X.values)
        X["y"] = Y
        X["y_result"] = result

        X.to_csv(os.path.join(self.path_parent, self.file_name+ "_tree.csv"), sep=';', index=False)

        self.plotMachine.show()
        self.plotMachine.plotTree(clf)

        Y = np.ravel(Y)
        precision = round(precision_score(Y, result, average='macro'), 2)
        recall = round(recall_score(Y, result, average='macro'),2)
        f1 = round(f1_score(Y, result, average='macro'), 2)
        
        self.lineEdit_precision.setText(str(precision)) 
        self.lineEdit_recall.setText(str(recall)) 
        self.lineEdit_f1.setText(str(f1)) 

    #Aplicação do Naive Bayes
    def execBayes(self):
        X, Y = self.selectInputOutput()
        clf = GaussianNB()
        Y = Y.astype('int').values
        result = clf.fit(X, Y).predict(X.values)
        X["y"] = Y
        X["y_result"] = result

        X.to_csv(os.path.join(self.path_parent, self.file_name+ "_bayes.csv"), sep=';', index=False)
        self.plotMachine.show()
        self.plotMachine.plotConfution(Y, result)

        Y = np.ravel(Y)
        precision = round(precision_score(Y, result, average='macro', zero_division=0), 2)
        recall = round(recall_score(Y, result, average='macro', zero_division=0),2)
        f1 = round(f1_score(Y, result, average='macro', zero_division=0), 2)

        self.lineEdit_precision.setText(str(precision)) 
        self.lineEdit_recall.setText(str(recall)) 
        self.lineEdit_f1.setText(str(f1)) 


class Canvas(FigureCanvas):
    '''Cria uma figura dentro de uma janela'''

    def __init__(self, parent=None, width=4, height=3, dpi=100):
        fig, self.ax = plt.subplots(figsize=(width, height), dpi=dpi)
        super().__init__(fig)
        fig.tight_layout()


class ViewMachine(QMainWindow):
    '''Visualição dos resultados ao aplicar algoritmos de aprendizagem
    de máquina'''

    # Método construtor
    def __init__(self):
        super().__init__()
        uic.loadUi('view_machine.ui', self)

    #Coloca um novo gráfico na janela
    def updatePlot(self):
        try:
            self.verticalLayout.removeWidget(self.chart)
            sip.delete(self.chart)
        except:
            pass

        self.chart = Canvas()
        self.verticalLayout.addWidget(self.chart)

    #Visualização da Árvore de Decisão
    def plotTree(self, clf):
        self.updatePlot()
        self.chart.ax.cla()
        ax = self.chart.ax
        ax = tree.plot_tree(clf)

    #Visualização da Matriz de Confusão
    def plotConfution(self, y_pred, y_true):
        self.updatePlot()
        self.chart.ax.cla()
        ax = self.chart.ax
        cm = confusion_matrix(y_true, y_pred)
        ax = sns.heatmap(cm, annot=True)

    #Visualização do agrupamento
    def plotCluster(self, class_data):
        self.updatePlot()
        self.chart.ax.cla()
        ax = self.chart.ax
        ax = sns.countplot(x=class_data)
        

class ViewPlots(QMainWindow):
    '''Permite a visualização de gráficos'''

    # Método construtor
    def __init__(self, pathfile):
        super().__init__()
        uic.loadUi('views.ui', self)
        self.pathfile = pathfile
       
        self.pushButton_histo.clicked.connect(self.plotHisto)
        self.pushButton_points.clicked.connect(self.plotPoints)
        self.pushButton_dp.clicked.connect(self.plotDp)
        self.pushButton_confution.clicked.connect(self.plotConfution)

        self.path_parent = self.pathfile.parent
        self.file_name = self.pathfile.stem
 
        self.df = pd.read_csv(self.pathfile, sep=';')

        for cl_name in  self.df.columns:
            self.comboBox_data.addItem(cl_name)
            self.comboBox_y.addItem(cl_name)
            self.comboBox_x.addItem(cl_name)
            self.comboBox_yTrue.addItem(cl_name)
            self.comboBox_yPred.addItem(cl_name)

    #Coloca um novo gráfico na janela
    def updatePlot(self):

        try:
            self.verticalLayout.removeWidget(self.chart)
            sip.delete(self.chart)
        except:
            pass

        self.chart = Canvas(self)
        self.verticalLayout.addWidget(self.chart)

    #Visualização da distribuição
    def plotDp(self):
        x_column = str(self.comboBox_x.currentText())
        X = np.ravel(self.df[[x_column]].values)

        y_column = str(self.comboBox_y.currentText())
        Y = np.ravel(self.df[[y_column]].values)

        self.updatePlot()

        self.chart.ax.cla()
        ax = self.chart.ax
        ax.scatter(X, Y)
        ax.grid()

    #Visualização do Histograma
    def plotHisto(self):
        data_histo = str(self.comboBox_data.currentText())
        data_histo = self.df[[data_histo]].values

        self.updatePlot()

        self.chart.ax.cla()
        ax = self.chart.ax
        ax.hist(data_histo)
        ax.grid()

    #Gráfico de Pontos
    def plotPoints(self):
        x_column = str(self.comboBox_data.currentText())
        x = self.df[[x_column]].values

        bins = np.arange(0, max(x) + 1, 0.5)
        hist, edges = np.histogram(x, bins=bins)

        y = np.arange(1, hist.max() + 1)
        x = np.arange(0, max(x) + 0.5, 0.5)
        X,Y = np.meshgrid(x,y)

        self.updatePlot()

        self.chart.ax.cla()
        ax = self.chart.ax
        ax.scatter(X, Y, c = Y<=hist, cmap="Blues")
        ax.set_xticks(np.arange(max(x) + 2))
        ax.set_yticks([])

    #Matriz  de Confusão
    def plotConfution(self):
        y_pred = str(self.comboBox_yTrue.currentText())
        y_pred = np.ravel(self.df[[y_pred]].values)

        y_true = str(self.comboBox_yPred.currentText())
        y_true = np.ravel(self.df[[y_true]].values)

        self.updatePlot()

        self.chart.ax.cla()
        ax = self.chart.ax
        cm = confusion_matrix(y_true, y_pred)
        ax = sns.heatmap(cm, annot=True)


class ViewTables(QMainWindow):
    '''Visualização das tabelas'''

    # Método construtor
    def __init__(self, txtfiles):
        super().__init__()
        uic.loadUi('view_tables.ui', self)
        self.txtfiles = txtfiles

        self.addTableView = AddTable(self.txtfiles)

        for n, file in enumerate(self.txtfiles):
            self.comboBox_table.addItem(file.name)
 
        self.addTableView.pushButton_save.clicked.connect(self.update_list)
        
        self.comboBox_table.currentIndexChanged.connect(self.changeTable)

        self.pushButton_save.clicked.connect(self.save_csv)
        self.pushButton_deleteColumn.clicked.connect(self.removeColumn)
        self.pushButton_deleteRow.clicked.connect(self.removeRow)
        self.pushButton_mean.clicked.connect(self.replaceMean)
        self.pushButton_replace.clicked.connect(self.replaceValue)

        self.pushButton_minMax.clicked.connect(self.replaceMinMax)
        self.pushButton_zScore.clicked.connect(self.replaceZscore)
        self.pushButton_newTable.clicked.connect(self.addTable)

        self.pushButton_step.clicked.connect(self.replaceStep)
        self.pushButton_freq.clicked.connect(self.replaceFreq)

        self.pushButton_machine.clicked.connect(self.machineOptions)

        self.pushButton_views.clicked.connect(self.viewsPlot)
        self.changeTable()

    #Visualizar gráficos
    def viewsPlot(self):
        index = self.comboBox_table.currentIndex()
        path_file = self.txtfiles[index]
        self.plotView = ViewPlots(path_file)
        self.plotView.show()

    #Aplicação de modelos de aprendizagem de máquina
    def machineOptions(self):
        index = self.comboBox_table.currentIndex()
        path_file = self.txtfiles[index]
        self.viewMachine = OptionsMachine(path_file)
        self.viewMachine.show()

    #Atualiza tabela
    def update_list(self):
        self.txtfiles.append(os.path.join(self.txtfiles[0].parent, self.addTableView.new_name))
        self.comboBox_table.addItem(self.addTableView.new_name)

    #Salva tabela
    def save_csv(self):
        index = self.comboBox_table.currentIndex()
        self.model._data.to_csv(self.txtfiles[index], sep=';', index=False)
    
    #Remover colunas da tabela
    def removeColumn(self):
        self.model.removeColumn(int(self.spinBox_column.value()))

    #Remover linhas da tabela
    def removeRow(self):
        self.model.removeRow(int(self.spinBox_row.value()))

    #Substituir valores pela média
    def replaceMean(self):
        column_idx = self.spinBox_column_replace.value()
        self.model.replaceMean(column_idx)

    #Susbtituir valores
    def replaceValue(self):
        column_idx = self.spinBox_column_replace.value()
        rpl_value_x = self.lineEdit_replace_x.text()
        rpl_value_y = self.lineEdit_replace_y.text()
        self.model.replaceValue(column_idx, rpl_value_x, rpl_value_y)
    
    #Atualizar visualização da tabela
    def changeTable(self):
        index = self.comboBox_table.currentIndex()
        df = pd.read_csv(self.txtfiles[index], sep=';')
        self.model = pandasModel(df)
        self.tableView.setModel(self.model)

    #Normalização Z-Score
    def replaceZscore(self):
        column_idx = self.spinBox_norm.value()
        self.model.replaceZscore(column_idx)
    
    #Normalização Min-Max
    def replaceMinMax(self):
        column_idx = self.spinBox_norm.value()
        self.model.replaceMinMax(column_idx)    

    #Visualizar tabela
    def addTable(self):
        self.addTableView.show()

    #Discretização por frequências
    def replaceFreq(self):
        column_idx = self.spinBox_discret.value()
        bins = self.spinBox_bins.value()
        self.model.replaceFreq(column_idx, bins)  

    #Discretização por intervalo
    def replaceStep(self):
        column_idx = self.spinBox_discret.value()
        bins = self.spinBox_bins.value()
        self.model.replaceStep(column_idx, bins)  


class pandasModel(QAbstractTableModel):
    '''Aplicação de funções nas tabelas'''

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
            return self._data.columns[col] +" ("+str(col) + ")"
        elif orientation == Qt.Vertical and role == Qt.DisplayRole:
            return self._data.index.tolist()[col]
        return None

    #Atualizarcao dos valores da tabela
    def setData(self, index, value, role):
        self._data.iloc[index.row(),index.column()] = value
        return True

    def flags(self, index):
        return Qt.ItemIsEditable | Qt.ItemIsEnabled | Qt.ItemIsSelectable

    #Remover colunas da tabela
    def removeColumn(self, position, index=QModelIndex()):
        if self.columnCount() > 0 :
            
            self.beginRemoveColumns(index, position, position)
            self._data = self._data.drop(self._data.columns[position], axis=1)
            self.endRemoveColumns()

    #Remover linhas da tabela
    def removeRow(self, position, index=QModelIndex()):
        if self.rowCount() > 0 :
            
            self.beginRemoveRows(index, position, position)
            self._data = self._data.drop([position])
            self._data = self._data.reset_index(drop=True)
            self.endRemoveRows()
    
    #Substituir valores pela média
    def replaceMean(self, column_idx):
        self.beginResetModel()
        selected_column = pd.to_numeric(self._data[self._data.columns[column_idx]], errors='coerce')
        self._data[self._data.columns[column_idx]] = selected_column.fillna(selected_column.mean()).values
        self.endResetModel()

    #Substituir valores da tabela
    def replaceValue(self, column_idx, rpl_value_x, rpl_value_y):
        self.beginResetModel()
        self._data[self._data.columns[column_idx]] = self._data[self._data.columns[column_idx]].replace(rpl_value_x, rpl_value_y)
        self.endResetModel()

    #Normalização Z-Score
    def replaceZscore(self, column_idx):
        self.beginResetModel()
        selected_column = pd.to_numeric(self._data[self._data.columns[column_idx]], errors='coerce').values
        self._data[self._data.columns[column_idx]] = stats.zscore(selected_column)
        self.endResetModel()

    #Normalização Min-Max
    def replaceMinMax(self, column_idx):
        self.beginResetModel()
        selected_column = pd.to_numeric(self._data[self._data.columns[column_idx]], errors='coerce').values
        min_max_scaler = preprocessing.MinMaxScaler()
        self._data[self._data.columns[column_idx]] = min_max_scaler.fit_transform(selected_column.reshape(-1, 1))
        self.endResetModel()

    #Discretização por frequências
    def replaceFreq(self, column_idx, bins):
        self.beginResetModel()
        selected_column = pd.to_numeric(self._data[self._data.columns[column_idx]], errors='coerce').values
        est = KBinsDiscretizer(n_bins=bins, encode='ordinal', strategy='quantile')
        self._data[self._data.columns[column_idx]] = est.fit_transform(selected_column.reshape(-1, 1))
        self.endResetModel()

    #Discretização por intervalo
    def replaceStep(self, column_idx, bins):
        self.beginResetModel()
        selected_column = pd.to_numeric(self._data[self._data.columns[column_idx]], errors='coerce').values
        est = KBinsDiscretizer(n_bins=bins, encode='ordinal', strategy='uniform')
        self._data[self._data.columns[column_idx]] = est.fit_transform(selected_column.reshape(-1, 1))
        self.endResetModel()


if __name__ == '__main__':

    app = QApplication(sys.argv)
    myApp = MyApp()
    myApp.show()

    try:
        sys.exit(app.exec_())

    except SystemExit:
        print("Finalizando o programa")
