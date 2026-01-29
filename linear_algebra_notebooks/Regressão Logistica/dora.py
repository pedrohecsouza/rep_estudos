import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns


plt.style.use('cyberpunk')

palette = ['#34EDF3', '#F715AB', '#9201CB', "#00FF59"]

class Dora:
    '''
    Algumas funções básicas para análise exploratória de dados (EDA).
    '''
    def __init__(self, data):
        self.data = data

    def summarize(self):
        return self.data.describe()
    
    def pairplot(self, hue=None):
        sns.pairplot(self.data, hue=hue)
        plt.show()
    
    def summary_statistics(self):
        stats_summary = self.data.agg(['mean', 'median', 'std', 'min', 'max'])
        return stats_summary
    
    def correlation_matrix(self):
        corr = self.data.corr()
        sns.heatmap(corr, annot=True, cmap='coolwarm')
        plt.show()
        return corr
    def plot_distribution(self, column):
        sns.histplot(self.data[column], kde=True, color=palette[0])
        plt.title(f'Distribution of {column}')
        plt.show()