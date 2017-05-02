# -*- coding: utf8 -*-
__author__ = 'Xuesong Wang'

from matplotlib.pylab import plt
from sklearn.metrics import precision_recall_curve
def plot_PR(y_true,y_pred):
    fpr,tpr,th=precision_recall_curve(y_true,y_pred)
    plt.plot(fpr,tpr)
    plt.title('Precision recall curve')
    plt.show()