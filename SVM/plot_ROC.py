# -*- coding: utf8 -*-
__author__ = 'Xuesong Wang'

from matplotlib.pylab import plt
from sklearn.metrics import precision_recall_curve, roc_curve
import seaborn
def Plot_ROC(probas_,testy):
    fpr, tpr, thresholds = roc_curve(testy, probas_[:, 1],pos_label='valid')
    plt.plot(fpr, tpr, lw=1)
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC_picture')
    plt.show()