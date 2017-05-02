# -*- coding: utf8 -*-
__author__ = 'Xuesong Wang'
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix,accuracy_score
import plot_ROC as Plot_ROC
import plot_PR as plot_PR

def evaluation(clf,trainx,trainy,testx,testy):
    # clf.fit(trainx, trainy)
    probas_ = clf.predict_proba(testx)
    # quantative evaluation
    print("对测试集的预测结果：")
    predy = clf.predict(testx)
    print("分类准确率：%3f" % accuracy_score(testy,predy))
    # classification
    print("分类报告：")
    print(classification_report(testy, predy, target_names=['negative', 'positive']))
    print("混淆矩阵")
    print(confusion_matrix(testy, predy))
    # plot_PR.plot_PR(testy,predy)
    Plot_ROC.Plot_ROC(probas_, testy)



