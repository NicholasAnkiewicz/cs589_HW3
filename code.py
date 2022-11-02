import math
import numpy as np
import csv
from sklearn.tree import DecisionTreeClassifier  
from sklearn.model_selection import KFold
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import make_scorer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC




def q1():
  splits = [0.5, 1.5, 2.5, 3.5, 4.5, 5.5]
  x = [1.0, 2.0, 3.0, 4.0, 5.0]
  y = [1, 1, 0, 0, 1]
  Ip = Icalc(y)
  InfoGain = []
  for split in splits:
    left = []
    right = []
    for o in x:
      if (o > split):
        right.append(y[x.index(o)]) 
      else:
        left.append(y[x.index(o)])
    InfoGain.append(
      Ip-
      (((len(left) * Icalc(left)) + 
      (len(right) * Icalc(right)))/
      len(y)))
  return InfoGain

def Icalc(arr):
  if (len(arr)==0):
    return 0
  value = 0.0
  for p in range(0, 2):
    if (arr.count(p) == 0):
      continue
    j = arr.count(p)/len(arr)
    j = (j*-1)*(math.log(j, 10))
    value += j
  return value

stuff=np.load("./data.npz")
X_trn = stuff["X_trn"]
y_trn = stuff["y_trn"]
X_tst = stuff["X_tst"]
X_trn = X_trn.reshape(6000, 2883)
X_tst = X_tst.reshape(1200, 2883)

def q4():
  kf = KFold(shuffle=True, n_splits=5)
  depths = [1, 3, 6, 9, 12, 14]
  avg_errs = []
  for d in depths:
    errs = []
    for train_index, test_index in kf.split(X_trn):
      X_train , X_test = X_trn[train_index], X_trn[test_index]
      y_train , y_test = y_trn[train_index] , y_trn[test_index]
      clf1 = DecisionTreeClassifier(max_depth=d)
      clf1.fit(X_train, y_train)
      pred = clf1.predict(X_test)
      class_err = 1-accuracy_score(pred, y_test)
      errs.append(class_err)
    mean_err = sum(errs)/len(errs)
    avg_errs.append(mean_err)
  print(avg_errs)

def q5():
  clf = DecisionTreeClassifier(max_depth=12)
  clf.fit(X_trn, y_trn)
  y_pred = clf.predict(X_tst)
  write_csv(y_pred, 'sample_predictions.csv')

def write_csv(y_pred, filename):
    """Write a 1d numpy array to a Kaggle-compatible .csv file"""
    with open(filename, 'w') as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(['Id', 'Category'])
        for idx, y in enumerate(y_pred):
            csv_writer.writerow([idx, y])

def q7():
  kf = KFold(shuffle=True, n_splits=5)
  ks = [1, 3, 5, 7, 9, 11]
  avg_errs = []
  def custScorer(pred, y_test):
    return 1 - accuracy_score(pred, y_test)
  score = make_scorer(custScorer)
  for k in ks:
    knn = KNeighborsClassifier(n_neighbors=k, n_jobs=-1)
    err = (cross_val_score(knn, X_trn, y_trn, n_jobs = -1, cv=5, scoring=score))
    avg_errs.append(sum(err)/len(err))
  print(avg_errs)
    
def q8():
  knn = KNeighborsClassifier(n_neighbors=5, n_jobs=-1)
  knn.fit(X_trn, y_trn)
  y_pred = knn.predict(X_tst)
  write_csv(y_pred, 'knn_predictions.csv')

def q9():
  alphas = [0.0001, 0.01, 1, 10, 100]
  def softmax(x):
    return np.log(np.sum(np.exp(x)))
  def log_loser(pred, y_test):
    return (np.sum((-y_test) + softmax(pred)))/len(pred)
  log_loss_scorer = make_scorer(log_loser)
  def hinge_loser(pred, y_test):
    return (np.sum(max(0, 1+np.max(pred-y_test))))/len(pred)
  hinge_loss_scorer = make_scorer(hinge_loser)
  def loss01(pred, y_test):
    l = 0
    if y_test <= pred:
      l += 1
    return l/len(pred)
  loss01_scorer = make_scorer(loss01)
  for alp in alphas:
    log_clf = LogisticRegression(C=1/alp, n_jobs=-1)
    hinge_clf = LinearSVC(C=1/alp, loss="hinge")
    log_loss_log_clf = sum(cross_val_score(log_clf, X_trn, y_trn, n_jobs = -1, cv=5, scoring=log_loss_scorer))/5
    log_loss_hinge_clf = sum(cross_val_score(hinge_clf, X_trn, y_trn, n_jobs = -1, cv=5, scoring=log_loss_scorer))/5
    hinge_loss_hinge_clf = sum(cross_val_score(hinge_clf, X_trn, y_trn, n_jobs = -1, cv=5, scoring=hinge_loss_scorer))/5
    hinge_loss_log_clf = sum(cross_val_score(hinge_clf, X_trn, y_trn, n_jobs = -1, cv=5, scoring=hinge_loss_scorer))/5
    loss01_hinge_clf = sum(cross_val_score(hinge_clf, X_trn, y_trn, n_jobs = -1, cv=5, scoring=loss01_scorer))/5
    loss01_log_clf = sum(cross_val_score(hinge_clf, X_trn, y_trn, n_jobs = -1, cv=5, scoring=loss01_scorer))/5


def show(x):
  img = x.reshape((3,31,31)).transpose(1,2,0)
  plt.imshow(img)
  plt.axis('off')
  plt.draw()
  plt.pause(0.01)


def main():
  #print(q1())
  #q4()
  #q7()
  q9()
  
if __name__ == "__main__":
  main()