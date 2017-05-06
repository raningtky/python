 __author__ = 'lium'



#主要看下里面函数用法就行，我两点有事，就先不注释了。。。另外可以去官网上看例子



import numpy as np
from sklearn.svm import SVR,SVC
from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import *
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as pl
from sklearn.linear_model import LogisticRegressionCV
from sklearn import linear_model
import pandas as pd


def center(X_lst):
    X_center = []
    for k in range(len(X_lst[0])):
        sum = 0
        num = 0
        for X in X_lst:
            sum += X[k]
            num += 1
        X_center.append(sum/num)
    return X_center

def distance(X_1, X_2):
    sum = 0
    num = 0
    for k in range(0, len(X_1)):
        sum += (X_1[k]-X_2[k])**2
        num += 1
    return (sum/num)**0.5


def classify(X_lst, Y_lst, center1, center2):
    X_1 = []
    X_2 = []
    Y_1 = []
    Y_2 = []
    for k in range(0,len(X_lst)):
        if distance(X_lst[k],center1) < distance(X_lst[k],center2):
            X_1.append(X_lst[k])
            Y_1.append(Y_lst[k])
        else:
            X_2.append(X_lst[k])
            Y_2.append(Y_lst[k])
    return X_1, Y_1, X_2, Y_2



def read_data(filename):
    data = []
    file = open(filename,'r')
    for line in file:
        a = line.split(',')
        b = []
        for x in a:
            b.append(float(x))
        data.append(b)
    return data


def write_data(data):
    file = open('write.txt','w')
    for x in data:
        for k in x:
            file.write(str(k)+' ')
        file.write('\n')
    file.close()


def cal_rate(Y_est,Y_true,threshold):
    sum = 0
    num_right = 0
    for k in range(0,len(Y_est)):
        if abs(Y_est[k] - Y_true[k]) <= threshold:
            sum += 1
            num_right += 1
        else:
            sum += 1
    return num_right / sum

def cal_rmse(Y_est,Y_true):
    sum  = 0
    Q = 0
    for k in range(0, len(Y_est)):
        sum += 1
        Q += (Y_est[k] - Y_true[k])**2
    return (Q/sum)**0.5

def cal_V90_rate(Y_est,Y_true):
    sum = 0
    num_right = 0
    for k in range(0, len(Y_est)):
        if Y_true[k] < 1:
            if abs(Y_est[k] - Y_true[k]) <= 0.15:
                sum += 1
                num_right += 1
            else:
                sum += 1
        elif 1 <= Y_true[k] <= 2.5:
            if abs(Y_est[k] - Y_true[k]) <= 0.2:
                sum += 1
                num_right += 1
            else:
                sum += 1
        else:
            if abs(Y_est[k] - Y_true[k]) <= 0.25:
                sum += 1
                num_right += 1
            else:
                sum += 1
    return num_right/sum

def cal_V10_rate(Y_est,Y_true):
    sum = 0
    num_right = 0
    for k in range(0, len(Y_est)):
        if Y_true[k] < 2:
            if abs(Y_est[k] - Y_true[k]) <= 0.6:
                sum += 1
                num_right += 1
            else:
                sum += 1
        elif 2 <= Y_true[k] <= 3.5:
            if abs(Y_est[k] - Y_true[k]) <= 0.75:
                sum += 1
                num_right += 1
            else:
                sum += 1
        else:
            if abs(Y_est[k] - Y_true[k]) <= 0.9:
                sum += 1
                num_right += 1
            else:
                sum += 1
    return num_right/sum

def read_index(filename):
    file = open(filename,'r')
    index = []
    for line in file:
        a = line.split()[0]
        index.append(int(a))
    return index

def Generate_data(X,Y,filename):
    index = read_index(filename)
    X_data = []
    Y_data = []
    for k in index:
        X_data.append(X[int(k)-1])
        Y_data.append(Y[int(k)-1])
    return X_data,Y_data


def Generate_data_for_tsn(X,Y,filename):
    X_train = []
    Y_train = []
    X_test = []
    Y_test = []
    index = read_index(filename)
    for k in range(0,len(X)):
        if k in index:
            X_test.append(X[k])
            Y_test.append(int(Y[k][0]))
        else:
            X_train.append(X[k])
            Y_train.append(int(Y[k][0]))
    return X_train,Y_train,X_test,Y_test


def Generate_ising(X):
    X_ising = []
    for x in X:
        x_ising = []
        for k in x:
            if k <= 0.1:
                x_ising.append(0)
            else:
                x_ising.append(1)
        X_ising.append(x_ising)
    return X_ising

def Generate_y(Y,i):
    y = []
    for k in Y:
        y.append(k[i])
    return y

def exam_result(estimator,X,Y):
    Y_est = []
    for x in X:
        Y_est.append(estimator.predict(x))
    return cal_rmse(Y_est,Y),cal_V10_rate(Y_est,Y)

def insert(X, Y):
    for k in range(0, len(X)):
        X[k].append(Y[k])
    return X

def mean(Y):
    S = 0
    num = 0
    for k in Y:
        S += k
        num += 1
    return S/num


def enlarge(Y,a):
    Y_ = []
    mean_Y = mean(Y)
    for k in Y:
        k_ = k + a*(k-mean_Y)**3
        Y_.append(k_)
    return Y_

def Generate_error(Y_output,Y_target):
    Y_error = []
    for i in range(0, len(Y_output)):
        Y_error.append(Y_target[i]-Y_output[i])
    return Y_error

def adjust(Y_output,Y_error):
    Y_ = []
    for i in range(0, len(Y_output)):
        Y_.append(Y_output[i]+Y_error[i])
    return Y_


def Generate_X_for_error(Y_pred):
    X = []
    for k in Y_pred:
        X.append([k])
    return X

def Generate_01(Y_error,Y,threshold,bound):
    Y_ = []
    for k in range(0,len(Y_error)):
        if -threshold <= Y_error[k] <= threshold and Y[k]<bound:
            Y_.append(0)
        else:
            Y_.append(1)
    return Y_

def Generate_class(X,Y,Y_class):
    X_0 = []
    X_1 = []
    Y_0 = []
    Y_1 = []
    for k in range(0,len(Y_class)):
        if Y_class[k] == 0:
            X_0.append(X[k])
            Y_0.append(Y[k])
        else:
            X_1.append(X[k])
            Y_1.append(Y[k])
    return X_0,X_1,Y_0,Y_1

##X_all = read_data('X_data1.txt')
##Y_all = read_data('Y_data1.txt')

##X_ising =Generate_ising(X_all)
##write_data(X_ising)


X_train_1 = read_data('A10_train_1.txt')
Y_train_10 = read_data('B10_train_1.txt')
X_train_2 = read_data('A10_train_2.txt')
Y_train_20 = read_data('B10_train_2.txt')
X_test = read_data('A10_test.txt')
Y_test0 = read_data('B10_test.txt')

Y_train_1 = Generate_y(Y_train_10, 0)
Y_train_2 = Generate_y(Y_train_20, 0)
Y_test = Generate_y(Y_test0, 0)

X_1 = center(X_train_1)
X_2 = center(X_train_2)

X_test_1, Y_test_1, X_test_2, Y_test_2 = classify(X_test, Y_test, X_1, X_2)


svr11 = SVR()
svr_poly_1 = GridSearchCV(cv=5,estimator=svr11,param_grid={'kernel':['poly'],'degree':[1,2],'C':[1,3,6,9,15,20],'epsilon':[0,0.0001]})
svr12 = SVR()
svr_rbf_1 = GridSearchCV(cv=5,estimator=svr12,param_grid={'kernel':['rbf'],'gamma':[0.0005,0.001,0.01,0.1,1,10],'C':[1,3,6,9,20],'epsilon':[0,0.0001]})
svr21 = SVR()
svr_poly_2 = GridSearchCV(cv=5,estimator=svr11,param_grid={'kernel':['poly'],'degree':[1,2],'C':[1,3,6,9,15,20],'epsilon':[0,0.0001]})
svr22 = SVR()
svr_rbf_2 = GridSearchCV(cv=5,estimator=svr12,param_grid={'kernel':['rbf'],'gamma':[0.0005,0.001,0.01,0.1,1,10],'C':[1,3,6,9,20],'epsilon':[0,0.0001]})


svr_poly_1.fit(X_train_1,Y_train_1)
svr_rbf_1.fit(X_train_1,Y_train_1)
svr_poly_2.fit(X_train_2,Y_train_2)
svr_rbf_2.fit(X_train_2,Y_train_2)

rmse_11, rate_11 = exam_result(svr_poly_1,X_test_1,Y_test_1)
rmse_12, rate_12 = exam_result(svr_rbf_1,X_test_1,Y_test_1)

rmse_21, rate_21 = exam_result(svr_poly_2,X_test_2,Y_test_2)
rmse_22, rate_22 = exam_result(svr_rbf_2,X_test_2,Y_test_2)

rate_poly = (rate_11*len(Y_test_1) + rate_21*len(Y_test_2))/(len(Y_test))
rate_rbf = (rate_12*len(Y_test_1) + rate_22*len(Y_test_2))/(len(Y_test))
print(rate_poly,rate_rbf)

##write_data(Y_test)


##Y_v10_train = Generate_y(Y_train,-1)
##Y_v10_test = Generate_y(Y_test, -1)

##Y_eta_train = Generate_y(Y_train,-3)
##Y_eta_test = Generate_y(Y_test,-3)

##v10_pred = SVR()
##v10_model = GridSearchCV(cv=6,estimator=v10_pred,param_grid={'kernel':['poly'],'degree':[1,2],'C':[1,3,5,7,9,15,20],'epsilon':[0,0.0001]})
##v10_model.fit(X_train,Y_v10_train)
##v10_predict_train = v10_model.predict(X_train)
##v10_predict_test = v10_model.predict(X_test)

##X_v10_train = insert(X_train, Y_v10_train)
##X_v10_test = insert(X_test, Y_v10_test)
##print(v10_predict_test)
##svr = SVR()
##v10_estimator = GridSearchCV(cv=6,estimator=svr,param_grid={'kernel':['rbf'],'gamma':[0.0001,0.0005,0.001,0.005,0.01,0.05,0.1,0.5,1],'C':[0.1,1,3,5,7,9,15,20],'epsilon':[0,0.0001]})
##v10_estimator.fit(X_v10_train,Y_eta_train)


##print(exam_result(v10_estimator,X_v10_test,Y_eta_test))






##Y_eta_train = Generate_y(Y_train,-3)
##Y_eta_test = Generate_y(Y_test,-3)

##X_pca_train = read_data('X_pca.txt')
##X_pca_test = read_data('X_pca_test.txt')
##Y_pca_train = read_data('Y_pca_train.txt')
##Y_pca_test = read_data('Y_pca_test.txt')
##Y_1_train = Generate_y(Y_pca_train,0)
##Y_1_test = Generate_y(Y_pca_test,0)
##X_for_tsn_all = read_data('X_for_tsn.txt')
##Y_tsn = read_data('tsn.txt')

##X_tsn_train,Y_tsn_train,X_tsn_test,Y_tsn_test = Generate_data_for_tsn(X_for_tsn_all,Y_tsn,'testtsn.txt')
##write_data(Y_tsn_test)
##print(len(X_tsn_train))
##print(len(Y_tsn_test))
##svr_estimator = SVR()
##svr = GridSearchCV(cv=5,estimator=svr_estimator,param_grid={'kernel':['poly'],'degree':[1,2],'C':[1,3,5,7,9,15,20,30],'epsilon':[0,0.0001]})
##svr.fit(X_train,Y_eta_train)


##rint(exam_result(svr,X_test,Y_eta_test))
##print(exam_result(svr,X_train,Y_eta_train))




##X_predict_train = Generate_X_for_error(svr.predict(X_train))
##Y_error_train = Generate_error(svr.predict(X_train),Y_eta_train)

##Y_class = Generate_01(Y_error_train,Y_eta_train,1.5,35)

##clf = KNeighborsClassifier()
##clf_eta = GridSearchCV(cv=5,estimator=clf,param_grid={'n_neighbors':[1,3,5,7,9,11,13,15]})
##clf_eta.fit(X_train,Y_class)
##class_train = clf_eta.predict(X_train)
##print(cal_rmse(class_train,Y_class))
##X_0_train,X_1_train,Y_0_train,Y_1_train =  Generate_class(X_train,Y_eta_train,Y_class)
##class_test = clf_eta.predict(X_test)
##X_0_test,X_1_test,Y_0_test,Y_1_test =  Generate_class(X_test,Y_eta_test,class_test)

##svr0_es = SVR()
##svr1_es = SVR()
##svr0 = GridSearchCV(cv=5,estimator=svr0_es,param_grid={'kernel':['poly'],'degree':[1,2],'C':[1,3,5,7,9,15,20,30],'epsilon':[0,0.0001]})
##svr1 = GridSearchCV(cv=5,estimator=svr1_es,param_grid={'kernel':['poly'],'degree':[1,2],'C':[1,3,5,7,9,15,20,30],'epsilon':[0,0.0001]})
##svr0.fit(X_0_train,Y_0_train)
##svr1.fit(X_1_train,Y_1_train)

##print(exam_result(svr0,X_0_test,Y_0_test))
##print(exam_result(svr1,X_1_test,Y_1_test))

##print(len(Y_0_test))
##print(len(Y_1_test))



##svr_ = SVR()
##svr_error = GridSearchCV(cv=5,estimator=svr_,param_grid={'kernel':['poly'],'degree':[3],'C':[0.001,0.01,0.1,1],'epsilon':[0,0.0001]})
##svr_error.fit(X_predict_train,Y_error_train)


##pl.plot(Y_eta_train,Y_error_train,'o')
##pl.show()


##Y_predict = svr.predict(X_test)
##X_predict_test = Generate_X_for_error(Y_predict)
##Y_test_adjust = adjust(Y_predict,svr_error.predict(X_predict_test))

##print(Y_test_adjust)
##sum  = 0
##Q = 0
##for k in range(0, len(Y_test_adjust)):
 ##   sum += 1
 ##   Q += (Y_test_adjust[k] - Y_eta_test[k])**2
##print((Q/sum)**0.5)

##sum = 0
##num_right = 0
##for k in range(0,len(Y_eta_test)):
 ##   if abs(Y_test_adjust[k] - Y_eta_test[k]) <= 5:
 ##       sum += 1
 ##       num_right += 1
 ##   else:
 ##       sum += 1
##print(num_right / sum)




##write_data(svr.predict(X_train))

##log_clf = LogisticRegressionCV(cv=5,penalty='l2',max_iter=1000,Cs=10)
##log_clf.fit(X_tsn_train,Y_tsn_train)
##print(exam_result(log_clf,X_tsn_test,Y_tsn_test))
##svc_class = SVC()
##svc = GridSearchCV(cv=6,estimator=svc_class,param_grid={'kernel':['poly'],'degree':[1,2,3],'C':[1,3,5,7,9,10,15,20,30,50]})
##svc.fit(X_tsn_train,Y_tsn_train)
##print(exam_result(svc,X_tsn_test,Y_tsn_test))
##print(exam_result(svc,X_tsn_train,Y_tsn_train))
##clf = KNeighborsClassifier()
##clf_optimal = GridSearchCV(cv=5,estimator=clf,param_grid={'n_neighbors':[1,3,5,7,9,11,13,15,17,19,21]})
##clf_optimal.fit(X_tsn_train,Y_tsn_train)
##print(exam_result(clf_optimal,X_tsn_test,Y_tsn_test))
##C_lst =[]
##gamma_lst = []
##rate_lst = []
##rmse_lst = []
##for i in range(0,10):
 ##   C_lst.append(0.04*i**3.5 +0.001)
   ## rate_lst1 = []
   ## rmse_lst1 = []
   ## for j in range(0,10):
   ##     gamma_lst.append(j**3 * 0.001+0.0001)
   ##     svr_np = SVC()
   ##     svr_exp = GridSearchCV(cv=5,estimator=svr_np,param_grid={'kernel':['rbf'],'gamma':[j**5 * 0.00001+0.00001],'C':[0.001*i**3.5 +0.001]})
   ##     svr_exp.fit(X_tsn_train,Y_tsn_train)
   ##     rmse,rate = exam_result(svr_exp,X_tsn_test,Y_tsn_test)
   ##     rate = int(10*rate) * 0.1
   ##     rate_lst1.append(rate)
   ##     rmse_lst1.append(rmse[0])
  ##  rate_lst.append(rate_lst1)
  ##  rmse_lst.append(rmse_lst1)
##print(rate_lst)
##print(rmse_lst)

##ate_heat = np.array(rate_lst)
##rmse_heat = np.array(rate_lst)
##print(rate_heat)

##X_ray = np.array(X_ising)
##fig, ax = pl.subplots()
##heatmap = ax.pcolor(X_ray, cmap=pl.cm.Blues, alpha=0.3)
##pl.show()

