import csv as csv
import pprint
import numpy as np
import pylab as P
from sklearn import svm
import sklearn.metrics as skm
import featureNormalize as fn
import pickle
import re
import sklearn.ensemble as en
from sklearn import tree
from scipy import stats

import matplotlib.pyplot as plt

def _process_data(isTest, t_dept):
    
    filename = 'train.csv'
    idx_age = 5
    idx_gender = 4
    idx_Pclass= 2
    idx_SibSp = 6
    idx_Parch = 7
    idx_fare = 9
    idx_name = 3
    idx_ticket = 8
    
    if(isTest):
        idx_ticket -=1
        idx_name -=1
        idx_age -=1
        idx_gender -=1
        idx_Pclass -=1
        idx_SibSp -=1
        idx_Parch -=1
        idx_fare -=1
        filename = 'test.csv'
        
    csv_obj = csv.reader(open(filename, 'r'))
    header = csv_obj.next()
    
    data=[]
    for row in csv_obj:
        data.append(row)
        
    data = np.array(data)
    
    names = data[0::, idx_name]

    titles = []
    for i in range(0,len(names)):
        titles.append(names[i].split(",")[1].split(".")[0].strip())
    
    titles = np.array(titles)
    title_features = []
    #original
    #['Sir', 'Major', 'Don', 'Mlle', 'Capt', 'Dr', 'Lady', 'Rev', 'Mrs', 'Ms', 'Mr']
    
    use_titles = ['Sir', 'Major', 'Don', 'Mlle', 'Capt', 'Dr', 'Lady', 'Rev', 'Ms', 'Mr']
#     use_titles = [t_dept]
#     print(use_titles)
#     title_not_known = np.ones(names.shape)
    for t in use_titles:
#         print(t)
        k = np.zeros(names.shape)
        k[titles == t] = 1
#         title_not_known[titles == t] = 0
#         print(np.sum(k))
#         print(k)
        title_features.append(np.array(k))
    
#     title_features.append(title_not_known)
#     print(np.array(title_features).shape)
    #access data like this
    #print(data[1:10,0::])
    
    
    
#     blank_age = data[0::,idx_age] == ''
#     data[blank_age,idx_age] = 23 #set to what most people were
#     age = data[0::,idx_age].astype(np.float)
    
    females = data[0::,idx_gender] == 'female'
    data[females, idx_gender] = 1
    males = data[0::, idx_gender] == 'male'
    data[males, idx_gender] = 0
    
    gender = data[0::,idx_gender].astype(np.float)
    
    pClass = data[0::,idx_Pclass].astype(np.float)
    
    sibsp = data[0::,idx_SibSp].astype(np.float)
    parch = data[0::,idx_Parch].astype(np.float)
    
    #n, bins, patches = P.hist(gender, 50, normed=1, histtype='stepfilled')
    #P.setp(patches, 'facecolor', 'g', 'alpha', 0.75)
    #P.show()
    
    #pClass should be 3 different features
    first_class = np.zeros(pClass.shape)
    first_class[pClass==1] = 1
    
    second_class = np.zeros(pClass.shape)
    second_class[pClass==2] = 1
    
    third_class = np.zeros(pClass.shape)
    third_class[pClass==3] = 1
    
    fare = np.zeros(pClass.shape)
    for i in range(0, len(pClass)):
        if data[i, idx_fare].strip() == '':
            data[i, idx_fare] = '0'
#         print (data[i, idx_fare])
        fare[i] = data[i, idx_fare].astype(np.float)
    
#     ticket_dept = ['S', 'P', 'C', 'A', 'W', 'F', 'L']
    ticket_dept = ['A','W']
    
    ticket_features = []
    for d in ticket_dept:
        k = np.zeros(pClass.shape)
        for i in range(0, len(data[0::, idx_ticket])):
            if data[i, idx_ticket].strip()[:1] == d:
                k[i] = 1
        ticket_features.append(np.array(k))
    
    #ticket_has_letter
#     ticket_has_letter = np.zeros(pClass.shape)
#     hl = re.compile("[a-zA-Z]")
#     for i in range(0, len(ticket_has_letter)):
#         if hl.match(data[i,8]) is not None:
#             ticket_has_letter[i] = 1
    
#     fare_populated = fare != 0
    
#     med_fare_first = np.median(data[data[fare_populated, idx_Pclass] == '1', idx_fare].astype(np.float))
#     med_fare_second = np.median(data[data[fare_populated, idx_Pclass] == '2', idx_fare].astype(np.float))
#     med_fare_third = np.median(data[data[fare_populated, idx_Pclass] == '3', idx_fare].astype(np.float))
#     
#     print(med_fare_first)
#     print(med_fare_second)
#     print(med_fare_third)
    
    for i in range(0, len(fare)):
        if fare[i] == 0:
            if(pClass[i] == 1):
                fare[i] = 25.925
            elif(pClass[i] == 2):
                fare[i] = 16.05
            elif(pClass[i] == 3):
                fare[i] = 13
    
#     em_c = np.zeros(pClass.shape)
#     em_c [data[0::, 11] == 'C'] = 1
#     # 
#     em_q = np.zeros(pClass.shape)
#     em_q [data[0::, 11] == 'Q'] = 1
#     # 
#     em_s = np.zeros(pClass.shape)
#     em_s [data[0::, 11] == 'S'] = 1
    
    fam = sibsp + parch
#     fam[fam > 0] = 1
    
#     print(fam)

#     ticket_number = data[0::, idx_ticket]
#     ticket_codes = []
#     for item in ticket_number:
#         deptName = re.sub(r"$\d+\W+|\b\d+\b|\W+\d+$", "", item)
#         if len(deptName) == 0:
#             deptName = 'none'
#         ticket_codes.append(ord(deptName[0]) + len(deptName))
#     ticket_codes = np.array(ticket_codes)
    
    features = [gender, pClass, fam, fare]
    
#     features.append(title_features[0])
#     features.append(title_features[9])

#     print(features)
#     print(len(title_features))
    for f in title_features:
        features.append(f)
     
#     print(len(ticket_features))
#     features.append(ticket_features[1])
    for f in ticket_features:
        features.append(f)
    
    X = np.zeros((data[0::,0].size, len(features)))
    for i in range(0,len(features)):
        X[0::,i] = features[i]
    
    return X, data

def _loadtrain(d):
    
    X, data = _process_data(False, d)
    
    mean, stdev = fn.getMeanAndStdVar(X)
    
    X = (X - mean) / stdev
    
    y = data[0::,1].astype(np.int)
    
    return X,y,mean,stdev

def _loadtest(mean, stdev):
    X, data = _process_data(True, '')
    X = (X - mean) / stdev
    return X,data

def _test_model(X,y):
    m = len(X)
    f1_scores = []
    f1_scores_train = []
    scores = []
    
    perfC = []
    perfScores = []
    perfF1Train = []
    
    f1_scores_all = []
    
    for C in np.linspace(0.01, 5, num=25):
        f1_scores = []
        scores = []
        f1_scores_train = []
        #print(C)
        for i in range(0,500):
            th = m*0.75 #train on 75% of data, test on 25%
            rand_perm = np.random.permutation(m)
            X_train = X[rand_perm[0:th],0::]
            y_train = y[rand_perm[0:th]]
            
            X_val = X[rand_perm[th:m],0::]
            y_val = y[rand_perm[th:m],0::]
            
            #split X and y into X_train and X_val, y_train and y_val
            
            clf = svm.SVC(C=2.04, gamma=0.122)
            
#             clf = tree.DecisionTreeClassifier()
#             clf = svm.LinearSVC(C=C, dual=False)
#             clf = en.GradientBoostingClassifier(n_estimators=100, learning_rate=1, max_depth=None, random_state=0)
#             clf = en.ExtraTreesClassifier(n_estimators=10, max_features=5, max_depth=None, min_samples_split=1, random_state=0)
#             clf = en.RandomForestClassifier(n_estimators=10, max_features=3, max_depth=None, min_samples_split=1)
            clf.fit(X_train,y_train)
            
            y_predict = clf.predict(X_val)
            
            f1_scores.append(skm.f1_score(y_val, y_predict))
            f1_scores_all.append(skm.f1_score(y_val, y_predict))
            f1_scores_train.append(skm.f1_score(y_train, clf.predict(X_train)))
            
            #print(clf.score(X_val,y_val))
            scores.append(clf.score(X_val, y_val))
            
        perfC.append([np.mean(np.array(f1_scores)), C])
        perfScores.append([np.mean(np.array(scores)), C])
        perfF1Train.append([np.mean(np.array(f1_scores_train)), C])
        
    
    perfC = np.array(perfC)
    perfScores = np.array(perfScores)
    perfF1Train = np.array(perfF1Train)
    
   
    
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    l1,l2,l3 = ax.plot(perfC[0::,1], perfC[0::,0], 'r', perfScores[0::,1], perfScores[0::,0], 'b', perfF1Train[0::,1], perfF1Train[0::,0], 'g')
    
    fig.legend((l1,l2,l3), ('F1_CrossVal','Accuracy_CrossVal','F1_Train'), 'upper left')
    
    plt.show()
    
    
    print(perfC[perfC[:,0].argsort()])
    print(perfScores[perfScores[:,0].argsort()])
    print(perfF1Train[perfF1Train[:,0].argsort()])
    
    result = perfF1Train
    result[0::, 0] = np.abs(np.subtract(perfF1Train[0::, 0], perfC[0::,0]))
    
    print(result[result[:,0].argsort()])
    
    print(np.std(f1_scores_all))
    
if __name__ == '__main__':
    
#     for d in ['Sir', 'Major', 'Don', 'Mlle', 'Capt', 'Dr', 'Lady', 'Rev', 'Mrs', 'Ms', 'Mr']:
    X,y,mean,stdev = _loadtrain('')
    print(X.shape)
    m,n = X.shape
#     for i in range(0, n):
#         print("TESTING FEATURE " + str(i) )
#         _test_model(X[0::, i:i+1],y)
#     _test_model(X, y)
    
    #filter model 0,1,2,13
    filter = [0,1,2,3,13]
    X_filtered = np.zeros( shape=(m, len(filter)) )
    for i in range(0, len(filter)):
        X_filtered[0::,i] = X[0::, filter[i]]
        
#     _test_model(X_filtered, y)
    clf_svm = svm.SVC(C=2.04, gamma=0.122)
# #     clf_forest = en.RandomForestClassifier(n_estimators=1000, max_features=3, max_depth=None, min_samples_split=1, random_state=0)
# #     clf_boost = en.GradientBoostingClassifier(n_estimators=1000, learning_rate=1, max_depth=None, random_state=0)
# #     
    clf_svm.fit(X,y)
# #     clf_forest.fit(X,y)
# #     clf_boost.fit(X,y)
# #     
# #     
# #     
# # # #     clf = RandomForestClassifier(n_estimators=1000)
# # #        
# # #     clf.fit(X,y)
# # #        
    X_test, df = _loadtest(mean, stdev)
    m,n = X_test.shape
    X_filtered_test = np.zeros( shape=(m, len(filter)) )
    for i in range(0, len(filter)):
        X_filtered_test[0::,i] = X_test[0::, filter[i]]
# #     
    svm_predict = clf_svm.predict(X_test)
# #     forest_predict = clf_forest.predict(X_test)
# #     boost_predict = clf_boost.predict(X_test)
# #     
# #     predictions = []
# #     for i in range(0, len(svm_predict)):
# #         t = [svm_predict[i],forest_predict[i],boost_predict[i]]
# #         prediction, count = stats.mode(t, 0)
# #         print(t)
# #         predictions.append(prediction[0])
# #        
#     
#     
# #        
    modelwriter = csv.writer(open('svm_model_better_parameters_no_filtering.csv', 'w'))
      
    modelwriter.writerow(['PassengerId','Survived'])
    for i in range(0,m):
        modelwriter.writerow([df[i,0],svm_predict[i].astype(np.int)])



