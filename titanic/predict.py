import csv as csv
import pprint
import numpy as np
import pylab as P
from sklearn import svm
import sklearn.metrics as skm


csv_obj = csv.reader(open('train.csv', 'r'))

header = csv_obj.next()

print(header)

data=[]
for row in csv_obj:
    data.append(row)
    
data = np.array(data)

#access data like this
#print(data[1:10,0::])

age = data[0::,5] == ''
data[age,5] = 20 #set to something close to the average age
age = data[0::,5].astype(np.float)

females = data[0::,4] == 'female'
data[females, 4] = 0
males = data[0::, 4] == 'male'
data[males, 4] = 1

gender = data[0::,4].astype(np.float)

pClass = data[0::,2].astype(np.float)

sibsp = data[0::,6].astype(np.float)
parch = data[0::,7].astype(np.float)

#n, bins, patches = P.hist(gender, 50, normed=1, histtype='stepfilled')
#P.setp(patches, 'facecolor', 'g', 'alpha', 0.75)
#P.show()

X = np.zeros((data[0::,0].size, 5))
X[0::,0] = age
X[0::,1] = gender
X[0::,2] = pClass
X[0::,3] = sibsp
X[0::,4] = parch

y = data[0::,1].astype(np.float)

m,n = X.shape

clf = svm.SVC(C=3.5)

clf.fit(X,y)

test_obj = csv.reader(open('test.csv', 'r'))

test_obj.next() #kick out the header

data=[]
for row in test_obj:
    data.append(row)
    
data = np.array(data)

m,n = data.shape

#access data like this
#print(data[1:10,0::])

age = data[0::,4] == ''
data[age,4] = 23 #set to something close to the average age
age = data[0::,4].astype(np.float)

females = data[0::,3] == 'female'
data[females, 3] = -1
males = data[0::, 3] == 'male'
data[males, 3] = 1

gender = data[0::,3].astype(np.float)

pClass = data[0::,1].astype(np.float)

sibsp = data[0::,5].astype(np.float)
parch = data[0::,6].astype(np.float)

#n, bins, patches = P.hist(gender, 50, normed=1, histtype='stepfilled')
#P.setp(patches, 'facecolor', 'g', 'alpha', 0.75)
#P.show()

X = np.zeros((data[0::,0].size, 5))
X[0::,0] = age
X[0::,1] = gender
X[0::,2] = pClass
X[0::,3] = sibsp
X[0::,4] = parch

y_predict = clf.predict(X)

modelwriter = csv.writer(open('svm_model.csv', 'w'))

modelwriter.writerow(['PassengerId','Survived'])
for i in range(0,m):
    modelwriter.writerow([data[i,0],y_predict[i].astype(np.int)])

