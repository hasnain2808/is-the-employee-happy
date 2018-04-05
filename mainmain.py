import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import seaborn as sb
from sklearn.utils import shuffle
from sklearn import svm


plt.ion()

path=os.getcwd()+'/data/HR_comma_sep.csv'
data = pd.read_csv(path)

print(data.head(50))
print(data.describe())

#describe all the data

print('\n\nindividual desc\n')
print(data['satisfaction_level'].describe())
print('\n\n\n')
print(data['last_evaluation'].describe())
print('\n\n\n')
print(data['number_project'].describe())
print('\n\n\n')
print(data['average_montly_hours'].describe())
print('\n\n\n')
print(data['time_spend_company'].describe())
print('\n\n\n')
print(data['Work_accident'].describe())
print('\n\n\n')
print(data['left'].describe())
print('\n\n\n')
print(data['promotion_last_5years'].describe())
print('\n\n\n')
print(data['sales'].describe())
print('\n\n\n')
print(data['salary'].describe())
print('\n\n\n')

fig, ax = plt.subplots(figsize=(12,8))  
ax.bar(data['average_montly_hours'],data['left'])  
axes = plt.gca()
axes.set_xlim([0,400])
plt.show()

#sb.barplot(data['satisfaction_level'],data['left'])
#input('')
#plt.clf()
#sb.barplot(data['last_evaluation'],data['left'])
#input('')
#plt.clf()
#sb.barplot(data['number_project'],data['left'])
#input('')
#plt.clf()
#sb.barplot(data['average_montly_hours'],data['left'])
#input('')
#plt.clf()
#sb.barplot(data['time_spend_company'],data['left'])
#input('')
plt.clf()
sb.barplot(data['Work_accident'],data['left'])
input('')
plt.clf()
#sb.barplot(data['promotion_last_5years'],data['left'])
#input('')
#plt.clf()
#sb.barplot(data['sales'],data['left'])
#input('')
#plt.clf()
#sb.barplot(data['salary'],data['left'])

#data=np.random.shuffle(data)
data =pd.DataFrame(shuffle(data))
print(data.head(25))
Y=data['left']
data=data.drop(columns=['left'])
print(Y.head(25))
print(data.head(25))


saledi={'sales':1,'accounting':2,'hr':3,'technical':4,'support':5,'management':6,'IT':7,'product_mng':8,'marketing':9,'RandD':10}
data['sales']=data['sales'].map(saledi)

salarydi={'low':0,'medium':1,'high':2}
data['salary']=data['salary'].map(salarydi)



Xtrain = data.iloc[:12000,:]  
Xtest = data.iloc[12001:,:]  

print(Xtest.describe())
print(Xtrain.describe())

Ytrain = Y.iloc[:12000]  
Ytest = Y.iloc[12001:,]  

print(Ytrain.describe())
print(Ytest.describe())
print(Xtest.describe())
print(Xtrain.describe())

svc = svm.SVC(C=20)  
svc.fit(Xtrain, Ytrain)  
print('Test accuracy = {0}%'.format(np.round(svc.score(Xtest, Ytest) * 100, 2)))  

svc = svm.LinearSVC()  
svc.fit(Xtrain, Ytrain)  
print('Test accuracy = {0}%'.format(np.round(svc.score(Xtest, Ytest) * 100, 2)))  

input('')
plt.close('all')
