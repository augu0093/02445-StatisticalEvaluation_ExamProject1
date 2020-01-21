import numpy as np
from sklearn import model_selection
from sklearn import preprocessing
from scipy import stats
from toolbox_02450 import mcnemar
import matplotlib.pyplot as plt


from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

# The data needs to be imported form R. The code that creates the imported data files is within the R file
x = np.genfromtxt("C:/Users/Karlu/Desktop/Proj1/x9.csv", delimiter=' ')
y = np.genfromtxt("C:/Users/Karlu/Desktop/Proj1/y9.csv", delimiter=' ')

#stadardizing x
x = preprocessing.scale(x)

K = 100
CV = model_selection.KFold(n_splits=K,shuffle=True)
#random_state = 12

svm_predict = np.array([])
LDA_predict = np.array([])
KNN_predict = np.array([])
GNB_predict = np.array([])
DTC_predict = np.array([])
RF_predict = np.array([])
LR_predict = np.zeros(50).reshape(1,50)

opt_lambda = np.array([])

y_true = np.array([])

c = 0
for (train_index, test_index) in CV.split(x,y):
    x_train = x[train_index]
    x_test = x[test_index]
    
    y_train = y[train_index]
    y_test = y[test_index]
    y_true = np.append(y_true,y_test)
    
    #support vector machine
    m_svm = SVC(gamma = "auto")
    m_svm.fit(x_train,y_train)
    svm_predict = np.append(svm_predict,m_svm.predict(x_test))
    
    #Linear Discriminant Analysis
    m_LDA = LinearDiscriminantAnalysis()
    m_LDA.fit(x_train,y_train)
    LDA_predict = np.append(LDA_predict,m_LDA.predict(x_test))
    
    #KNeighbors Classifier
    m_KNN = KNeighborsClassifier()
    m_KNN.fit(x_train,y_train)
    KNN_predict = np.append(KNN_predict,m_KNN.predict(x_test))
    
    #GaussianNB
    m_GNB = GaussianNB()
    m_GNB.fit(x_train,y_train)
    GNB_predict = np.append(GNB_predict,m_GNB.predict(x_test))
    
    #Decision Tree Classifier
    m_DTC = DecisionTreeClassifier()
    m_DTC.fit(x_train,y_train)
    DTC_predict = np.append(DTC_predict,m_DTC.predict(x_test))
    
    #Random forrest classification
    m_RF = RandomForestClassifier()
    m_RF.fit(x_train,y_train)
    RF_predict = np.append(RF_predict,m_RF.predict(x_test))
    
    #Logistic regulized Regression with optimal lambda value
    k = 50
    lambda_interval = np.logspace(-2,2, k)
    
    temp_LR_predict = np.zeros(int(100/K)*k).reshape(int(100/K),k)
    for lambd in np.arange(k):
        m_LR = LogisticRegression(penalty='l2',solver= "liblinear",C=1/lambda_interval[lambd])
        m_LR.fit(x_train,y_train)
        temp_LR_predict[:,lambd] = m_LR.predict(x_test)
    
    
    LR_predict = np.vstack((LR_predict, temp_LR_predict))
    
    c +=1 
    if c % 10 == 0:
        print(c)

LR_predict = LR_predict[1:,:]

opt_lambda = np.zeros(50)
for i in range(k):
    opt_lambda[i] = np.mean(LR_predict[:,i] == y_true) 
    
opt_lambda_predict = LR_predict[:,opt_lambda.argmax()]


#majority voting
grouped = np.vstack((svm_predict,LDA_predict,GNB_predict,RF_predict,opt_lambda_predict))


majority_voting = np.zeros(grouped.shape[1])
for i in range(grouped.shape[1]):
    majority_voting[i] = stats.mode(grouped[:,i])[0][0]


#all the acc
print("svm_acc:", np.mean(y_true == svm_predict))   
print("LDA_acc:", np.mean(y_true == LDA_predict)) 
print("KNN_acc:", np.mean(y_true == KNN_predict))  
print("GNB_acc:", np.mean(y_true == GNB_predict)) 
print("DTC_acc:", np.mean(y_true == DTC_predict)) 
print("RF_acc:", np.mean(y_true == RF_predict)) 
print("LR_acc:", np.max(opt_lambda), "-with lambda", lambda_interval[opt_lambda.argmax()]) 
print("Majority voting:", np.mean(y_true == majority_voting))



#P-values from mcnemar test in a "confusion matrix" 
worst_to_best = np.vstack((DTC_predict,KNN_predict,RF_predict, LDA_predict, svm_predict,GNB_predict,LR_predict[:,opt_lambda.argmax()],majority_voting))
models = worst_to_best.shape[0]

acc_matrix = np.zeros(models**2).reshape(models,models)
for i in range(models):
    x1 = worst_to_best[i,:]
    
    for j in range(models):
        if j != i:
            x2 = worst_to_best[j,:]
            acc_matrix[i,j] = mcnemar(y_true,x1,x2)[2]

np.round(acc_matrix,2)
a = np.round(acc_matrix,2)
np.savetxt("subject_CI.csv", a, delimiter=",")



#plot CI for acc compared to logistic regression
#getting the CI's    
conf_int = np.array([])
compare =5 #logictiv regression if = 7
for i in range(worst_to_best.shape[0]):
    x1 = worst_to_best[compare,:]
    x2 = worst_to_best[i,:]
    
    if i != compare:
        conf_int = np.append(conf_int,mcnemar(y_true, x1, x2)[1])
    else:
        conf_int = np.append(conf_int,np.zeros(2))
      
  
      
#the actual plot of the CI      
width = 0.15
for i in range(int(np.size(conf_int)/2)):
    m = np.mean(conf_int[i*2:i*2+2])
    lower = conf_int[i*2]
    upper = conf_int[i*2+1]
    if i != 5:
        
        if lower < 0 and 0 < upper:
            plt.vlines(i, ymin= lower, ymax= upper, color ="red")
            plt.hlines(conf_int[i*2], xmin = i-width, xmax= i+width, color ="red")
            plt.hlines(conf_int[i*2+1],xmin = i-width, xmax= i+width, color ="red")
            plt.plot(i,m,"o",color="black")
        
        else: 
            plt.vlines(i, ymin= lower, ymax= upper)
            plt.hlines(conf_int[i*2], xmin = i-width, xmax= i+width)
            plt.hlines(conf_int[i*2+1],xmin = i-width, xmax= i+width)
            plt.plot(i,m,"o",color="black")
    
plt.axhline(0,linestyle="--", color = "blue")
plt.xticks(np.arange(8), ["DT", "KNN", "RF", "LDA","SVM","GNB","LR", "MV"])
plt.yticks(np.array([-0.1,0,0.1,0.2,0.3]), ["-10%","0%","10%","20%","30%"])
plt.title("CI for the difference in accuracy between GNB and ...")
plt.grid()
plt.savefig('classifier_CI',bbox_inches='tight',dpi=300)
plt.show()

# CI plot for mean difference of experiments
CI_exp = np.genfromtxt("C:/Users/Karlu/Desktop/Proj1/CI.csv", delimiter=' ')

    
width = 0.15
c = 0.0 #correction
for i in range(int(np.size(CI_exp)/2)):
    if i != 8:
        m = np.mean(CI_exp[i*2:i*2+2])
        lower = CI_exp[i*2]
        upper = CI_exp[i*2+1]
        
        if lower < 0 and 0 < upper:
            plt.vlines(i+c, ymin= lower, ymax= upper,color="red")
            plt.hlines(CI_exp[i*2], xmin = i-width, xmax= i+width,color="red")
            plt.hlines(CI_exp[i*2+1],xmin = i-width, xmax= i+width,color="red")
            plt.plot(i,m,"o",color="black")
        else: 
            plt.vlines(i+c, ymin= lower, ymax= upper)
            plt.hlines(CI_exp[i*2], xmin = i-width, xmax= i+width)
            plt.hlines(CI_exp[i*2+1],xmin = i-width, xmax= i+width)
            plt.plot(i,m,"o",color="black")

plt.axhline(0,linestyle="--", color = "blue")
ticks = []
for i in range(16):
    ticks.append("exp"+str(i+1))

plt.xticks(np.arange(16), ticks,rotation=60)
plt.title("CI on the bifference between means of exp-x and exp-9")
plt.grid()
plt.savefig('exp_CI',bbox_inches='tight',dpi=300)
plt.show()






#a = np.genfromtxt("C:/Users/Karlu/Desktop/Proj1/exp_CI.csv", delimiter=' ')
#np.savetxt("exp_CI1.csv", a, delimiter=",")
#
#a = np.genfromtxt("C:/Users/Karlu/Desktop/Proj1/exp_data.csv", delimiter=' ')
#np.savetxt("exp_data1.csv", a, delimiter=",")


    
            

    
    



















