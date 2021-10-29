#!/usr/bin/env python
# coding: utf-8

# # About the comapny

# Dream Housing Finance company deals in all home loans. They have presence across all urban, semi urban and rural areas. Customer first apply for home loan after that company validates the customer eligibility for loan.

# # Problem

# Company wants to automate the loan eligibility process (real time) based on customer detail provided while filling online application form. These details are Gender, Marital Status, Education, Number of Dependents, Income, Loan Amount, Credit History and others. To automate this process, they have given a problem to identify the customers segments, those are eligible for loan amount so that they can specifically target these customers. Here they have provided a partial data set.

# # Let us Start!

# In[1]:


#Importing the libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#Reading the training and test datafile

train=pd.read_csv('C:\\Users\\User-1\\Downloads\\train_lp.csv')
train['LoanAmount']=train['LoanAmount']*1000
test=pd.read_csv('C:\\Users\\User-1\\Downloads\\test_.csv')
test['LoanAmount']=test['LoanAmount']*1000

#Printing first 5 rows of training file

train.head()


# In[2]:


#To study descriptive statistics of our data

train.describe(include='all')


# In[3]:


#Plotted the countplot of our target variable to check whether our data is proper or biased

train['Loan_Status'].value_counts().plot(kind='bar')
plt.xlabel('Loan Status')
plt.ylabel('Count')


# In[4]:


#Plotted the heatmap to check the corelation of numerical featurres

plt.figure()
sns.heatmap(train.corr(),annot=True)
plt.show()


# In[5]:


#Since we know that Loan_Amount_Term and Credit_History are also categorical features we have use this code instead of using dtypes for selecting categorical features

cat_features=[index for index in train.columns if len(train[index].unique())<15]
cat_features


# In[6]:


#Plotted varoius countplot for categorical feaures 

for cat in cat_features:
    train[cat].value_counts().plot(kind='bar')
    plt.xlabel(cat)
    plt.show()


# In[7]:


#Defined the function to check the unique values 

def counts(feature):
    return train[feature].value_counts()
counts('Married')


# After analysing all features independents now we will see how differnt categorical features have their effect on our target variable loan status

# In[8]:


for feature_x in cat_features:
       a=pd.crosstab(train[feature_x],train['Loan_Status'])
       a.plot(kind='bar',stacked=True)
       plt.show()


# Graph 3-Distribution of applicants with 1 or 3+ dependents is similar across both categories of loan status.But applicant with 0 dependents have higher proportion of approved loans

# Graph 1-Proportion of male and female applicant is more or less same for approved and not approved loan status But male has somewhat higher proportion of approved loans compared to female applicant

# Graph 2-Proportion of married applicant is higher for approved loans

# 
# Graph 4-Generally applicants who apply for loan are graduates and proportion of graduate applicant is higher for approved loans

# Graph 5-There is nothing significant that we can infer from graph of self employed vs loan status but there are are more no of appliacnt who are unemployed and they apply for loan.and proportion of unemployed applicant as well as employed applicant is higher for approved loans 

# Graph-7 It seems like people with credit history as 1 are more likely to get their loans approved

# Graph 8-Proportion of loans getting approved in semi urban areas is higher as compared to rural and urban area

# In[9]:


#To check for the missing values

train.isnull().sum()


# As there is no stong correlation of these variables with any other variable so i decided to fill missing values either by their median value or mode value

# In[10]:


#Fill the missing values

train['Dependents'].replace('3+','3',inplace=True)
train['Married'].fillna(train['Married'].mode()[0],inplace=True)
train['Dependents'].fillna('0',inplace=True)

#
train['Loan_Amount_Term'].fillna(train['Loan_Amount_Term'].mode()[0],inplace=True)


# In[11]:


sns.catplot('Loan_Amount_Term','LoanAmount',data=train)


# As we can see that there are many outliers in loan amount so it is better to fill missing values with the median or mode 

# In[12]:



train['LoanAmount'].fillna(train['LoanAmount'].median(),inplace=True)


# In[13]:


a=pd.crosstab(train['Self_Employed'],train['Dependents'])
a.plot(kind='bar')
plt.show()


# Form the bar plot of dependents vs self_employed it can be observed that if there are 0 dependents around 270 of them are unemployed out of 345.
# 
# 
# 0 Dependents also means that no one in his family is dependent on his/her income so one possibililty is that he/she is still studying
# 
# But at the end of the day  90 % values of self employed corresponds to 'No' so we will fill the missing values with no.

# In[14]:


train['Self_Employed'].fillna('No',inplace=True)


# I decided to fill missing values of credit history with 2 since null values in credit history signifies that applicant had not borrowd any loan previously

# In[126]:


train['Credit_History'].fillna(2,inplace=True)


# We shall fill missing values of Credit_History with 2 , since : -
# 
# 0 stands for applicants who took a Loan in the past but could not repay the Loan , due to different factors
# 
# 1 stands for applicants who took a Loan in the past and have repayed the Loan generously
# 
# 2 stands for applicants who are basically , the First Timers (Never - ever took a Loan)

# In[127]:


a=pd.crosstab(train['Gender'],train['Married'])
a.div(a.sum(1),axis=0).plot(kind='bar',stacked=True)
plt.show()


# None of the graph are able to depict as to what gender should be but looking at thi plot one can clearly say that counts of Female being married is really low

# In[15]:


dict={'Yes':'Male','No':'Female'}
train['Gender'].fillna(train['Married'].map(dict),inplace=True)


# In[ ]:


train['Gender'].value


# In[129]:


train.isnull().sum()


# After analyzing categorical features now we will look at our continuous variable and plot their histogram and boxplot to check the skewness and to identify the outliers present in our continuous variable

# ### Histograms

# In[130]:



features=['ApplicantIncome','CoapplicantIncome','LoanAmount']
for i in features:
    plt.hist(train[i],bins=10)
    plt.xlabel(i)
    plt.show()


# The above histograms are positively skewed so in order to remove the skewness we will do log transformation

# In[131]:


train['log_appincome']=np.log(train['ApplicantIncome'])
train['sqrt_coappincome']=np.sqrt(train['CoapplicantIncome'])


# In[132]:


g=sns.FacetGrid(train,col='Self_Employed')
g.map(plt.hist,'LoanAmount',bins=10)


# ### Boxplots

# In[133]:


feature=['ApplicantIncome','CoapplicantIncome','LoanAmount']
for i in feature:
    plt.figure()
    train[i].plot(kind='box')
    plt.show()


# In[134]:


def boxplot(feature,basis):
    return sns.boxplot(train[feature],train[basis])
boxplot('Education','ApplicantIncome')


# I've tried to make a function which can plot any graph amongst boxplot,barplot,catplot which is important for this dataset and i have missed . Instead of making many graph and rejecting it if it is not useful,function can plot all types of graph by writing just one line of code.

# In[135]:


feature=['ApplicantIncome','CoapplicantIncome','LoanAmount']
def plot_graph(x_axis_feature,type_of_graph,hue=None):
    for i in feature:
        if (type_of_graph=='boxplot'):
            sns.boxplot(x_axis_feature,i,hue=hue,data=train)
            plt.show()
        elif (type_of_graph=='barplot'):
             sns.barplot(x_axis_feature,i,data=train)
             plt.show()
        elif (type_of_graph=='catplot'):
             sns.barplot(x_axis_feature,i,hue=hue,data=train)
             plt.show()
plot_graph('Dependents','boxplot')


# From above graph it is clearly visible that applicant having more no of dependents apply for more loan amount and they also have higher source of income

# ### Removing the outliers

# Interquartile Range (IQR) is important because it is used to define the outliers. It is the difference between the third quartile and the first quartile (IQR = Q3 -Q1). Outliers in this case are defined as the observations that are below (Q1 − 1.5x IQR) or boxplot lower whisker or above (Q3 + 1.5x IQR) or boxplot upper whisker.
# 
# If data distribution is approx normal then about 68% values lie within one standard deviation of mean and about 95% lie within two standard deviation and about 99.7% values lie within three standard deviation.
# Therefore, if you have any data point that is more than 3 times standard deviation then those points are likely to be considered as outlier
# 
# 

# In[136]:


#Removing outliers with the help of z score

def remove_outlier(column):
    std=np.std(train[column])
    mean=np.mean(train[column])
    outlier=[]
    for i in train[column]:
        zscore=(i-mean)/std
        
        #Considering z>3 because z>3 sinifies 99.7%values fall in that region
        
        if(zscore>3):
            outlier.append(i)
            minimum=np.min(outlier)
    return minimum


# In[137]:


print(remove_outlier('ApplicantIncome'))
print(remove_outlier('CoapplicantIncome'))
print(remove_outlier('LoanAmount'))


# In[138]:


train['ApplicantIncome']=train['ApplicantIncome'].where(train['ApplicantIncome']<23803,train['ApplicantIncome'].median())
train['CoapplicantIncome']=train['CoapplicantIncome'].where(train['CoapplicantIncome']<10968,train['CoapplicantIncome'].median())
train['LoanAmount']=train['LoanAmount'].where(train['LoanAmount']<400000,train['LoanAmount'].median())


# In[139]:


train.describe()


# ### Feature Engineering

# In[140]:


train['loanamt']=pd.cut(train['LoanAmount'],bins=[8999,100250,160000,380001],labels=['low','avg','high'])
a=pd.crosstab(train['loanamt'],train['Loan_Status'])
a.div(a.sum(1),axis=0).plot(kind='bar')
plt.show()


# From above graph it can be concluded that if the loan amount is less than chances of loan getting approved is higher

# In[141]:


train['Total_Income']=train['ApplicantIncome']+train['CoapplicantIncome']
train['Total_Income_bin']=pd.cut(train['Total_Income'],bins=[1441,3997.5,7165,27501],labels=['low','avg','high'])
a=pd.crosstab(train['Total_Income_bin'],train['Loan_Status'])
a.div(a.sum(1),axis=0).plot(kind='bar')
plt.show()


# In[142]:


def set_scr(train):
    if train["Total_Income_bin"] == "low" and train["loanamt"] == "high":
        return 1
    elif train["Total_Income_bin"] == "avg" and train["loanamt"] == "high":
        return 2
    elif train["Total_Income_bin"] == "low" and train["loanamt"] == "avg":
        return 4
    elif train["Total_Income_bin"] == "high" and train["loanamt"] == "high":
        return 8
    elif train["Total_Income_bin"] == "avg" and train["loanamt"] == "avg":
        return 7
    elif train["Total_Income_bin"] == "low" and train["loanamt"] == "low":
        return 3
    elif train["Total_Income_bin"] == "high" and train["loanamt"] == "avg":
        return 6
    elif train["Total_Income_bin"] == "avg" and train["loanamt"] == "low":
        return 5
    else :
        return 9
train['score']=train.apply(set_scr,axis=1)
train.head()


# Previously we had given the score from 1 to 9 following the general trend bu when we studied the behaviour of our bank we came to know that our bank follows diff trend to give loan so we manipulated the credit score

# In[143]:


a=pd.crosstab(train['score'],train['Loan_Status'])
a.div(a.sum(1),axis=0).plot(kind='bar',stacked=True)
plt.show()


# In[144]:


sns.lmplot('ApplicantIncome','LoanAmount',data=train)


# Above graph shows the positive correlation between loanamount and applicantincome.
# 
# As ApplicantIncome increases loanamount also increases

# In[145]:


sns.lmplot('Total_Income','LoanAmount',data=train,hue='Education')


# In[146]:


#I've calculated EMI that every person will pa eah and every month
train['EMI']=(train['LoanAmount']*(7.85/1200)*((1+(7.85/1200))**train['Loan_Amount_Term']))/((1+(7.85/1200))**(train['Loan_Amount_Term']-1))
#Now we will calculate Risk factor
train['Risk_Percent']=(train['EMI']/train['ApplicantIncome'])


# In[147]:


sns.barplot('Loan_Status','Risk_Percent',data=train)


# Above graph clearly signifies that there is high risk percent for loan status of no.
# 
# So if there is high risk percent chances of loan approval is low

# In[148]:


train['Risk_Percent_bin']=pd.cut(train['Risk_Percent'],bins=[0.019859,0.153936,0.204763,0.281542,5.926015],labels=[0,1,2,3])


# 0 - Very low risk
# 
# 1 - Low risk
# 
# 2 - High risk
# 
# 3 - Very high risk

# In[149]:


a=pd.crosstab(train['Risk_Percent_bin'],train['Loan_Status'])
a.plot(kind='bar')
plt.show()


# Creating risk percent bin didnt justify our main aim as for high risk there are more chances for approvd loans

# We have tried to make new columns which shows no of coapplicant based on coapplicant income

# In[150]:


def num(train):
    if train['CoapplicantIncome']==0:
        return 0
    elif train['CoapplicantIncome']>=0 and train['CoapplicantIncome']<=2250:
        return 1
    elif train['CoapplicantIncome']>2250 and train['CoapplicantIncome']<=9000:
        return 2
train['coapplicant']=train.apply(num,axis=1)


# In[151]:


a=pd.crosstab(train['coapplicant'],train['Loan_Status'])
a.plot(kind='bar')
plt.show()


# In[152]:


a


# So if there are more no of coapplicant there is low chance that your loan will get rejected but that proportion of higher approved loans is not that significant for coapplicant 1 and 2

# In[153]:


plt.hist(train['Total_Income'],bins=8)
plt.xlabel('Total Income')
plt.show()


# In[154]:


plt.hist(train['EMI'],bins=8)
plt.xlabel('EMI')
plt.show()


# In[155]:


# So we also need to remove skewness from Total Income
train['log_total_income']=np.log(train['Total_Income'])


# In[156]:


# As there are 650 zero's in coapplicantincome and we have craeted a new column total income so we can drop coapplicant income
train=train.drop(['CoapplicantIncome'],axis=1)


# ### Label Encoding

# In[157]:


train.info()


# In[158]:


from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
train['Gender']=le.fit_transform(train['Gender'])
train['Married']=le.fit_transform(train['Married'])

train['Self_Employed']=le.fit_transform(train['Self_Employed'])
train['Loan_Status']=le.fit_transform(train['Loan_Status'])
train['loanamt']=train['loanamt'].astype('object')
train['Total_Income_bin']=train['Total_Income_bin'].astype('object')
dict={'low':1,'avg':2,'high':3}
dict1={'low':1,'avg':2,'high':3}
dict2={'Graduate':1,'Not Graduate':0}
dict3={"Rural":0,'Urban':2,'Semiurban':1}
dict4={12:2,36:3,60:4,84:5,120:6,180:7,240:8,300:9,350:10,360:11,480:12}
train['Loan_Amount_Term']=train['Loan_Amount_Term'].map(dict4)
train['loanamt']=train['loanamt'].map(dict)
train['Education']=train['Education'].map(dict2)
train['Total_Income_bin']=train['Total_Income_bin'].map(dict1)
train['Property_Area']=train['Property_Area'].map(dict3)


# In[159]:


pd.set_option('display.max_columns',None)
train.head()


# # Feature Selection

# In[160]:


train.columns


# In[161]:


X=train[['Gender', 'Married', 'Dependents', 'Education',
       'Self_Employed', 'ApplicantIncome', 'LoanAmount',
       'Loan_Amount_Term', 'Credit_History', 'Property_Area',
       'log_appincome', 'sqrt_coappincome', 'loanamt',
       'Total_Income', 'Total_Income_bin', 'score', 'EMI', 'Risk_Percent','Risk_Percent_bin','log_total_income','coapplicant']]
y=train[['Loan_Status']]
X1=train[['loanamt','Total_Income_bin','score','Gender','Married','Dependents','Education','Self_Employed','Property_Area','coapplicant']]


# ### Performing train test split
# 

# Using train test split to create training set and validation set

# In[162]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=31,test_size=0.2)


# ### Standard scaling on training and testing data

# Standardisation is a scaling technique where the values are centred around mean with a unit standard deviation and mean of attributed becomes zero.
# 
# Normalisation is good to use when we do not have gaussian distribution but generally normalization is used for models like knn and neural network.In general we use standadisation since it is not affected by outliers and it works better compared to normalisation.At the end of the day we use scaling technique which works better.
# 
# It is good practice to fit scaler on training data and then use it to transform testing data to avoid data leakage

# In[163]:


from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
X_train[['ApplicantIncome','LoanAmount','Total_Income','EMI','log_appincome','Risk_Percent','sqrt_coappincome','log_total_income']]=scaler.fit_transform(X_train[['ApplicantIncome','LoanAmount','Total_Income','EMI','log_appincome','Risk_Percent','sqrt_coappincome','log_total_income']])
X_test[['ApplicantIncome','LoanAmount','Total_Income','EMI','log_appincome','Risk_Percent','sqrt_coappincome','log_total_income']]=scaler.transform(X_test[['ApplicantIncome','LoanAmount','Total_Income','EMI','log_appincome','Risk_Percent','sqrt_coappincome','log_total_income']])


# ## Univariate Feature Selection Methods

# In[164]:


from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier
knn=KNeighborsClassifier(n_neighbors=30)
from sklearn.linear_model import LogisticRegression
logreg=LogisticRegression(random_state=31)
from sklearn.ensemble import RandomForestClassifier
log=RandomForestClassifier(random_state=31)
from sklearn.tree import DecisionTreeClassifier
dtc=DecisionTreeClassifier(random_state=31)
from sklearn.ensemble import GradientBoostingClassifier
gbc=GradientBoostingClassifier(random_state=31)
from sklearn.ensemble import AdaBoostClassifier
abc=AdaBoostClassifier(random_state=31)
from sklearn.ensemble import BaggingClassifier
bc=BaggingClassifier(AdaBoostClassifier(random_state=31))
from mlxtend.classifier import StackingClassifier
scc=StackingClassifier(classifiers=[gbc],meta_classifier=abc)
from sklearn.svm import SVC
svc=SVC(random_state=31)
from sklearn.naive_bayes  import GaussianNB
gb=GaussianNB()


# ## Filter Methods

# Pearson’s Correlation: It is used as a measure for quantifying linear dependence between two continuous variables X and Y. Its value varies from -1 to +1. Pearson’s correlation is given as:
# fs2
# 
# LDA: Linear discriminant analysis is used to find a linear combination of features that characterizes or separates two or more classes (or levels) of a categorical variable.
# 
# ANOVA: ANOVA stands for Analysis of variance. It is similar to LDA except for the fact that it is operated using one or more categorical independent features and one continuous dependent feature. It provides a statistical test of whether the means of several groups are equal or not.
# 
# Chi-Square: It is a is a statistical test applied to the groups of categorical features to evaluate the likelihood of correlation or association between them using their frequency distribution.
# 
# One thing that should be kept in mind is that filter methods do not remove multicollinearity. So, you must deal with multicollinearity of features as well before training models for your data.

# ### Anova Test

# In[165]:


from sklearn.feature_selection import chi2,f_classif
sel=f_classif(X_train,y_train)
p=pd.Series(sel[1])
p.index=X_train.columns
p=p[p<0.05]
p


# ### Chi Square Test

# In[166]:


from sklearn.feature_selection import chi2
sel=chi2(X1,y)
pv=pd.Series(sel[1])
pv.index=X1.columns

pv


# In[167]:


from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LogisticRegression
sel=SelectFromModel(LogisticRegression(penalty='l1',C=0.05,solver='liblinear'))
sel.fit(X_train,y_train)
X_train.columns[sel.get_support()]


# In[168]:


from sklearn.feature_selection import SelectPercentile , mutual_info_classif
sel=SelectPercentile(mutual_info_classif,percentile=10).fit(X_train,y_train)
X_train.columns[sel.get_support()]


# In[169]:


corr_features=set()
corr_matrix=train.corr()
for i in range(len(corr_matrix.columns)):
    for j in range (i):
        if(abs(corr_matrix.iloc[i,j]>0.8)):
            corr_features.add(corr_matrix.columns[i])
corr_features


# ## Wraper Methods

# Forward Selection: Forward selection is an iterative method in which we start with having no feature in the model. In each iteration, we keep adding the feature which best improves our model till an addition of a new variable does not improve the performance of the model.
# 
# Backward Elimination: In backward elimination, we start with all the features and removes the least significant feature at each iteration which improves the performance of the model. We repeat this until no improvement is observed on removal of features.
# 
# Recursive Feature elimination: It is a greedy optimization algorithm which aims to find the best performing feature subset. It repeatedly creates models and keeps aside the best or the worst performing feature at each iteration. It constructs the next model with the left features until all the features are exhausted. It then ranks the features based on the order of their elimination.

# In[170]:


from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from sklearn.ensemble import RandomForestClassifier
sfs=SFS(AdaBoostClassifier(random_state=31),
        k_features=5,
        forward= True,
        floating=False,
        verbose=2,
        cv=4,
        scoring='accuracy',
        n_jobs=-1
       
       ).fit(X_train,y_train)
sfs.k_feature_names_


# In[ ]:


from mlxtend.feature_selection import ExhaustiveFeatureSelector as EFS
from sklearn.ensemble import RandomForestClassifier
efs=EFS(LogisticRegression(random_state=31),
        max_features=5,
        min_features=4,
        cv = None,
        scoring='accuracy',
        n_jobs=-1
       
       ).fit(X_train,y_train)
efs.best_feature_names_


# In[171]:


from sklearn.feature_selection import RFE
sel=RFE(AdaBoostClassifier(random_state=31),n_features_to_select=7)
sel.fit(X_train,y_train)
X_train.columns[sel.get_support()]


# # Training the Model

# I've written this for loop to make the task easier for any analyst.
# 
# We just have to dump all possible permutation of features in one list and names of the model in another list.
# So the loop will run in such a way that for one combination of features it will give us the accuracy score of that list of features with all the models

# In[209]:


x1=X_train.copy()
x2=X_test.copy()


# With the help of the sample codes above , one can manipulate the parameters inside to get the following List
# 
# List 1 : Is a result of features that our initial training datafile had (feature engineering features excluded)
# 
# List 2 : Is a result of features that out training datafile has (feature engineering features included)
# 
# List 3 : Is a result of estimator = AdaBoostRegressor() in FFS
# 
# List 4 : Is a result of estimator = LogisticRegression() in FFS
# 
# List 5 : Is a result of estimator = GradientBoostingRegressor() in FFS
# 
# List 6 : Is a result of estimator = AdaBoostRegressor() in BFE
# 
# List 7 : Is a result of estimator = LogisticRegression() and n_features_to_select = 5 in RFE
# 
# List 8 : Is a result of List 9 (keeping only Net_Income) , out of which 1 correlated feature was removed
# 
# List 9 : Is a result of List 9 (keeping only log_Net_Income) , out of which 1 correlated feature was removed
# 
# List 10 : Is a result of estimator = AdaBoostRegressor() and cv=2 in EFS
# 
# List 11 : Is a result of estimator = AdaBoostRegressor() and cv=3 in EFS

# In[210]:


# List of all possible sets of feature
li1=[['EMI','Gender', 'Married', 'Dependents', 'Education',
       'Self_Employed', 'ApplicantIncome', 'LoanAmount',
       'Loan_Amount_Term', 'Credit_History', 'Property_Area',
       'log_appincome', 'sqrt_coappincome', 'loanamt',
       'Total_Income', 'Total_Income_bin', 'score', 'Risk_Percent','Risk_Percent_bin','log_total_income','coapplicant'],['Loan_Amount_Term','Gender', 'Married', 'Dependents', 'Education',
       'Self_Employed', 'ApplicantIncome', 'LoanAmount',
     'Credit_History', 'Property_Area'],['Gender', 'Married', 'Dependents', 'Education', 'Credit_History'],['Married', 'Education', 'Self_Employed', 'Credit_History', 'loanamt'],['Education','Married', 'Credit_History', 'loanamt', 'coapplicant'],['score','Gender', 'Education', 'Credit_History', 'log_total_income'],['Married', 'LoanAmount', 'Credit_History', 'Total_Income_bin',
       'log_total_income'],['ApplicantIncome', 'LoanAmount', 'Credit_History', 'Risk_Percent',
       'log_total_income'],['ApplicantIncome', 'Credit_History', 'EMI', 'Risk_Percent',
       'log_total_income'],['Loan_Amount_Term', 'Credit_History', 'loanamt', 'score', 'coapplicant'],['ApplicantIncome', 'log_appincome', 'Total_Income', 'EMI',
       'Risk_Percent'],['ApplicantIncome', 'Total_Income', 'Credit_History', 'Risk_Percent',
       'Gender'],['ApplicantIncome', 'Total_Income', 'Credit_History', 'Risk_Percent',
       'Dependents'],['LoanAmount','Loan_Amount_Term','Risk_Percent','Credit_History','sqrt_coappincome']]


# In[211]:


ya=y_train['Loan_Status']
yb=y_test['Loan_Status']


# In[212]:


# List of all models used for classification problem
li2=['Logistic','ada','stacking','DecisionTree','RandomForest','Gradient','Bagging','knn','svc','naive']


# Okay , so we're ready with an all possible features list and models list . What Next ?
# 
# Lets think like this . What if we have a list 1 and we wish to try that list with each and every model type ? Think .. Think ! ! !
# 
# .......
# 
# Okay so a short , sweet and simple for loop would suffice . But, what if we have multiple lists and multiple models ?
# 
# Ahahahh ....You got that right ! Use 2 for loops !
# 
# Dont hesitate when you have a look at the code below . It's a pretty simple code (a function named get_models) that shall help us build a dataframe .
# 
# We've created some temporary files , that can help us overwrite data , in our for loop

# In[213]:


row=[]
for i in li1:
    for j in li2:
        if j=='Logistic':
            X_train=X_train[i]
            X_test=X_test[i]
            logreg.fit(X_train,ya)
            final=logreg.predict(X_test)
            finall=logreg.predict(X_train)
            sub=(metrics.accuracy_score(yb,final)-metrics.accuracy_score(ya,finall))
            row.append([i,j,metrics.accuracy_score(yb,final),sub])
            X_train=x1
            X_test=x2
        elif j=='ada':
            X_train=X_train[i]
            X_test=X_test[i]
            abc.fit(X_train,ya)
            final1=abc.predict(X_test)
            final11=logreg.predict(X_train)
            sub=(metrics.accuracy_score(yb,final)-metrics.accuracy_score(ya,final11))
            row.append([i,j,metrics.accuracy_score(yb,final1),sub])
            X_train=x1
            X_test=x2
        elif j=='stacking':
            X_train=X_train[i]
            X_test=X_test[i]
            scc.fit(X_train,ya)
            final2=scc.predict(X_test)
            final22=logreg.predict(X_train)
            sub=(metrics.accuracy_score(yb,final)-metrics.accuracy_score(ya,final22))
            row.append([i,j,metrics.accuracy_score(yb,final2),sub])
            X_train=x1
            X_test=x2
        elif j=='DecisionTree':
            X_train=X_train[i]
            X_test=X_test[i]
            dtc.fit(X_train,ya)
            final3=dtc.predict(X_test)
            final33=logreg.predict(X_train)
            sub=(metrics.accuracy_score(yb,final)-metrics.accuracy_score(ya,final33))
            row.append([i,j,metrics.accuracy_score(yb,final3),sub])
            X_train=x1
            X_test=x2
        elif j=='RandomForest':
            X_train=X_train[i]
            X_test=X_test[i]
            log.fit(X_train,ya)
            final4=log.predict(X_test)
            final44=logreg.predict(X_train)
            sub=(metrics.accuracy_score(yb,final)-metrics.accuracy_score(ya,final44))
            row.append([i,j,metrics.accuracy_score(yb,final4),sub])
            X_train=x1
            X_test=x2
        elif j=='Gradient':
            X_train=X_train[i]
            X_test=X_test[i]
            gbc.fit(X_train,ya)
            final5=gbc.predict(X_test)
            final55=logreg.predict(X_train)
            sub=(metrics.accuracy_score(yb,final)-metrics.accuracy_score(ya,final55))
            row.append([i,j,metrics.accuracy_score(yb,final5),sub])
            X_train=x1
            X_test=x2
        elif j=='Bagging':
            X_train=X_train[i]
            X_test=X_test[i]
            bc.fit(X_train,ya)
            final6=bc.predict(X_test)
            final66=logreg.predict(X_train)
            sub=(metrics.accuracy_score(yb,final)-metrics.accuracy_score(ya,final66))
            row.append([i,j,metrics.accuracy_score(yb,final6),sub])
            X_train=x1
            X_test=x2
        elif j=='knn':
            X_train=X_train[i]
            X_test=X_test[i]
            knn.fit(X_train,ya)
            final7=knn.predict(X_test)
            final77=logreg.predict(X_train)
            sub=(metrics.accuracy_score(yb,final)-metrics.accuracy_score(ya,final77))
            row.append([i,j,metrics.accuracy_score(yb,final7),sub])
            X_train=x1
            X_test=x2
        elif j=='svc':
            X_train=X_train[i]
            X_test=X_test[i]
            svc.fit(X_train,ya)
            final8=svc.predict(X_test)
            final88=logreg.predict(X_train)
            sub=(metrics.accuracy_score(yb,final)-metrics.accuracy_score(ya,final88))
            row.append([i,j,metrics.accuracy_score(yb,final8),sub])
            X_train=x1
            X_test=x2
        elif j=='naive':
            X_train=X_train[i]
            X_test=X_test[i]
            gb.fit(X_train,ya)
            final9=gb.predict(X_test)
            final99=logreg.predict(X_train)
            sub=(metrics.accuracy_score(yb,final)-metrics.accuracy_score(ya,final99))
            row.append([i,j,metrics.accuracy_score(yb,final9),sub])
            X_train=x1
            X_test=x2


# In[219]:


df=pd.DataFrame(row,columns=['Features','model','score','sub'])
df=df.sort_values(by=['sub'],ascending=True)
df.head(10)
#df.iloc[111]['Features']


# I've written this for loop to make the task easier for any analyst.
# 
# We just have to dump all possible permutation of features in one list and names of the model in another list.
# So the loop will run in such a way that for one combination of features it will give us the accuracy score of that list of features with all the models

# In[69]:


from imblearn.under_sampling import RandomUnderSampler


# As our dataset has yes to no ratio of 80% is to 20% so i thought of doing undersampling and oversampling.
# 
# Random oversampling duplicates examples from the minority class in the training dataset and can result in overfitting for some models.
# 
# Random undersampling deletes examples from the majority class and can result in losing information invaluable to a model.

# In[70]:


undersample=RandomUnderSampler(sampling_strategy='majority')
undersample=RandomUnderSampler(sampling_strategy=1)


# In[220]:


Xa=X_train[['ApplicantIncome', 'Total_Income', 'Credit_History', 'Risk_Percent']]
Xb=X_test[['ApplicantIncome', 'Total_Income', 'Credit_History', 'Risk_Percent']]
Xaa,yaa = undersample.fit_resample(Xa,ya)


# But we are not getting any better result after doing this since this technique is efficient for dataset having yes to no ratio ver less . 
# 
# for example 100:1,1000:1

# # KNeighborsClassifier

# In[174]:


from sklearn.neighbors import KNeighborsClassifier
knn=KNeighborsClassifier(n_neighbors=30)
knn.fit(Xa,ya)
pred0=knn.predict(Xb)
from sklearn import metrics
print(metrics.classification_report(yb,pred0))
print(metrics.confusion_matrix(yb,pred0))


# In[175]:


from sklearn.preprocessing import binarize
y_prob=knn.predict_proba(Xb)[:,1]

print(metrics.confusion_matrix(yb,pred0))


# In[176]:


import sklearn.metrics as metrics
fpr,tpr,threshold=metrics.roc_curve(yb,y_prob)
plt.plot(fpr,tpr)
plt.plot([0,1],[0,1])
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.show()


# In[75]:


from sklearn.model_selection import cross_val_score
score=cross_val_score(knn,Xa,ya,cv=10,scoring='accuracy')
score.mean()


# In[76]:


from sklearn.model_selection import GridSearchCV
k_range=range(1,31)
weight_option=['uniform','distance']
param_grid = {'n_neighbors':k_range , 'weights':weight_option}
grid=GridSearchCV(knn,param_grid,cv=10,scoring='accuracy')
grid.fit(Xa,ya)
print(grid.best_score_)
print(grid.best_params_)


# # Logistic Regression

# In[77]:


from sklearn.linear_model import LogisticRegression
logreg=LogisticRegression(random_state=1,multi_class='auto',solver='newton-cg',penalty='l2')
logreg.fit(Xa,ya)
pred1=logreg.predict(Xb)
print(metrics.classification_report(yb,pred1))
print(metrics.confusion_matrix(yb,pred1))


# In[78]:


y_prob=logreg.predict_proba(Xb)[:,1]
pred1=binarize([y_prob],0.50)[0]
print(metrics.confusion_matrix(yb,pred1))


# In[79]:


from sklearn.model_selection import cross_val_score
score=cross_val_score(logreg,Xa,ya,cv=10,scoring='accuracy')
score.mean()


# In[224]:


state=np.arange(1,50,1)
option=['auto','ovr']
weight_option=['newton-cg','lbfgs','liblinear','sag','saga']
param_grid = {'multi_class': option , 'solver':weight_option,'penalty':['l2']}
grid=GridSearchCV(logreg,param_grid,cv=10,scoring='accuracy')
grid.fit(Xa,ya)
print(grid.best_score_)
print(grid.best_params_)


# # RandomForestClassifier

# In[80]:


from sklearn.ensemble import RandomForestClassifier
log=RandomForestClassifier(random_state=5)
log.fit(Xa,ya)
pred2=log.predict(Xb)
print(metrics.classification_report(yb,pred2))
print(metrics.confusion_matrix(yb,pred2))


# In[81]:


y_prob=log.predict_proba(Xb)[:,1]
pred2=binarize([y_prob],0.60)[0]
print(metrics.confusion_matrix(yb,pred2))


# In[77]:


from sklearn.model_selection import cross_val_score
score=cross_val_score(log,Xa,ya,cv=10,scoring='accuracy')
score.mean()


# # DecisionTreeClassifier

# In[82]:


from sklearn.tree import DecisionTreeClassifier
dtc=DecisionTreeClassifier(random_state=0,splitter='random',max_features='auto')
dtc.fit(Xa,ya)
pred3=dtc.predict(Xb)
print(metrics.accuracy_score(yb,pred3))
print(metrics.confusion_matrix(yb,pred3))


# In[83]:


from sklearn.model_selection import cross_val_score
score=cross_val_score(dtc,Xa,ya,cv=10,scoring='accuracy')
score.mean()


# In[84]:


option=['gini','entropy']
weight_option=['auto','sqrt','log2']
param_grid = {'criterion': option , 'max_features':weight_option,'splitter':['best','random']}
grid=GridSearchCV(dtc,param_grid,cv=10,scoring='accuracy')
grid.fit(Xa,ya)
print(grid.best_score_)
print(grid.best_params_)


# # GradientBoostingClassifier

# In[208]:


from sklearn.ensemble import GradientBoostingClassifier
gbc=GradientBoostingClassifier(random_state=31)
gbc.fit(Xa,ya)
pred4=gbc.predict(Xa)
print(metrics.classification_report(ya,pred4))
print(metrics.confusion_matrix(ya,pred4))


# In[178]:


y_prob=gbc.predict_proba(Xb)[:,1]
pred4=binarize([y_prob],0.55)[0]
print(metrics.confusion_matrix(yb,pred4))


# In[179]:


from sklearn.model_selection import cross_val_score
score=cross_val_score(knn,Xa,ya,cv=10,scoring='accuracy')
score.mean()


# In[ ]:


param_grid = {'criterion':['friedman_mse','mse','mae'] , 'loss':['deviance','exponential'],'max_features':['auto','sqrt','log2'],'learning_rate':[0.1,0.01,0.001,1]}
grid=GridSearchCV(gbc,param_grid,cv=10,scoring='accuracy')
grid.fit(Xa,ya)
print(grid.best_score_)
print(grid.best_params_)


# # AdaBoostClassifier

# In[72]:


from sklearn.ensemble import AdaBoostClassifier
abc=AdaBoostClassifier(random_state=0,algorithm='SAMME',n_estimators=1000,learning_rate=0.1)
abc.fit(Xa,ya)
pred5=abc.predict(Xb)
print(metrics.classification_report(yb,pred5))
print(metrics.confusion_matrix(yb,pred5))


# In[73]:


best=pd.Series(abc.feature_importances_,index=Xa.columns)
best.nlargest(7).plot(kind='bar')
best.nlargest(7)


# In[74]:


y_prob5=abc.predict_proba(Xb)[:,1]
pred5=binarize([y_prob5],0.525)[0]
print(metrics.confusion_matrix(yb,pred5))
print(metrics.classification_report(yb,pred5))


# In[75]:


import sklearn.metrics as metrics
fpr,tpr,threshold=metrics.roc_curve(yb,y_prob5)
plt.plot(fpr,tpr)
plt.plot([0,1],[0,1])
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.show()


# In[76]:


plt.hist(y_prob5,bins=4)
plt.show()


# In[89]:


from sklearn.model_selection import cross_val_score
score=cross_val_score(abc,Xa,ya,cv=10,scoring='accuracy')
score.mean()


# In[120]:


param_grid = {'algorithm':['SAMME','SAMME.R'] ,'n_estimators':[10,100,1000,50],'learning_rate':[0.1,0.01,0.001,1]}
grid=GridSearchCV(abc,param_grid,cv=10,scoring='accuracy')
grid.fit(Xa,ya)
print(grid.best_score_)
print(grid.best_params_)


# # BaggingClassifier

# In[90]:


from sklearn.ensemble import BaggingClassifier
bc=BaggingClassifier(AdaBoostClassifier(random_state=0,algorithm='SAMME',n_estimators=10,learning_rate=0.1))
bc.fit(Xa,ya)
pred6=bc.predict(Xb)
print(metrics.accuracy_score(yb,pred6))


# In[91]:


from sklearn.preprocessing import binarize
y_prob=bc.predict_proba(Xb)[:,1]
pred6=binarize([y_prob],0.50)[0]

print(metrics.confusion_matrix(yb,pred6))


# # Stacking Classifier

# In[92]:


from mlxtend.classifier import StackingClassifier
scc=StackingClassifier(classifiers=[gbc],meta_classifier=logreg)
scc.fit(Xa,ya)
pred7=scc.predict(Xb)
print(metrics.accuracy_score(yb,pred7))


# In[93]:


y_prob=scc.predict_proba(Xb)[:,1]
pred7=binarize([y_prob],0.55)[0]

print(metrics.confusion_matrix(yb,pred7))


# # Support Vector Machine

# In[94]:


from sklearn.svm import SVC
svc=SVC()
svc.fit(Xa,ya)
pred8=svc.predict(Xb)
print(metrics.classification_report(yb,pred8))
print(metrics.confusion_matrix(yb,pred8))


# # Naive Bayes

# In[95]:


from sklearn.naive_bayes  import GaussianNB
gb=GaussianNB()
gb.fit(Xa,ya)
pred9=gb.predict(Xb)
print(metrics.classification_report(yb,pred9))
print(metrics.confusion_matrix(yb,pred9))


# We tried submitting our results with many different combinations , as shown by our dataframe
# 
# And....
# 
# Here is our best list and the best model

# As you may notice , we've got the best result when our model type is Gradient Boosting and List type is : 1

# # Preparing testing dataset for final submission

# In[180]:


test.isnull().sum()


# In[181]:


dict={'Yes':'Male','No':'Female'}
test['Gender'].fillna(test['Married'].map(dict),inplace=True)


# In[182]:


test['Credit_History'].fillna(2,inplace=True)


# In[183]:


test['Self_Employed'].fillna('No',inplace=True)


# In[184]:


test['Dependents'].replace('3+',3,inplace=True)
test['Dependents'].fillna('0',inplace=True)
test['Loan_Amount_Term'].fillna(test['Loan_Amount_Term'].mode()[0],inplace=True)
test['LoanAmount'].fillna(test['LoanAmount'].median(),inplace=True)


# In[185]:


test['ApplicantIncome']=test['ApplicantIncome'].where(test['ApplicantIncome']>0,test['ApplicantIncome'].mean())


# In[186]:


test['log_appincome']=np.log(test['ApplicantIncome'])
test['sqrt_coappincome']=np.sqrt(test['CoapplicantIncome'])


# In[187]:


test['loanamt']=pd.cut(test['LoanAmount'],bins=[27000,101000,157500,560000],labels=['low','avg','high'])


# In[188]:


test['Total_Income']=test['ApplicantIncome']+test['CoapplicantIncome']
test['Total_Income_bin']=pd.cut(test['Total_Income'],bins=[2082,4161.5,6980,72530],labels=['low','avg','high'])


# In[189]:


test.describe()


# In[190]:


def set_scr(test):
    if test["Total_Income_bin"] == "low" and test["loanamt"] == "high":
        return 1
    elif test["Total_Income_bin"] == "avg" and test["loanamt"] == "high":
        return 2
    elif test["Total_Income_bin"] == "low" and test["loanamt"] == "avg":
        return 4
    elif test["Total_Income_bin"] == "high" and test["loanamt"] == "high":
        return 8
    elif test["Total_Income_bin"] == "avg" and test["loanamt"] == "avg":
        return 7
    elif test["Total_Income_bin"] == "low" and test["loanamt"] == "low":
        return 3
    elif test["Total_Income_bin"] == "high" and test["loanamt"] == "avg":
        return 6
    elif test["Total_Income_bin"] == "avg" and test["loanamt"] == "low":
        return 5
    else :
        return 9
test['score']=test.apply(set_scr,axis=1)
test.head()


# In[191]:


test['EMI']=(test['LoanAmount']*(7.85/1200)*((1+(7.85/1200))**test['Loan_Amount_Term']))/((1+(7.85/1200))**(test['Loan_Amount_Term']-1))

test['Risk_Percent']=(test['EMI']/test['ApplicantIncome'])


# In[192]:


def num(test):
    if test['CoapplicantIncome']==0:
        return 0
    elif test['CoapplicantIncome']>=0 and test['CoapplicantIncome']<=1025:
        return 1
    elif test['CoapplicantIncome']>1025 and test['CoapplicantIncome']<=2430.5:
        return 1
    elif test['CoapplicantIncome']>2430.5 and test['CoapplicantIncome']<=25000:
        return 2
test['coapplicant']=test.apply(num,axis=1)


# In[193]:


# So we also need to remove skewness from Total Income

test['log_total_income']=np.log(test['Total_Income'])


# In[194]:


from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
dict5={'Male':1,'Female':0}
test['Gender']=test['Gender'].map(dict5)
test['Married']=le.fit_transform(test['Married'])
dict6={'Yes':1,'No':0}
test['Self_Employed']=test['Self_Employed'].map(dict6)
test['loanamt']=test['loanamt'].astype('object')
test['Total_Income_bin']=test['Total_Income_bin'].astype('object')
dict={'low':1,'avg':2,'high':3}
dict1={'low':1,'avg':2,'high':3}
dict2={'Graduate':1,'Not Graduate':0}
dict3={"Rural":0,'Urban':2,'Semiurban':1}
dict4={6:0,8:1,12:2,36:3,60:4,84:5,120:6,180:7,240:8,300:9,350:10,360:11,480:12}
test['Loan_Amount_Term']=test['Loan_Amount_Term'].map(dict4)
test['loanamt']=test['loanamt'].map(dict)
test['Education']=test['Education'].map(dict2)
test['Total_Income_bin']=test['Total_Income_bin'].map(dict1)
test['Property_Area']=test['Property_Area'].map(dict3)


# In[195]:


test.describe()


# In[196]:


from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
train[['ApplicantIncome','LoanAmount','Total_Income','EMI','log_appincome','Risk_Percent','log_total_income','sqrt_coappincome']]=scaler.fit_transform(train[['ApplicantIncome','LoanAmount','Total_Income','EMI','log_appincome','Risk_Percent','log_total_income','sqrt_coappincome']])


# In[197]:


from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
test[['ApplicantIncome','LoanAmount','EMI','Total_Income','log_appincome','Risk_Percent','log_total_income','sqrt_coappincome']]=scaler.fit_transform(test[['ApplicantIncome','LoanAmount','EMI','Total_Income','log_appincome','Risk_Percent','log_total_income','sqrt_coappincome']])


# In[198]:


XX=test[['Gender', 'Married', 'Dependents', 'Education',
       'Self_Employed', 'ApplicantIncome', 'LoanAmount','Loan_Amount_Term',
     'Credit_History', 'Property_Area']]


# In[199]:


X1=train[['Gender', 'Married', 'Dependents', 'Education',
       'Self_Employed', 'ApplicantIncome', 'LoanAmount','Loan_Amount_Term',
     'Credit_History', 'Property_Area']]
y1=train['Loan_Status']


# In[200]:


XX.isnull().sum()


# In[109]:


from sklearn.ensemble import AdaBoostClassifier
abc=AdaBoostClassifier(random_state=0,algorithm='SAMME',n_estimators=1000,learning_rate=0.1)
abc.fit(X1,y1)
predddd=abc.predict(XX)
predddd=pd.DataFrame(predddd,columns=['Loan_Status'])


# In[202]:


from sklearn.ensemble import GradientBoostingClassifier
gbc=GradientBoostingClassifier(random_state=31)
gbc.fit(X1,y1)
log_pred=gbc.predict(XX)
log_pred=pd.DataFrame(log_pred,columns=['Loan_Status'])


# In[203]:


final=pd.merge(test,gbc,left_index=True,right_index=True,how='outer')
final.head()


# In[204]:


final=final[['Loan_ID','Loan_Status']]
final['Loan_Status'].replace(1,'Y',inplace=True)
final['Loan_Status'].replace(0,'N',inplace=True)
final['Loan_Status'].value_counts()


# As you may notice , we've got the best result when our model type is Gradient Boosting and List type is : 1

# In[206]:


final.to_csv('C:\\Users\\User-1\\Desktop\\final4.csv')


# Lastly , We have a current rank of 45 for this competition .
# 
# It is an ongoing competition , so our rank would ofcourse decrease .
# 
# But as and when required , we would like to learn new skills and grow each and everyday . And therefore , we would bring our new skills to test by aiming for a better rank in the future

# In[ ]:




