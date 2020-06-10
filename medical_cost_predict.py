import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import seaborn as sns
from sklearn import metrics
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import PolynomialFeatures

data_file = "../统计学/insurance.csv"
data_frame = pd.read_csv(data_file) #读取数据集
#data_frame.head()   #观察读取的数据是否正确
#print(data_frame.head())
#data_frame.describe()  #给出各项数据的统计信息
#print(data_frame.describe())

#############对分类型数据如sex smoker region进行编码##########
#smoker
LabelEncoder_smoker = LabelEncoder()
data_frame.smoker = LabelEncoder_smoker.fit_transform(data_frame.smoker)
#sex
LabelEncoder_sex = LabelEncoder()
data_frame.sex = LabelEncoder_sex.fit_transform(data_frame.sex)
#region
LabelEncoder_region = LabelEncoder()
data_frame.region = LabelEncoder_region.fit_transform(data_frame.region)
#print(data_frame.head())

#按照与费用的相关性对各个要素进行排列
#print(data_frame.corr()['charges'].sort_values())
#用直方图表示平均有多少患者在治疗上花费了多少钱
'''
plt.hist(data_frame.charges,bins=10,alpha=0.5,histtype='bar',ec='black')
plt.title("Frequency Distribution of the charges")
plt.xlabel('Charges')
plt.ylabel('Frequency')
plt.show()
'''
raw_data_frame = pd.read_csv(data_file)
#不同地区对医疗花费的影响
'''
sns.boxenplot(x=raw_data_frame.region,y=raw_data_frame.charges,data=raw_data_frame)
plt.title("Medical charges per region")
plt.show()
'''
#吸烟与否对医疗花费的影响
'''
sns.boxplot(x=raw_data_frame.smoker,y=raw_data_frame.charges,data=raw_data_frame)
plt.title("Medical charges or Smokers and Non-Smokers")
plt.show()
'''
#进一步研究吸烟与否对医疗花费的影响
'''
f = plt.figure(figsize=(12,5))#设定plot的宽高
ax = f.add_subplot(121)
sns.distplot(data_frame[data_frame.smoker==1]['charges'],color='r',ax=ax)
ax.set_title('Medical charges for the smokers')
ax = f.add_subplot(122)
sns.distplot(data_frame[data_frame.smoker==0]['charges'],color='g',ax=ax)
ax.set_title('Medical charges for the non-smokers')
#plt.show()
'''
#性别对医疗花费的影响
'''
sns.boxplot(x=raw_data_frame.sex,y=raw_data_frame.charges,data=raw_data_frame)
plt.title("Charges by Gender")
plt.show()
'''

#探究吸烟者与不吸烟者在不同年龄段在医疗花费上的投入
'''
plt.subplot(1,2,1)
sns.distplot(data_frame[data_frame.smoker==1]['age'],color='red')
plt.title("Distribution of Smokers")
plt.subplot(1,2,2)
sns.distplot(data_frame[data_frame.smoker==0]['age'],color='blue')
plt.title("Distribution of Non-Smokers")
plt.show()
'''
#分析bmi与吸烟与否对医疗花费的影响
'''
sns.lmplot(x="bmi",y='charges',hue='smoker',data=raw_data_frame)
plt.show()
'''
#可视化回归模型
'''
sns.lmplot(x='age',y='charges',hue='smoker',data=raw_data_frame,palette='inferno_r')
plt.show()
'''
#建立回归模型
X = data_frame.iloc[:,:6].values
Y = data_frame.iloc[:,6].values
#这里用到多项式回归模型
poly_reg  = PolynomialFeatures(degree=2)
X_poly = poly_reg.fit_transform(X)
#分割数据集为训练集和测试集
X_train,X_test,Y_train,Y_test = train_test_split(X_poly,Y,test_size=0.25)
#训练模型
lin_reg = LinearRegression()
lin_reg  = lin_reg.fit(X_train,Y_train)
#预测准确率
#print('准确率为：',lin_reg.score(X_test,Y_test))
'''
X_poly_df = pd.DataFrame(X_poly,columns=poly_reg.get_feature_names())
print(X_poly_df.head())
X_poly_df.to_csv('coef.csv')
'''
#plt.scatter(X_test,Y_test,color='blue')
#plt.plot(X,lin_reg.predict(X_poly),color='red')

Y_predict = lin_reg.predict(X_test)
plt.figure()
plt.plot(range(len(Y_predict)),Y_predict,'r',label='charges_predict')
plt.plot(range(len(Y_predict)),Y_test,'b',label='charges_test')
plt.legend(loc='upper right')
plt.title('Polynomial Regression')
plt.show()

'''
print('coef: ',lin_reg.coef_)
print('intercept: ',lin_reg.intercept_)
'''
