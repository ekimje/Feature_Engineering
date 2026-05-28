#선택한 데이터: Adult Income Dataset
import pandas as pd
# 1. 데이터 준비

with open('File\\adult.names','r') as f:
    print(f.read())
    
# 타겟변수는 income으로, 2개의 클래스(<=50K, >50K)로 구성되어 있습니다.
# Prediction task is to determine whether a person makes over 50K a year.

columns = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status',
              'occupation', 'relationship', 'race','sex','capital-gain','capital-loss',
              'hours-per-week','native-country','income']

df = pd.read_csv('File\\adult.data', names=columns, na_values=' ?')
print(df.head())
print(df.shape) 
print(df.info())

# 타겟 변수 정의
x = df.drop('income', axis=1)
y = df['income']
print(y.value_counts())