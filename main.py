#선택한 데이터: Adult Income Dataset
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn as sklearn

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
print(df.shape) # (32561,15)
print(df.info()) # NULL 없음 확인.

# 타겟 변수 정의
x = df.drop('income', axis=1)
y = df['income']
print(y.value_counts())

# 2. 탐색적 데이터 분석
#결측치 비율 분석 : null 값이 존재한다면 결측치 분석
null_value = df.isnull().mean() * 100
print("--결측치 비율--")
print(null_value)

#이상치 탐색 : 수치형 변수의 이상치 탐색
outlier_result=[] # 이상치 비율 저장
numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
for col in numeric_cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    
    #이상치 기준 설정
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers_count= df[(df[col]<lower_bound) | (df[col]>upper_bound)].shape[0]
    ratio = outliers_count / len(df) * 100
    outlier_result.append([col, outliers_count, ratio])

outlier_df = pd.DataFrame(outlier_result, columns=['column','outliers_count','outliers_ratio'])
print(outlier_df)   
#변수 분포 시각화 - histogram을 이용해 수치형 변수의 분포를 시각화, Boxplot을 이용해 이상치 시각화
numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
df[numeric_cols].hist(figsize=(12,8),bins=20)
plt.show()

fig, ax = plt.subplots(2,3,figsize=(12,8))

ax = ax.flatten()
for i, col in enumerate(numeric_cols):
    sns.boxplot(x=df[col], ax=ax[i])

plt.show()

#상관관계 분석 - heatmap 확인(수치형 변수 간의 상관관계 분석)
correlation = df[numeric_cols].corr()
sns.heatmap(correlation, annot=True, cmap='coolwarm')
plt.xticks(rotation=0)
plt.show()

#타겟 변수 분포 확인 - 타겟 변수의 클래스 불균형 여부 확인
#타겟 변수 'income' 범주형 변수. 특정 소득을 초과하는지 아닌지를 확인하여 분류하는 타겟 변수. 각 클래스가 몇 개 존재하는가? --> countplot으로 결정
sns.countplot(x='income',data=df)
plt.show()