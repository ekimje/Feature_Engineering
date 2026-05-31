#특성 공학 파이프라인 구현
# 결측치 처리 비교
# 데이터 특성상 최소/최대 아닌 Most_Frequent, Drop Na 로 결측치 처리 비교
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.preprocessing import StandardScaler, RobustScaler


df = pd.read_csv('Bronze\\adult.csv')
# 컬럼 분리
x = df.drop('income', axis=1) # 입력 데이터
y = df['income'] # 정답 데이터

# 파생 변수 생성 (스케일링 먼저 수행하면 변수 간의 단위 차이로 인한 편향 방지 가능)
# 순자본 생성
x['net_capital'] = x['capital-gain'] - x['capital-loss']
# 근무 시간 구분
x['work_intensity'] = pd.cut(x['hours-per-week'], bins=[0, 20, 40, 60, 100], labels=['part','full','overtime','extreme'])

x_drop = x.dropna() # Drop Na는 별도의 단계로 처리
y_drop = y.loc[x_drop.index] # Drop Na에 해당하는 인덱스에 맞춰 y도 정렬

# 파이프라인 구축
df_most_one_standard = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(handle_unknown='ignore')),
    ('scaler', StandardScaler(with_mean=False)) # StandardScaler는 평균이 0이 되도록 조정하므로 with_mean=False로 설정하여 sparse matrix 지원
])

df_dropna_one_standard = Pipeline([
    ('encoder', OneHotEncoder(handle_unknown='ignore')),
    ('scaler', StandardScaler(with_mean=False))
])

df_most_one_robust = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(handle_unknown='ignore')),
    ('scaler', RobustScaler())
])

df_dropna_one_robust = Pipeline([
    ('encoder', OneHotEncoder(handle_unknown='ignore')),
    ('scaler', RobustScaler())
])

df_most_label_standard = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OrdinalEncoder()), # Label Encoding은 별도의 단계로 처리
    ('scaler', StandardScaler(with_mean=False))
])

df_most_label_robust = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OrdinalEncoder()), # Label Encoding은 별도의 단계로 처리
    ('scaler', RobustScaler())
]) 

df_dropna_label_standard = Pipeline([
    ('encoder', OrdinalEncoder()), # Label Encoding은 별도의 단계로 처리 
    ('scaler', StandardScaler(with_mean=False))
])

df_dropna_label_robust = Pipeline([
    ('encoder', OrdinalEncoder()), # Label Encoding은 별도의 단계로 처리 
    ('scaler', RobustScaler())
])

# 데이터 저장
x.to_csv('Silver\\adult_features.csv', index=False)