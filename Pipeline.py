#특성 공학 파이프라인 구현
# 결측치 처리 비교
# 데이터 특성상 최소/최대 아닌 Most_Frequent, Drop Na 로 결측치 처리 비교
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.preprocessing import StandardScaler, RobustScaler


df = pd.read_csv('Bronze\\adult.csv')
# 컬럼 분리
x = df.drop('income', axis=1) # 입력 데이터
y = df['income'] # 정답 데이터

# # 결측치 처리 비교
# # Most_Frequent Imputation
# most_imputer = SimpleImputer(strategy='most_frequent')

# x_most = pd.DataFrame(most_imputer.fit_transform(x), columns=x.columns)

# # Drop Na
# df_dropna = df.dropna().copy()

# x_dropna = df_dropna.drop('income', axis=1)
# y_dropna = df_dropna['income']

# # 범주형 인코딩 비교
# # One-Hot Encoding
# oh = OneHotEncoder()
# drop_of = oh.fit_transform(x_dropna.select_dtypes(include=['object']))
# most_of = oh.fit_transform(x_most.select_dtypes(include=['object']))

# # Label Encoding
# le = LabelEncoder()
# for col in x_dropna.select_dtypes(include=['object']).columns:
#     x_dropna[col] = le.fit_transform(x_dropna[col])
# for col in x_most.select_dtypes(include=['object']).columns:
#     x_most[col] = le.fit_transform(x_most[col])

# # 스케일링 비교 (StandardScaler, RobustScaler)
# # 1. StandardScaler
# scaler_most = StandardScaler()
# scaler_dropna = StandardScaler()
# x_dropna_scaled = scaler_dropna.fit_transform(x_dropna)
# x_most_scaled = scaler_most.fit_transform(x_most)
# # 2. RobustScaler
# robust_dropna = RobustScaler()
# robust_most = RobustScaler()
# x_dropna_robust = robust_dropna.fit_transform(x_dropna)
# x_most_robust = robust_most.fit_transform(x_most)

# 파이프라인 구축
df_most_one_standard = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder()),
    ('scaler', StandardScaler())
])

df_dropna_one_standard = Pipeline([
    ('dropna', 'passthrough'), # Drop Na는 별도의 단계로 처리
    ('encoder', OneHotEncoder()),
    ('scaler', StandardScaler())
])

df_most_one_robust = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder()),
    ('scaler', RobustScaler())
])

df_dropna_one_robust = Pipeline([
    ('dropna', 'passthrough'), # Drop Na는 별도의 단계로 처리
    ('encoder', OneHotEncoder()),
    ('scaler', RobustScaler())
])

df_most_label_standard = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', LabelEncoder()), # Label Encoding은 별도의 단계로 처리
    ('scaler', StandardScaler())
])

df_most_label_robust = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', LabelEncoder()), # Label Encoding은 별도의 단계로 처리
    ('scaler', RobustScaler())
]) 

df_dropna_label_standard = Pipeline([
    ('dropna', 'passthrough'), # Drop Na는 별도의 단계로 처리
    ('encoder', LabelEncoder()), # Label Encoding은 별도의 단계로 처리 
    ('scaler', StandardScaler())
])

df_dropana_label_robust = Pipeline([
    ('dropna', 'passthrough'), # Drop Na는 별도의 단계로 처리
    ('encoder', LabelEncoder()), # Label Encoding은 별도의 단계로 처리 
    ('scaler', RobustScaler())
])

# 파생 변수 생성 (스케일링 먼저 수행하면 변수 간의 단위 차이로 인한 편향 방지 가능)
# 순자본 생성
x['net_capital'] = x['capital-gain'] - x['capital-loss']
# 근무 시간 구분
x['work_intensity'] = pd.cut(x['hours-per-week'], bins=[0, 20, 40, 60, 100], labels=['part','full','overtime','extreme'])

# 데이터 저장
x.to_csv('Silver\\adult_features.csv', index=False)