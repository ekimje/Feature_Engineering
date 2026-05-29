#특성 공학 파이프라인 구현
# 결측치 처리 비교
# 데이터 특성상 최소/최대 아닌 Most_Frequent, Drop Na 로 결측치 처리 비교
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder


df = pd.read_csv('Bronze\\adult.csv')
# 데이터 분리
x = df.drop('income', axis=1)
y = df['income']
print(y)

# 결측치 처리 비교
# Most_Frequent Imputation
most_imputer = SimpleImputer(strategy='most_frequent')

x_most = pd.DataFrame(most_imputer.fit_transform(y), columns='income')

# Drop Na
df_dropna = df.dropna().copy()

# 범주형 인코딩 비교
# One-Hot Encoding
oh = OneHotEncoder()
drop_of = oh.fit_transform(df_dropna.select_dtypes(include=['object']))
most_of = oh.fit_transform(df_most.select_dtypes(include=['object']))

# Label Encoding
le = LabelEncoder()
for col in df_dropna.select_dtypes(include=['object']).columns:
    df_dropna[col] = le.fit_transform(df_dropna[col])
for col in df_most.select_dtypes(include=['object']).columns:
    df_most[col] = le.fit_transform(df_most[col])

