#특성 공학 파이프라인 구현
# 결측치 처리 비교
# 데이터 특성상 최소/최대 아닌 Most_Frequent, Drop Na 로 결측치 처리 비교
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, LabelEncoder
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.compose import ColumnTransformer

def add_features(features:pd.DataFrame)->pd.DataFrame:
    # 파생 변수 생성 (스케일링 먼저 수행하면 변수 간의 단위 차이로 인한 편향 방지 가능)
    # 순자본 생성
    features['net_capital'] = features['capital-gain'] - features['capital-loss']
    # 근무 시간 구분
    features['work_intensity'] = pd.cut(features['hours-per-week'], bins=[0, 20, 40, 60, 100], labels=['part','full','overtime','extreme'])
    return features


def get_features_groups(features:pd.DataFrame)->tuple[list[str],list[str]]:
    categorical_cols = features.select_dtypes(include=['object','category']).columns.tolist()
    numerical_cols = features.select_dtypes(include=['int64', 'float64']).columns.tolist()
    return categorical_cols, numerical_cols

def make_preprocessor(numeric_imputation:str|None, categorical_imputation:str|None, encoding:str, scaling:str|None)->ColumnTransformer:
    numeric_steps=[]
    categorical_steps=[]
    
    if numeric_imputation is not None:
        numeric_steps.append(('imputer', SimpleImputer(strategy=numeric_imputation)))
    if scaling == 'standard':
        numeric_steps.append(('scaler', StandardScaler()))
    elif scaling == 'robust':
        numeric_steps.append(('scaler', RobustScaler()))
        
    if categorical_imputation is not None:
        categorical_steps.append(('imputer',SimpleImputer(strategy=categorical_imputation)))
    if encoding == 'onehot':
        categorical_steps.append(('encoder', OneHotEncoder(handle_unknown='ignore')))
    elif encoding == 'label':
        categorical_steps.append(('encoder', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)))
    else : raise ValueError(f"지원하지 않는 인코딩 방식입니다.{encoding}") 
    
    return ColumnTransformer([('num',Pipeline(numeric_steps), numeric_features),
                              ('cat',Pipeline(categorical_steps), categorical_features)])

experiment_configs = {
    'MostFreq_OneHot_Standard': {
        'missing':'Most Frequent',
        'numeric_imputation':'most_frequent',
        'categorical_imputation':'most_frequent',
        'encoding':'One-Hot',
        'encoding_key':'onehot',
        'scaling':'Standard',
        'scaling_key':'standard',
        'dropna': False,
    },
    
    'MostFreg_Label_Robust': {
        'missing':'Most Frequent',
        'numeric_imputation':'most_frequent',
        'categorical_imputation':'most_frequent',
        'encoding':'Label',
        'encoding_key':'label',
        'scaling':'Robust',
        'scaling_key':'robust',
        'dropna': False,
    },
    
    'DropNa_OneHot_Standard': {
        'missing':'Drop Na',
        'numeric_imputation':None,
        'categorical_imputation':None,
        'encoding':'One-Hot',
        'encoding_key':'onehot',
        'scaling':'Standard',
        'scaling_key':'standard',
        'dropna': True,
    },
    
    'DropNa_Label_Robust': {
        'missing':'Drop Na',
        'numeric_imputation':None,
        'categorical_imputation':None,
        'encoding':'Label',
        'encoding_key':'label',
        'scaling':'Robust',
        'scaling_key':'robust',
        'dropna': True,
    }
}

df = pd.read_csv('Bronze\\adult.csv')
# 컬럼 분리
x = df.drop('income', axis=1) # 입력 데이터
y_raw= df['income'] # 정답 데이터

x = add_features(x)
categorical_features, numeric_features = get_features_groups(x)

x_drop = x.dropna() # Drop Na는 별도의 단계로 처리
y_drop = y_raw.loc[x_drop.index] # Drop Na에 해당하는 인덱스에 맞춰 y도 정렬
le = LabelEncoder()
y = le.fit_transform(y_raw)
y_drop = le.transform(y_drop)

# 파이프라인 구성
Preprocessor={
    name:make_preprocessor(
        numeric_imputation = config['numeric_imputation'],
        categorical_imputation = config['categorical_imputation'],
        encoding = config['encoding_key'],
        scaling = config['scaling_key']
    )
    for name, config in experiment_configs.items()
}

x.to_csv('Silver\\adult_features.csv', index=False)