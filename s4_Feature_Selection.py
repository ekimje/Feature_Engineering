from turtle import pd

from sklearn.pipeline import Pipeline
from s3_Pipeline import df_most_one_standard, df_dropna_one_standard, df_most_one_robust, df_dropna_one_robust
from s3_Pipeline import df_most_label_standard, df_most_label_robust, df_dropna_label_standard, df_dropna_label_robust
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.feature_selection import SelectKBest
from sklearn.model_selection import train_test_split
from s3_Pipeline import x_drop, y_drop

pipelines = {
    # Feature Selection 제거 전
    'most_standard_one_random_before': Pipeline([
        ('process', df_most_one_standard),
        ('model', RandomForestClassifier(random_state=42))
    ]),

    'most_standard_one_xgb_before': Pipeline([
        ('process', df_most_one_standard),
        ('model', XGBClassifier(random_state=42))
    ]),

    'most_standard_label_random_before': Pipeline([
        ('process', df_most_label_standard),
        ('model', RandomForestClassifier(random_state=42))
    ]),

    'most_standard_label_xgb_before': Pipeline([
        ('process', df_most_label_standard),
        ('model', XGBClassifier(random_state=42))
    ]),

    'most_robust_one_random_before': Pipeline([
        ('process', df_most_one_robust),
        ('model', RandomForestClassifier(random_state=42))
    ]), 

    'most_robust_one_xgb_before': Pipeline([
        ('process', df_most_one_robust),
        ('model', XGBClassifier(random_state=42))
    ]),

    'most_robust_label_random_before': Pipeline([
        ('process', df_most_label_robust),   
        ('model', RandomForestClassifier(random_state=42))
    ]),

    'most_robust_label_xgb_before': Pipeline([
        ('process', df_most_label_robust),    
        ('model', XGBClassifier(random_state=42))
    ]),

    'dropna_one_random_before': Pipeline([
        ('process', df_dropna_one_standard),
        ('model', RandomForestClassifier(random_state=42))
    ]),

    'dropna_one_xgb_before': Pipeline([
        ('process', df_dropna_one_standard),
        ('model', XGBClassifier(random_state=42))
    ]),

    'dropna_label_random_before': Pipeline([
        ('process', df_dropna_label_standard),
        ('model', RandomForestClassifier(random_state=42))
    ]),

    'dropna_label_xgb_before': Pipeline([
        ('process', df_dropna_label_standard),
        ('model', XGBClassifier(random_state=42))
    ]),

    'dropna_label_robust_random_before': Pipeline([
        ('process', df_dropna_label_robust),    
        ('model', RandomForestClassifier(random_state=42))
    ]),

    'dropna_label_robust_xgb_before': Pipeline([
        ('process', df_dropna_label_robust),    
        ('model', XGBClassifier(random_state=42))
    ]),

    'dropna_one_robust_random_before': Pipeline([
        ('process', df_dropna_one_robust),
        ('model', RandomForestClassifier(random_state=42))
    ]),

    'dropna_one_robust_xgb_before' : Pipeline([
        ('process', df_dropna_one_robust),
        ('model', XGBClassifier(random_state=42))
    ]),

    # Feature Selection 제거 후
    'most_standard_one_random_after': Pipeline([
        ('process', df_most_one_standard),
        ('FeatureSelection', SelectKBest(k=10)), # Feature Selection 단계 추가
        ('model', RandomForestClassifier(random_state=42))
    ]),
    
    'most_standard_one_xgb_after': Pipeline([
        ('process', df_most_one_standard),
        ('FeatureSelection', SelectKBest(k=10)),
        ('model', XGBClassifier(random_state=42))
    ]),

    'most_standard_label_random_after': Pipeline([
        ('process', df_most_label_standard),
        ('FeatureSelection', SelectKBest(k=10)),
        ('model', RandomForestClassifier(random_state=42))
    ]),

    'most_standard_label_xgb_after': Pipeline([
        ('process', df_most_label_standard),
        ('FeatureSelection', SelectKBest(k=10)),
        ('model', XGBClassifier(random_state=42))
    ]),

    'most_robust_one_random_after': Pipeline([
        ('process', df_most_one_robust),
        ('FeatureSelection', SelectKBest(k=10)),
        ('model', RandomForestClassifier(random_state=42))
    ]),

    'most_robust_one_xgb_after': Pipeline([
        ('process', df_most_one_robust),
        ('FeatureSelection', SelectKBest(k=10)),
        ('model', XGBClassifier(random_state=42))
    ]),

    'most_robust_label_random_after': Pipeline([
        ('process', df_most_label_robust),
        ('FeatureSelection', SelectKBest(k=10)),    
        ('model', RandomForestClassifier(random_state=42))
    ]),

    'most_robust_label_xgb_after': Pipeline([
        ('process', df_most_label_robust),    
        ('FeatureSelection', SelectKBest(k=10)),
        ('model', XGBClassifier(random_state=42))
    ]),

    'dropna_one_random_after': Pipeline([
        ('process', df_dropna_one_standard),
        ('FeatureSelection', SelectKBest(k=10)),
        ('model', RandomForestClassifier(random_state=42))
    ]),

    'dropna_one_xgb_after': Pipeline([
        ('process', df_dropna_one_standard),
        ('FeatureSelection', SelectKBest(k=10)),
        ('model', XGBClassifier(random_state=42))
    ]),

    'dropna_label_random_after': Pipeline([
        ('process', df_dropna_label_standard),
        ('FeatureSelection', SelectKBest(k=10)),
        ('model', RandomForestClassifier(random_state=42))
    ]),

    'dropna_label_xgb_after': Pipeline([
        ('process', df_dropna_label_standard),
        ('FeatureSelection', SelectKBest(k=10)),
        ('model', XGBClassifier(random_state=42))
    ]),

    'dropna_label_robust_random_after': Pipeline([
        ('process', df_dropna_label_robust),   
        ('FeatureSelection', SelectKBest(k=10)), 
        ('model', RandomForestClassifier(random_state=42))
    ]),

    'dropna_label_robust_xgb_after': Pipeline([
        ('process', df_dropna_label_robust),  
        ('FeatureSelection', SelectKBest(k=10)),  
        ('model', XGBClassifier(random_state=42))
    ]),

    'dropna_one_robust_random_after': Pipeline([
        ('process', df_dropna_one_robust),
        ('FeatureSelection', SelectKBest(k=10)),
        ('model', RandomForestClassifier(random_state=42))
    ]),

    'dropna_one_robust_xgb_after': Pipeline([
        ('process', df_dropna_one_robust),
        ('FeatureSelection', SelectKBest(k=10)),
        ('model', XGBClassifier(random_state=42))
    ])
}

x_train, x_test, y_train, y_test = train_test_split(x_drop, y_drop, test_size=0.2, random_state=42)

result = []

for name, pipeline in pipelines.items():
    if 'dropna' in name:
        pipeline.fit(x_train, y_train)
        score = pipeline.score(x_test, y_test)
    else:
        pipeline.fit(x_train, y_train)
        score = pipeline.score(x_test, y_test)
    
    result.append({
        'Experiment': name,
        'Score': score
    })

result_df = pd.DataFrame(result)
result_df.to_csv('Gold\\Feature_Selection_Results.csv', index=False)