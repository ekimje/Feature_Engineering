import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from s3_Pipeline import experiment_configs, Preprocessor, x, x_drop, y, y_drop

df = pd.read_csv('Silver\\adult_features.csv', index_col=False)
TOP_N_FEATURES = 10

def build_xgb_classifier()-> XGBClassifier:
    return XGBClassifier(eval_metric='logloss',random_state=42)

models = {
    'RandomForest':RandomForestClassifier(random_state=42),
    'XGBoost':build_xgb_classifier(),
}

def get_train_test_data(experiment_name:str):
    config = experiment_configs[experiment_name]
    if config['dropna']:
        return train_test_split(x_drop, y_drop, test_size=0.2, random_state=42,stratify=y_drop)
    else:
        return train_test_split(x, y, test_size=0.2, random_state=42,stratify=y)
    
def build_pipeline(experiment_name:str, model_name:str, use_feature_selection:bool)->Pipeline:
    steps = [('processor', Preprocessor[experiment_name])]
    if use_feature_selection:
        steps.append(('feature_selection', SelectKBest(f_classif, k=TOP_N_FEATURES)))
    steps.append(('classifier', models[model_name]))
    return Pipeline(steps)

def predict_positive_probability(pipeline:Pipeline, X_test:pd.DataFrame)->np.ndarray:
    if hasattr(pipeline.named_steps['classifier'], 'predict_proba'):
        return pipeline.predict_proba(X_test)[:, 1]
    else:
        return pipeline.decision_function(X_test)
    
def get_transformed_feature_names(pipeline:Pipeline)->np.array:
    process = pipeline.named_steps['processor']
    return process.get_feature_names_out()

def get_selected_features(pipeline:Pipeline)->pd.DataFrame:
    if 'feature_selection' not in pipeline.named_steps:
        return pd.DataFrame()
    
    feature_names = get_transformed_feature_names(pipeline)
    selector = pipeline.named_steps['feature_selection']
    selected_names = feature_names[selector.get_support()]
    selected_scores = selector.scores_[selector.get_support()]
    return pd.DataFrame({'feature': selected_names, 'score': selected_scores}).sort_values(by='score', ascending=False)

def get_random_forest_importance(pipeline:Pipeline)->pd.DataFrame:
    if pipeline.named_steps['model'].__class__.__name__ != 'RandomForestClassifier':
        return pd.DataFrame()
    
    feture_names = get_transformed_feature_names(pipeline)
    if 'feature_selection' in pipeline.named_steps:
        selector = pipeline.named_steps['feature_selection']
        feature_names = feature_names[selector.get_support()]
        
    importances = pipeline.named_steps['model'].feature_importances_
    return pd.DataFrame({'feature': feature_names, 'importance': importances}).sort_values('Importance', ascending=False).head(TOP_N_FEATURES)

def evaluate_pipeline(pipeline:Pipeline, x_test:pd.DataFrame, y_test:pd.Series)->dict[str,float]:
    y_pred = pipeline.predict(x_test)
    y_prob = predict_positive_probability(pipeline, x_test)
    
    return {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1_score': f1_score(y_test, y_pred),
        'roc_auc': roc_auc_score(y_test, y_prob)
    }

results = []
selected_features_rows = []
importance_rows = []

for experiment_name, config in experiment_configs.items():
    x_train, x_test, y_train, y_test = get_train_test_data(experiment_name)
    
    for model_name in models:
        for use_feature_selection in [False, True]:
            pipeline = build_pipeline(experiment_name, model_name, use_feature_selection)
            pipeline.fit(x_train, y_train)
            metrics = evaluate_pipeline(pipeline, x_test, y_test)
            
            results.append({
                'experiment': experiment_name,
                'model': model_name,
                '결측치 처리':config['missing'],
                '인코딩': config['encoding'],
                '스케일링': config['scaling'],
                'feature_selection': use_feature_selection,
                **metrics
            })
            
            selected_features = get_selected_features(pipeline)
            for rank, row in enumerate(selected_features.itertuples(index=False), start=1):
                selected_features_rows.append({
                    'experiment': experiment_name,
                    'model': model_name,
                    'feature': row.feature,
                    'selectKBest_score':row.selectKBest_score,
                    'rank': rank
                })
            
            feature_importance = get_random_forest_importance(pipeline)
            for rank, row in enumerate(feature_importance.itertuples(index=False), start=1):
                importance_rows.append({
                    'experiment': experiment_name,
                    'feature selection':'O' if use_feature_selection else 'X',
                    'rank': rank,
                    'feature': row.feature,
                    'importance': row.importance
                })

results_df = pd.DataFrame(results)
selected_features_df = pd.DataFrame(selected_features_rows)
feature_importance_df = pd.DataFrame(importance_rows)

results_df.to_csv('Gold\\experiment_results.csv', index=False)
selected_features_df.to_csv('Gold\\selected_features.csv', index=False)
feature_importance_df.to_csv('Gold\\feature_importance.csv', index=False)