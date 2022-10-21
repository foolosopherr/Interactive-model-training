import pycaret.classification as clf

def create_pycaret_model(df, 
                         target, 
                         ignore_features, 
                         models_to_use,
                         train_size,
                         polynomial_features,
                         polynomial_degree,
                         remove_multicollinearity,
                         multicollinearity_threshold,
                         remove_outliers,
                         outliers_threshold,
                         normalize,
                         normalize_method,
                         data_split_shuffle,
                         data_split_stratify,
                         fold_strategy,
                         fold,
                         fold_shuffle,
                         session_id,
                         optimize,
                         n_iter):
    
    s = clf.setup(data=df,
                  target=target,
                  ignore_features=ignore_features,
                  train_size=train_size,
                  polynomial_features=polynomial_features,
                  polynomial_degree=polynomial_degree,
                  remove_multicollinearity=remove_multicollinearity,
                  multicollinearity_threshold=multicollinearity_threshold,
                  remove_outliers=remove_outliers,
                  outliers_threshold=outliers_threshold,
                  normalize=normalize,
                  normalize_method=normalize_method,
                  data_split_shuffle=data_split_shuffle,
                  data_split_stratify=data_split_stratify,
                  fold_strategy=fold_strategy,
                  fold=fold,
                  fold_shuffle=fold_shuffle,
                  session_id=session_id,
                  html=False,
                  verbose=False)
    
    model = clf.compare_models(include=models_to_use)
    pulled_df_model = clf.pull().astype('str')
    
    tuned_model = clf.tune_model(model, choose_better=True, n_iter=n_iter, optimize=optimize)
    pulled_df_tuned_model = clf.pull().reset_index().astype('str').set_index('Fold')
    
    return tuned_model, pulled_df_model, pulled_df_tuned_model