from pycaret.datasets import get_data
import pycaret.classification as clf
from train_func import create_pycaret_model
import streamlit as st


all_models = {'Logistic Regression': 'lr',
              'K Neighbors Classifier': 'knn',
              'Naive Bayes': 'nb',
              'Decision Tree Classifier': 'dt',
              'SVM - Linear Kernel': 'svm',
              'SVM - Radial Kernel': 'rbfsvm',
              'Gaussian Process Classifier': 'gpc',
              'MLP Classifier': 'mlp',
              'Ridge Classifier': 'ridge',
              'Random Forest Classifier': 'rf',
              'Quadratic Discriminant Analysis': 'qda',
              'Ada Boost Classifier': 'ada',
              'Gradient Boosting Classifier': 'gbc',
              'Linear Discriminant Analysis': 'lda',
              'Extra Trees Classifier': 'et',
              'Light Gradient Boosting Machine': 'lightgbm',
              'Dummy Classifier': 'dummy'}

all_plots = {'auc':'Area Under the Curve',
             'threshold':'Discrimination Threshold',
             'pr':'Precision Recall Curve',
             'confusion_matrix':'Confusion Matrix',
             'error':'Class Prediction Error',
             'class_report':'Classification Report',
             'boundary':'Decision Boundary',
             'rfe':'Recursive Feature Selection',
             'learning':'Learning Curve',
             'manifold':'Manifold Learning',
             'calibration':'Calibration Curve',
             'vc':'Validation Curve',
             'dimension':'Dimension Learning',
             'feature':'Feature Importance',
             'feature_all':'Feature Importance (All)',
             'parameter':'Model Hyperparameter',
             'lift':'Lift Curve',
             'gain':'Gain Chart',
             'tree':'Decision Tree',
             'ks':'KS Statistic Plot'}


all_datasets = ['iris','bank', 'blood', 'cancer', 'credit', 'diabetes', 'electrical_grid',
                'employee', 'heart', 'heart_disease', 'hepatitis', 'income',
                'juice', 'nba', 'wine', 'telescope', 'titanic',
                'us_presidential_election_results', 'glass', 'poker',
                'questions', 'satellite', 'CTG']

st.title('Interactive & Automative Machine Learning\n', )
st.subheader("by Aleksander Petrov")


st.write(""" ## Choose dataset""")
option_dataset = st.selectbox('Dataset', all_datasets)

data = get_data(option_dataset)

st.write(f""" ## {option_dataset} Dataset""")
st.dataframe(data)


targets = list(data.iloc[:, :-1].select_dtypes('object').columns) + list(data.columns[-1:])
option_models = st.multiselect(""" Models to use""", all_models, default=all_models)
option_target = st.selectbox(""" Choose target feature""", targets)
option_ignore_features = st.multiselect(""" Choose features to ignore""", data.columns.drop(option_target))
models_to_use = [all_models[option_models[i]] for i in range(len(option_models))]

st.sidebar.write(""" ### Training parameters""")
option_train_ratio = st.sidebar.slider('Train/Validation & test ratio', 0.01, 0.99, 0.8, 0.01)

option_polynomial_features = st.sidebar.checkbox('Polynomial features', False)
option_polynomial_degree = st.sidebar.slider('Polynomial degree', 2, 10, 2, 1)
option_remove_multicollinearity = st.sidebar.checkbox('Remove multicollinearity', False)
option_multicollinearity_threshold = st.sidebar.slider('Multicollinearity threshold', 0.5, 0.99, 0.9, 0.01)
option_remove_outliers = st.sidebar.checkbox('Remove outliers', False)
option_outliers_threshold = st.sidebar.slider('Outliers threshold', 0.01, 0.3, 0.05, 0.01)
option_normalize = st.sidebar.checkbox('Normalize', False)
option_normalize_method = st.sidebar.selectbox('Normalize method', ['zscore', 'minmax', 'maxabs', 'robust'])
option_data_split_shuffle = st.sidebar.checkbox('Data split shuffle', True)
option_data_split_stratify = st.sidebar.checkbox('Data split statify', True)
option_fold = st.sidebar.slider('Number of folds', 2, 20, 5, 1)
option_fold_strategy = st.sidebar.selectbox('Fold strategy', ['stratifiedkfold', 'kfold'])
option_fold_shuffle = st.sidebar.checkbox('Fold shuffle', True)
option_session_id = st.sidebar.number_input('Random seed', 0, 1000000, 0, 1)

st.sidebar.write(""" ### Tuning parameters""")
option_optimize = st.sidebar.selectbox('Optimize', ['Accuracy','AUC','Recall','Prec.','F1','Kappa','MCC'])
option_n_iter = st.sidebar.number_input('Number of iterations', 10, 500, 20, 1)


# model = load_model('deployment_28042020')

def predict(model, input_df=None):
    clf.predict_model(estimator=model, data=input_df)
    pulled = clf.pull().set_index('Model')
    return pulled

col1_cb1 = st.checkbox('Start training', False)

col2, col3 = st.columns(2)
if col1_cb1:
    model, pulled_df_model, pulled_df_tuned_model = create_pycaret_model(df=data, 
                                                                         target=option_target, 
                                                                         ignore_features=option_ignore_features, 
                                                                         models_to_use=models_to_use,
                                                                         train_size=option_train_ratio,
                                                                         polynomial_features=option_polynomial_features,
                                                                         polynomial_degree=option_polynomial_degree,
                                                                         remove_multicollinearity=option_remove_multicollinearity,
                                                                         multicollinearity_threshold=option_multicollinearity_threshold,
                                                                         remove_outliers=option_remove_outliers,
                                                                         outliers_threshold=option_outliers_threshold,
                                                                         normalize=option_normalize,
                                                                         normalize_method=option_normalize_method,
                                                                         data_split_shuffle=option_data_split_shuffle,
                                                                         data_split_stratify=option_data_split_stratify,
                                                                         fold_strategy=option_fold_strategy,
                                                                         fold=option_fold,
                                                                         fold_shuffle=option_fold_shuffle,
                                                                         session_id=option_session_id,
                                                                         optimize=option_optimize,
                                                                         n_iter=option_n_iter)
    
    with col2:
        col2_cb2 = st.checkbox('Show trained models', False)
    with col3:
        col3_cb3 = st.checkbox('Show best tuned model', False)

    if col2_cb2:
        st.write(""" ## All trained models""")
        st.dataframe(pulled_df_model)
    if col3_cb3:
        st.write(""" ## Tuned models""")
        st.dataframe(pulled_df_tuned_model)
            
    if st.checkbox('Predict on test set'):
        st.write(""" ## Metrics on test set""")
        st.write(predict(model))

    option_plot = st.selectbox(""" Choose plot""", list(all_plots.keys()))
    option_plot_set = st.selectbox(""" Choose set""", ['Train set', 'Test set'])
    use_train_data = option_plot_set == 'Train set'
    st.write(f""" # {all_plots[option_plot]}""")
    clf.plot_model(model, option_plot, display_format='streamlit', use_train_data=use_train_data)
    
