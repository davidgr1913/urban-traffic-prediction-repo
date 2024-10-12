
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler, StandardScaler, OrdinalEncoder, PolynomialFeatures, FunctionTransformer, LabelEncoder, PowerTransformer, KBinsDiscretizer
from sklearn.decomposition import PCA
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, RepeatedStratifiedKFold, RepeatedKFold, cross_validate
from sklearn.metrics import confusion_matrix, precision_score, recall_score, accuracy_score, precision_recall_curve, classification_report, mean_squared_error, mean_absolute_error, r2_score
from sklearn.datasets import load_iris
from sklearn.inspection import permutation_importance
from sklearn import metrics
from imblearn.pipeline import Pipeline as ImbPipeline
from scipy.stats import norm,skew
from pandas.plotting import scatter_matrix

# Modelos:
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
import numpy as np
import pandas as pd
import yaml

from src.utils.logs import get_logger

class ModelEvaluator:
    def __init__(self,config_path):
        self.model_dict = {
            'LR': LinearRegression(),
            'kNN': KNeighborsRegressor(),
            'DTree': DecisionTreeRegressor(),
            'RF': RandomForestRegressor(),
            'XGBoost': XGBRegressor(),
            'SVR': SVR()
        }
        self.config = yaml.safe_load(open(config_path))
        self.logger = get_logger('DATA_PREPROCESS', log_level=self.config['base']['log_level'])
        self.logger.info('Get dataset')

    def get_data(self):
        self.logger.info('Get dataset')
        df_train = pd.read_csv(self.config['data_split']['data_train_path'], sep=';')
        df_test = pd.read_csv(self.config['data_split']['data_test_path'], sep=';')
        self.Xtrain = df_train.drop(self.config['base']['target'], axis=1)
        self.ytrain = df_train[self.config['base']['target']]
        self.Xtest = df_test.drop(self.config['base']['target'], axis=1)
        self.ytest = df_test[self.config['base']['target']]


    
    def get_selected_models(self):
        selected_models = []
        selected_names = []
        for name in self.config['model']['list_models']:
            if name in self.model_dict:
                selected_models.append(self.model_dict[name])
                selected_names.append(name)
        return selected_models, selected_names
    

    def evaluate_initial_models(self):
        modelos, nombres = self.get_selected_models()

        num_pipe = Pipeline([
            ("log_transform", FunctionTransformer(np.log1p)),
            ('z-score', StandardScaler())
        ])

        num_pipe_nombres = self.Xtrain.select_dtypes(include=np.number).columns.tolist()
        columnasTransformer = ColumnTransformer([('num', num_pipe, num_pipe_nombres)], remainder = 'passthrough')
        
        resultados = []

        for i in range(len(modelos)):
            pipeline = Pipeline(steps=[('ct', columnasTransformer), ('m', modelos[i])])

            micv = RepeatedKFold(n_splits=self.config['model']['n_splits'], n_repeats=self.config['model']['n_repeats'], 
                                 random_state=self.config['base']['random_state'])

            mismetricas = self.config['model']['metrics']

            scores = cross_validate(pipeline,
                                    self.Xtrain,
                                    self.ytrain,
                                    scoring=mismetricas,
                                    cv=micv,
                                    return_train_score=True)

            resultados.append(scores)

            print(f'>> {nombres[i]}')
            for j, k in enumerate(list(scores.keys())):
                if j > 1:
                    print(f'\t {k} {np.mean(scores[k]):.3f} ({np.std(scores[k]):.3f})')

        
    
    def optimize_hyperparameters(self):
        modelos, nombres = self.get_selected_models()

        num_pipe = Pipeline([
            ("log_transform", FunctionTransformer(np.log1p)),
            ('z-score', StandardScaler())
        ])

        num_pipe_nombres = self.Xtrain.select_dtypes(include=np.number).columns.tolist()
        columnasTransformer = ColumnTransformer([('num', num_pipe, num_pipe_nombres)], remainder = 'passthrough')

        modeloXGB = XGBRegressor()

        param_grid = [{
            'max_depth': [1, 5, 10],
            'learning_rate': [0.01, 0.1, 0.3, 0.5, 1, 5],
            'subsample': [0.1, 0.3, 0.5, 0.7, 1]
        }]

        grid_search = GridSearchCV(modeloXGB, param_grid, cv=5, scoring='neg_root_mean_squared_error')
        grid_search.fit(self.Xtrain, self.ytrain)

        best_params = grid_search.best_params_
        best_model = grid_search.best_estimator_
        print("Mejores hiperparámetros encontrados para XGBoost:", best_params)