from sklearn.preprocessing import OrdinalEncoder,OneHotEncoder,StandardScaler
from sklearn.metrics import recall_score,accuracy_score,precision_score,roc_auc_score
from sklearn.model_selection import train_test_split,GridSearchCV,StratifiedKFold
from sklearn.impute import SimpleImputer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from imblearn.pipeline import Pipeline
from typing import Dict,Any,Tuple
from sklearn.compose import ColumnTransformer
from imblearn.over_sampling import SMOTE
import pandas as pd
import sqlite3
import numpy as np
import pandas as pd
import gradio as gr
import yaml
import re
from xgboost import XGBClassifier
import warnings
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

warnings.filterwarnings(action='ignore')

class Preprocess:

    def __init__(self,config:Dict[str,Any]) -> None:
        """
        Initializes the Preprocess class with configuration parameters.

        Args:
            config (Dict[str, Any]): Configuration dictionary with file path, table name, and feature settings.
        """
        self.config = config
    
    def read_data(self) -> pd.DataFrame:
        """
        Reads data from the specified path to the database
        
        Returns:
            pd.DataFrame: DataFrame containing the retrieved data.
        
        """
        logger.info("Reading data from database ...")
        conn = sqlite3.connect(self.config['file_path'])
        table_name = self.config['table_name']
        query = f'SELECT DISTINCT * FROM {table_name}'
        df = pd.read_sql_query(query,conn)
        conn.close()
        logger.info("Data retrieved")
        return df
    
    def clean_data(self,df: pd.DataFrame) -> pd.DataFrame:
        """
        Cleans and transforms the input DataFrame according to specific rules.

        Args:
            df (pd.DataFrame): Raw input DataFrame.

        Returns:
            pd.DataFrame: Cleaned DataFrame.
        """
        logger.info("Cleaning data ...")
        df = df.copy()
        df['Age'] = df['Age'].map(lambda x:int(re.search('\d+',x).group()))
        df['Age'] = df['Age'].replace({150:np.nan})
        df['Occupation'] = df['Occupation'].replace({'unknown':np.nan})
        df['Marital Status'] = df['Marital Status'].replace({'unknown':np.nan})
        df['Education Level'] = df['Education Level'].replace({'unknown':np.nan})
        df['Credit Default'] = df['Credit Default'].replace({'yes':1,'no':0,'unknown':np.nan})
        df['Housing Loan'] = df['Housing Loan'].replace({'yes':1,'no':0,'unknown':np.nan,None:np.nan})
        df['Personal Loan'] = df['Personal Loan'].replace({'yes':1,'no':0,'unknown':np.nan,None:np.nan})
        df['Contact Method'] = df['Contact Method'].replace({'telephone':1,'Telephone':1,'Cell':0,'cellular':0})
        df['Campaign Calls'] = df['Campaign Calls'].map(abs)
        df['Contact Made'] = (df['Previous Contact Days']!=999).astype(int)
        df = df.drop(labels=['Client ID','Previous Contact Days'],axis=1)
        df['Subscription Status'] = df['Subscription Status'].replace({'yes':1,'no':0})
        logger.info("Data cleaned successfully")
        return df

    def create_preprocessor(self) -> ColumnTransformer:
        """
        Creates a ColumnTransformer preprocessor for the dataset using pipelines.

        Returns:
            ColumnTransformer: Preprocessing pipeline.
        """
        logger.info("Creating preprocessing pipeline.")
        passthrough_categories = self.config['passthrough_features']
        numeric_categories = self.config['numerical_features']
        impute_categories = self.config['impute_features']
        nominal_categories = self.config['nominal_features']
        ordinal_categories = self.config['ordinal_features']
        
        impute_pipeline = Pipeline([
            ('impute',SimpleImputer(missing_values=np.nan,strategy='most_frequent'))
        ])
        
        numeric_pipeline = Pipeline([
            ('impute',SimpleImputer(missing_values=np.nan,strategy='median')),
            ('scale',StandardScaler())
        ])
        
        ordinal_pipeline = Pipeline([
            ('impute',SimpleImputer(missing_values=np.nan,strategy='most_frequent')),
            ('encode',OrdinalEncoder(categories=[self.config['education_ranking']],handle_unknown='use_encoded_value',unknown_value=-1))
        ])
        
        onehot_pipeline = Pipeline([
            ('impute',SimpleImputer(missing_values=np.nan,strategy='most_frequent')),
            ('encode',OneHotEncoder())
        ])
        
        preprocessor = ColumnTransformer(transformers=[
            ('impute_only',impute_pipeline,impute_categories),
            ('num',numeric_pipeline,numeric_categories),
            ('ordinal',ordinal_pipeline,ordinal_categories),
            ('one_hot',onehot_pipeline,nominal_categories),
            ('pass_through','passthrough',passthrough_categories)
        ],n_jobs=-1,remainder='drop')

        logger.info("Preprocessing pipeline created.")
        
        return preprocessor
    

class ModelTrainer:

    def __init__(self,config:Dict[str,Any]) -> None:
        """
        Initializes the ModelTrainer class with configuration parameters.

        Args:
            config (Dict[str, Any]): Configuration dictionary with file path, table name, and feature settings.
        """
        self.config = config
        
    def split(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Splits the dataset into training and testing sets.

        Args:
            df (pd.DataFrame): The cleaned dataset.

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]: X_train, X_test, y_train, y_test
        """
        logger.info("Splitting data into train and test sets.")
        X,y = df.drop(self.config['target_label'],axis=1),df[self.config['target_label']]
        X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=self.config['test_size'],stratify=y,random_state=self.config['random_state'])
        logger.info("Data split successfully")
        return X_train,X_test,y_train,y_test
    
    def evaluate(self, X: pd.DataFrame, y: pd.Series, pipeline: Pipeline) -> Dict[str, float]:
        """
        Evaluates the model pipeline on the provided dataset.

        Args:
            X (pd.DataFrame): Features.
            y (pd.Series): True labels.
            pipeline (Pipeline): Trained model pipeline.

        Returns:
            Dict[str, float]: Dictionary containing evaluation metrics.
        """
        logger.info("Evaluating model performance.")
        y_pred = pipeline.predict(X)
        metrics = {
            'recall':recall_score(y,y_pred),
            'accuracy':accuracy_score(y,y_pred),
            'precision':precision_score(y,y_pred),
            'roc_auc_score':roc_auc_score(y,y_pred)
        }
        logger.info(f"Evaluation complete. Metrics: {metrics}")
        return metrics

    def train_eval_baseline(
        self,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        y_train: pd.Series,
        y_test: pd.Series,
        preprocessor: Pipeline
    ) -> Tuple[Dict[str, Pipeline], Dict[str, Dict[str, float]]]:
        """
        Trains and evaluates baseline models without hyperparameter tuning.

        Args:
            X_train (pd.DataFrame): Training features.
            X_test (pd.DataFrame): Testing features.
            y_train (pd.Series): Training labels.
            y_test (pd.Series): Testing labels.
            preprocessor (Pipeline): Preprocessing pipeline.

        Returns:
            Tuple[Dict[str, Pipeline], Dict[str, Dict[str, float]]]: Trained pipelines and their evaluation metrics.
        """
        logger.info("Training and evaluating baseline models ...")
        
        models = {
            'svm':LinearSVC(),
            'xgb':XGBClassifier(),
            'knn':KNeighborsClassifier(),
            'random_forest':RandomForestClassifier()
        }
        self.base_pipelines = {}
        self.base_metrics = {}
        
        for name,model in models.items():
            logger.info(f"Model {name} being trained ...")
            pipeline = Pipeline([
                ('preprocess',preprocessor),
                ('resample',SMOTE(random_state=self.config['random_state'])),
                ('classifier',model)
            ])
            pipeline.fit(X_train,y_train)
            self.base_pipelines[name] = pipeline
            self.base_metrics[name] = self.evaluate(X_test,y_test,pipeline)
            
        logger.info("All models trained and evaluated")
            
        return self.base_pipelines,self.base_metrics

    def train_eval_hyperparameters(
        self,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        y_train: pd.Series,
        y_test: pd.Series,
        preprocessor: Pipeline
    ) -> Tuple[Dict[str, Pipeline], Dict[str, Dict[str, float]]]:
        """
        Performs hyperparameter tuning for each model and evaluates them.

        Args:
            X_train (pd.DataFrame): Training features.
            X_test (pd.DataFrame): Testing features.
            y_train (pd.Series): Training labels.
            y_test (pd.Series): Testing labels.
            preprocessor (Pipeline): Preprocessing pipeline.

        Returns:
            Tuple[Dict[str, Pipeline], Dict[str, Dict[str, float]]]: Tuned pipelines and their evaluation metrics.
        """
        logger.info("Starting hyperparameter tuning for models ...")
        param_grid = {
            'svm':{
                    'classifier__penalty':['l1','l2']
            },
            'xgb':{
                'classifier__max_depth':[2,3,4]
            },
            'random_forest':{
                'classifier__criterion': ['gini', 'entropy'],
                'classifier__max_depth': [3, 4, 5],
            },
                'knn':{
                    'classifier__n_neighbors':[3,5,7]
            }
        }
        models = {
            'svm':LinearSVC(),
            'xgb':XGBClassifier(),
            'knn':KNeighborsClassifier(),
            'random_forest':RandomForestClassifier()
        }
        self.tuned_pipelines = {}
        self.tuned_metrics = {}
        best_score = 0
        for name,model in models.items():
            logger.info(f"Hyperparameter tuning for model {name}")
            pipeline = Pipeline([
                ('preprocess',preprocessor),
                ('resample',SMOTE(random_state=self.config['random_state'])),
                ('classifier',model)
            ])
            cv = StratifiedKFold(n_splits=5,shuffle=True,random_state=self.config['random_state'])
            grid_search = GridSearchCV(pipeline,param_grid[name],scoring='recall',n_jobs=-1,cv=cv)
            grid_search.fit(X_train,y_train)
            self.tuned_pipelines[name] = grid_search.best_estimator_
            self.tuned_metrics[name] = self.evaluate(X_test,y_test,grid_search.best_estimator_)
            
            best_score = 0
            self.best_model = None
            for name,metrics in self.tuned_metrics.items():
                if metrics['recall'] > best_score:
                    best_score = metrics['recall']
                    self.best_model = name
            
            
        logger.info(f"Best model selected: {self.best_model} with precision: {best_score}")
        return self.tuned_pipelines,self.tuned_metrics
            
    def create_interface(self) -> None:
        """
        Creates and launches a Gradio interface for the best tuned model.
        """
        logger.info("Creating Gradio interface.")
        
        def fn(age,occupation,marital,education,credit,housing,personal,contact,calls,contact_made):
            if not occupation or not marital or not education:
                raise gr.Error("Please select value for all dropdown fields")
            credit = int(credit)
            housing = int(housing)
            personal = int(personal)
            contact = int(contact)
            x = pd.DataFrame(data=[[age,occupation,marital,education,credit,housing,personal,contact,calls,contact_made]],
                    columns=['Age','Occupation','Marital Status','Education Level','Credit Default','Housing Loan','Personal Loan','Contact Method','Campaign Calls','Contact Made'])
            pipeline = self.tuned_pipelines[self.best_model]
            y_pred = pipeline.predict(x)[0]
            accuracy,precision,recall,roc_auc = self.tuned_metrics[self.best_model]['accuracy'],self.tuned_metrics[self.best_model]['precision'],self.tuned_metrics[self.best_model]['recall'],self.tuned_metrics[self.best_model]['roc_auc_score'],
            if y_pred:
                prediction = 'Yes'
            else:
                prediction = 'No'
            return prediction,self.best_model,round(accuracy,2),round(precision,2),round(recall,2),round(roc_auc,2)
        
        demo = gr.Interface(fn=fn,
    inputs=[
        gr.Number(label="Age",info="Age must be between 0 and 100"),
        gr.Dropdown(label="Occupation",choices=["admin.","blue-collar","technician","services","management","retired","entrepreneur","self-employed","housemaid","unemployed","student"]),
        gr.Dropdown(label="Marital Status",choices=["married","single","divorced"]),
        gr.Dropdown(label="Education Level",choices=["university.degree","professional.course","high.school","basic.9y","basic.6y","basic.4y","illiterate"]),
        gr.Checkbox(label="Credit Default"),
        gr.Checkbox(label="Housing Loan"),
        gr.Checkbox(label="Personal Loan"),
        gr.Checkbox(label="Mobile Contact Method"),
        gr.Number(label="Campaign Calls"),
        gr.Number(label="Previous Contact Days")
    ],
    outputs=[
        gr.Textbox(label="Subscription Status Prediction"),
        gr.Textbox(label="Best Tuned Model"),
        gr.Textbox(label="Accuracy"),
        gr.Textbox(label="Precision"),
        gr.Textbox(label="Recall"),
        gr.Textbox(label="Roc Auc")])
        demo.launch()

