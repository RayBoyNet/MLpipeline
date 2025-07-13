from MachineLearningPipeline import Preprocess,ModelTrainer
import yaml
import warnings

warnings.filterwarnings(action="ignore")

with open('./src/config.yaml') as file:
    config = yaml.safe_load(file)

if __name__ == '__main__':
    prep = Preprocess(config)
    trainer = ModelTrainer(config)
    df = prep.read_data()
    clean_df = prep.clean_data(df)
    X_train,X_test,y_train,y_test = trainer.split(clean_df)
    pipeline,metric = trainer.train_eval_baseline(X_train,X_test,y_train,y_test,prep.create_preprocessor())
    pipeline,metric = trainer.train_eval_hyperparameters(X_train,X_test,y_train,y_test,prep.create_preprocessor())
    trainer.create_interface()
    
    print(trainer.base_metrics)
    print(trainer.tuned_metrics)
