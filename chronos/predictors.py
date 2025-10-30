import joblib
import pandas as pd
import gdown
import os
import xgboost as xgb

from .config import Models, Datasets, Optimizers, Devices


class Predictor:
    def __init__(self):
        self.script_dir = os.path.dirname(__file__)
        self.models_dir = os.path.join(self.script_dir, 'models')
        
        if not os.path.exists(self.models_dir):
            print(f"{self.models_dir} not found. Downloading and extracting models...")

            gdown.download_folder("https://drive.google.com/drive/folders/1nCRqkJYJw15sk2Zt6K0Tbfc5FUoBmSs7", output=self.models_dir, quiet=False)
    
    @staticmethod
    def predict(self, features):
        pass
    

class MLPEpochTimePredictor(Predictor):
    def __init__(self):
        super().__init__()
        model_path = f'{self.models_dir}/mlp_epoch_time_model.json'
        
        self.model = xgb.XGBRegressor()
        self.model.load_model(model_path)
        
    
    def predict(self, features):
        return self.model.predict(features)
    

class CNNEpochTimePredictor(Predictor):
    def __init__(self):
        super().__init__()
        model_path = f'{self.models_dir}/cnn_epoch_time_model.json'
        
        self.model = xgb.XGBRegressor()
        self.model.load_model(model_path)
        
    
    def predict(self, features):
        return self.model.predict(features)
   

class TransformerEpochTimePredictor(Predictor):
    def __init__(self):
        super().__init__()
        model_path = f'{self.models_dir}/transformer_epoch_time_model.json'
        
        self.model = xgb.XGBRegressor()
        self.model.load_model(model_path)
        
    
    def predict(self, features):
        return self.model.predict(features)
    

class MLPEpochConvergencePredictor(Predictor):
    def __init__(self):
        super().__init__()
        model_path = f'{self.models_dir}/mlp_epoch_convergence_model.pkl'
        
        self.model = joblib.load(model_path)
        
    
    def predict(self, features):
        return self.model.predict(features)
    

class CNNEpochConvergencePredictor(Predictor):
    def __init__(self):
        super().__init__()
        model_path = f'{self.models_dir}/cnn_epoch_convergence_model.pkl'
        
        self.model = joblib.load(model_path)
        
    
    def predict(self, features):
        return self.model.predict(features)
    

class TransformerEpochConvergencePredictor(Predictor):
    def __init__(self):
        super().__init__()
        model_path = f'{self.models_dir}/transformer_epoch_convergence_model.pkl'
        
        self.model = joblib.load(model_path)
        
    
    def predict(self, features):
        return self.model.predict(features)
    
    

class TrainingTimePredictor:
    """
    A class to predict training time based on model, dataset, and hardware configuration.
    """
    def __init__(self, model: Models, dataset: Datasets, batch_size: int, 
                 learning_rate: float, optimizer: Optimizers, gpu: Devices):
        
        self.model = model
        self.dataset = dataset
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.optimizer = optimizer
        self.gpu = gpu


        if model in {Models.Mixer, Models.Resmlp, Models.Asmlp}:
            self.epoch_time_predictor = MLPEpochTimePredictor()
            self.epoch_convergence_predictor = MLPEpochConvergencePredictor()

        
        print(f"Predictor configured for: {self.model.value} on {self.gpu.value}")
        print(f"Optimizer: {self.optimizer.value}, Dataset: {self.dataset.value}")

    
    def predict_epoch_time(self):
        return self.epoch_time_predictor.predict()
    

    def predict_number_of_epochs(self):
        return self.epoch_convergence_predictor.predict()