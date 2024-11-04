from torch.nn import MSELoss, AdaptiveAvgPool1d, Linear, Module
from ncps.wirings import AutoNCP, FullyConnected
from ncps.torch import LTC, CfC
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import Adam, RMSprop

import logging
logging.getLogger("pytorch_lightning").setLevel(logging.WARNING)

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler

import pickle
import os
import re
from pathlib import Path
from deprecation import deprecated

from torch import Tensor
from typing import Any, Self
from numpy import ndarray
from pandas import DataFrame
from pytorch_lightning import LightningModule
from typing import Protocol
# TODO: Implement logging system

class LiquidModel(Module, Protocol):
    def draw_model(self) -> None:
        ...
        
    def forward(self, x) -> Tensor:
        ...


# TODO: Move to a forecast data-preprocess script 
def scale_data(data_train:ndarray, data_test:ndarray|None, scaler_type:str|MinMaxScaler|StandardScaler='std') -> tuple[ndarray, ndarray|None, StandardScaler|MinMaxScaler]:
    if isinstance(scaler_type, str):
        match scaler_type:
            case 'std':
                sc = MinMaxScaler()
            case 'minmax':
                sc = StandardScaler()
            case _:
                # TODO: log this
                print(f'[WARNING] No Scaler type found with name {scaler_type}, using StandardScaler')
                sc = StandardScaler()
    else:
        sc = scaler_type

    data_train = sc.fit_transform(data_train)
    if data_test is not None:
        data_test = sc.transform(data_test)
    return data_train, data_test, sc

# TODO: Move to a forecast data-preprocess script 
def prepare_lnn_data(data:ndarray, scaler_type:str='std', test_index:int|None=None) -> tuple[Tensor, Tensor|None, int|None, StandardScaler|MinMaxScaler]:
    if test_index is not None:
        data_train, data_test  = data[:test_index, :], data[test_index:, :]
    else: 
        data_test = None
        data_train = data

    data_train, data_test, sc = scale_data(data_train, data_test, scaler_type=scaler_type)

    data_train = np.expand_dims(data_train, axis=0).astype(np.float32)  # Adding batch dimension
    
    if data_test is not None:
        data_test = np.expand_dims(data_test, axis=0).astype(np.float32)  # Adding batch dimension
        data_test_ = Tensor(np.concatenate([data_train, data_test], axis=1))
    else:
        data_test_ = None
    data_train_ = Tensor(data_train)
    return data_train_, data_test_, test_index, sc


class LiquidNeuralNetwork:
    def __init__(self, n_neurons:int, neuron_type:LTC|CfC, in_features:int, out_features:int, seed:int) -> None:
        super(LiquidNeuralNetwork, self).__init__()
        
        self.ncp_wiring_output:AutoNCP = AutoNCP(n_neurons, out_features, seed=seed) # LNN
        self.ncp_layer_output:LTC|CfC = neuron_type(in_features, self.ncp_wiring_output, batch_first=True)
        
        self.global_avg_pool:AdaptiveAvgPool1d = AdaptiveAvgPool1d(out_features)  # GlobalAveragePooling1D equivalent
        self.fc:Linear = Linear(out_features, out_features)  # Adjust based on hidden size of the RNN layer

    def forward(self, x) -> Tensor:
        x, _ = self.ncp_layer_output.forward(x)
        x = self.global_avg_pool.forward(x)
        x = self.fc.forward(x)
        return x
    
    def draw_model(self) -> None:
        sns.set_style("white")
        plt.figure(figsize=(6, 5))
        plt.title(f"NCP Layer Architecture")
        legend_handles = self.ncp_wiring_output.draw_graph(layout='circular', draw_labels=False, neuron_colors={"command": "tab:cyan"})
        plt.legend(handles=legend_handles, loc="upper center", bbox_to_anchor=(1, 1))
        sns.despine(left=True, bottom=True)
        plt.tight_layout()
        plt.show()

@deprecated
class LiquidNeuralNetwork_old:
    def __init__(self, n_neurons:int, model_type:str|None='LTC', verbose:int=1) -> None:
        self.n_neurons = n_neurons
        self.verbose = verbose

        if model_type is None:
            self.model_type = 'LTC'
        else:
            match model_type:
                case 'LTC':
                    self.model_type = 'LTC'
                case 'CfC':
                    self.model_type = 'CfC'
                case _:
                    raise ValueError(f'Model type selected ({model_type}) does not match any of the known ones: "LTC" / "CfC"')
                
        self.metadata = {'is_trained':False, 'model_type':self.model_type, 'n_neurons':self.n_neurons, 'verbose':self.verbose}

    def fit(self, X_train:Tensor, y_train:Tensor, epochs:int, clip_value:int|float=1, learning_rate:int|float=0.01, n_jobs:int=8, logger_step:int=1, enable_progress_bar:bool=False, enable_model_summary:bool=False, profiler:bool=False) -> None:
        self.in_features = X_train.shape[-1]
        self.out_features = y_train.shape[-1]
        dataloader = DataLoader(TensorDataset(X_train, y_train), batch_size=X_train.shape[0], shuffle=True, num_workers=n_jobs, persistent_workers=True)

        self.wiring = AutoNCP(self.n_neurons, self.out_features)

        if self.model_type == 'LTC':
            self.model = LTC(self.in_features, self.wiring, batch_first=True)
        elif self.model_type == 'CfC':
            self.model = CfC(self.in_features, self.wiring, batch_first=True)

        learn = SequenceLearner_LNN(self.model, lr=learning_rate)
        trainer = pl.Trainer(max_epochs=epochs, gradient_clip_val=clip_value, log_every_n_steps=logger_step, precision='bf16-mixed', enable_progress_bar=enable_progress_bar, enable_model_summary=enable_model_summary, profiler=profiler, enable_checkpointing=False, logger=False) # Clip gradient to stabilize training
        # trainer = pl.Trainer(max_epochs=epochs, gradient_clip_val=clip_value, log_every_n_steps=logger_step, gpus=1, precision='16-mixed') # Use GPU

        if self.verbose > 0:
            sns.set_style("white")
            plt.figure(figsize=(6, 5))
            legend_handles = self.wiring.draw_graph(draw_labels=False, neuron_colors={"command": "tab:cyan"})
            plt.legend(handles=legend_handles, loc="upper center", bbox_to_anchor=(1, 1))
            sns.despine(left=True, bottom=True)
            plt.tight_layout()
            plt.show()

        trainer.fit(learn, dataloader)
        self.metadata['is_trained'] = True
        self.metadata['epochs'] = epochs
        self.metadata['clip_value'] = clip_value
        self.metadata['learning_rate'] = learning_rate
        self.metadata['in_features'] = self.in_features
        self.metadata['out_features'] = self.out_features

    def predict(self, X_test:Tensor) -> Tensor:
        with torch.no_grad():
            prediction = self.model(X_test)[0].numpy()
            # prediction = self.model(X_test)[0].cpu().numpy() # Use GPU
        return prediction
    
    def save_model(self, model_path:Path) -> None:
        if not os.path.exists(model_path.parent):
            os.makedirs(model_path.parent)
        
        with open(os.path.join(model_path.parent, 'metadata_lnn.pkl'), 'wb') as f:
            pickle.dump(self.metadata, f)

        torch.save({'model_state_dict':self.model.state_dict(), 'other_metadata':self.metadata}, model_path)

    def load_model(self, model_path) -> Self:
        checkpoint = torch.load(model_path)
        self.metadata = checkpoint['other_metadata']
        self.wiring = AutoNCP(self.metadata['n_neurons'], self.metadata['out_features']) # FIXME: Use seed in metadata to correct retrieval of the architecture

        if self.model_type == 'LTC':
            self.model = LTC(self.metadata['in_features'], self.wiring, batch_first=True) # TODO: This prints an 'alloc!' message which should not happen FIXME: Redirect sdtout
        elif self.model_type == 'CfC':
            self.model = CfC(self.metadata['in_features'], self.wiring, batch_first=True) # TODO: This prints an 'alloc!' message which should not happen FIXME: Redirect sdtout

        self.model.load_state_dict(checkpoint['model_state_dict'])

        return self


# LightningModule for training a RNNSequence module
@deprecated
class SequenceLearner_LNN(LightningModule):
    def __init__(self, model:LiquidNeuralNetwork, lr:float=0.005) -> None:
        super().__init__()
        self.model = model
        self.lr = lr

    def training_step(self, batch:Tensor, batch_idx:int) -> dict[str, Any]:
        x, y = batch
        # x, y = x.to(self.device), y.to(self.device) # Use GPU
        y_hat, _ = self.model.forward(x)
        y_hat = y_hat.view_as(y)
        loss = MSELoss()(y_hat, y)
        self.log("train_loss", loss, prog_bar=True)
        return {"loss": loss}

    def test_step(self, batch:Tensor, batch_idx:int) -> Tensor:
        x, y = batch
        # x, y = x.to(self.device), y.to(self.device) # Use GPU
        y_hat, _ = self.model.forward(x)
        y_hat = y_hat.view_as(y)
        loss = MSELoss()(y_hat, y)
        # self.log("val_loss", loss, prog_bar=True)

        # return validation_step(batch, batch_idx)
        return loss

    def configure_optimizers(self) -> Adam:
        return Adam(self.model.parameters(), lr=self.lr)
    
""" def validation_step(self, batch:Tensor, batch_idx:int) -> Tensor:
        x, y = batch
        # x, y = x.to(self.device), y.to(self.device) # Use GPU
        y_hat, _ = self.model.forward(x)
        y_hat = y_hat.view_as(y)
        loss = MSELoss()(y_hat, y)

        self.log("val_loss", loss, prog_bar=True)
        return loss"""
    
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # Set device as GPU
@deprecated
class MultiLNN:
    def __init__(self, n_neurons:int|None, model_type:str|None, verbose:int=1) -> None:
        self.n_neurons:int|None = n_neurons
        self.verbose:int = verbose
        self.profiler = 'simple' if self.verbose>0 else None

        if model_type is None:
            self.model_type = 'LTC'
        else:
            match model_type:
                case 'LTC':
                    self.model_type = 'LTC'
                case 'CfC':
                    self.model_type = 'CfC'
                case _:
                    raise ValueError(f'Model type selected ({model_type}) does not match any of the known ones: "LTC" / "CfC"')
        self.metadata = {'is_trained':False, 'model_type':self.model_type, 'n_neurons':self.n_neurons, 'verbose':self.verbose}

    def _create_model(self) -> LiquidNeuralNetwork:
        model:LiquidNeuralNetwork = LiquidNeuralNetwork(self.n_neurons, self.model_type, self.verbose)
        return model

    def fit(self, X_train:Tensor, y_train:Tensor, epochs:int, clip_value:int|float=1, learning_rate:int|float=0.01, n_jobs:int=8, logger_step:int=1) -> None:
        self.in_features = X_train.shape[-1]
        self.out_features = y_train.shape[-1]
        self.metadata['prediction_shape'] = y_train.shape
        self.model:dict[str, LiquidNeuralNetwork] = {}
        for i in range(1, y_train.shape[-1]+1):
            self.model[f'layer {i}'] = self._create_model()
            self.model[f'layer {i}'].fit(X_train, y_train[:, :, i-1:i], epochs, clip_value=clip_value, learning_rate= learning_rate, n_jobs=n_jobs, logger_step=logger_step, enable_progress_bar=self.verbose>0, enable_model_summary=self.verbose>0, profiler=self.profiler)
        
        self.metadata['is_trained'] = True
        self.metadata['epochs'] = epochs
        self.metadata['clip_value'] = clip_value
        self.metadata['learning_rate'] = learning_rate
        self.metadata['in_features'] = self.in_features
        self.metadata['out_features'] = self.out_features
    
    def predict(self, X_test:Tensor|ndarray) -> Tensor:
        y_pred = np.zeros((self.metadata['prediction_shape'][0], X_test.shape[1], self.metadata['prediction_shape'][2]))
        for layer, model in self.model.items():
            prediction:Tensor = model.predict(X_test)
            y_pred[:, :, int(layer[-1])-1:int(layer[-1])] = prediction
        return y_pred
    
    def save_model(self, model_path:str|Path, model_name:str, metadata:dict[str, Any]) -> None:
        self.metadata.update(metadata)

        if not os.path.exists(model_path):
            os.makedirs(model_path)
        # Save metadata
        with open(os.path.join(model_path, 'metadata_lnn.pkl'), 'wb') as f:
            pickle.dump(self.metadata, f)

        for layer, model in self.model.items():
            model.save_model(os.path.join(model_path, f'{layer}_{model_name}.pth'))

    def load_model(self, model_path:str|Path, model_name:str) -> Self:
        # Load metadata
        with open(os.path.join(model_path, 'metadata_lnn.pkl'), 'rb') as f:
            self.metadata = pickle.load(f)

        files = os.listdir(model_path)
        self.model = {}
        for file in files:
            match = re.match(r'layer (\d+)', file)
            if match:
                layer_num = int(match.group(1))
                layer_key = f'layer {layer_num}'

                # Load model
                self.model[layer_key] = LiquidNeuralNetwork(n_neurons=self.metadata['Hyperparameters']['n_neurons'], model_type=self.model_type, verbose=self.verbose).load_model(os.path.join(model_path, f'{layer_key}_{model_name}.pth'))

        return self
    

# NOT IMPLEMENTED
class SequenceLearner_GLNN(Module):
    def __init__(self, n_neurons, in_features, out_features, seed) -> None:
        super(SequenceLearner_GLNN, self).__init__()
        wiring_input = FullyConnected(in_features, None, erev_init_seed=seed)
        self.input_LTC_layer = LTC(in_features, wiring_input, batch_first=True)
        
        wiring_lnn = AutoNCP(n_neurons, out_features, seed=seed)
        self.LNN_LTC_layer = LTC(in_features, wiring_lnn, batch_first=True)
        
        self.global_avg_pool = AdaptiveAvgPool1d(out_features)  # GlobalAveragePooling1D equivalent
        self.fc = Linear(out_features, out_features)  # Adjust based on hidden size of the RNN layer

    def forward(self, x):
        x, _ = self.input_LTC_layer(x) # GLNN
        x, _ = self.LNN_LTC_layer(x) # LNN
        x = self.global_avg_pool(x)
        x = self.fc(x)
        return x


class GeneralizedLiquidNeuralNetwork(Module):
    def __init__(self, n_neurons:int, neuron_type:LTC|CfC, in_features:int, out_features:int, seed:int) -> None:
        super(GeneralizedLiquidNeuralNetwork, self).__init__()
        
        self.ncp_wiring_input:FullyConnected = FullyConnected(in_features, None, erev_init_seed=seed) # GLNN
        self.ncp_wiring_output:AutoNCP = AutoNCP(n_neurons, out_features, seed=seed) # LNN
        self.ncp_layer_input:LTC|CfC = neuron_type(in_features, self.ncp_wiring_input, batch_first=True)
        self.ncp_layer_output:LTC|CfC = neuron_type(in_features, self.ncp_wiring_output, batch_first=True)
        
        self.global_avg_pool:AdaptiveAvgPool1d = AdaptiveAvgPool1d(out_features)  # GlobalAveragePooling1D equivalent
        self.fc:Linear = Linear(out_features, out_features)  # Adjust based on hidden size of the RNN layer

    def forward(self, x) -> Tensor:
        x, _ = self.ncp_layer_input.forward(x)
        x, _ = self.ncp_layer_output.forward(x)
        x = self.global_avg_pool.forward(x)
        x = self.fc.forward(x)
        return x
    
    def draw_model(self) -> None:
        sns.set_style("white")
        
        plt.figure(figsize=(6, 5))
        plt.title(f"Input NCP Layer Architecture with LTC Neurons")
        legend_handles = self.ncp_wiring_input.draw_graph(layout='circular', draw_labels=False, neuron_colors={"command": "tab:cyan"})
        plt.legend(handles=legend_handles, loc="upper center", bbox_to_anchor=(1, 1))
        sns.despine(left=True, bottom=True)
        plt.tight_layout()
        plt.show()
        
        plt.figure(figsize=(6, 5))
        plt.title(f"NCP Layer Architecture")
        legend_handles = self.ncp_wiring_output.draw_graph(layout='circular', draw_labels=False, neuron_colors={"command": "tab:cyan"})
        plt.legend(handles=legend_handles, loc="upper center", bbox_to_anchor=(1, 1))
        sns.despine(left=True, bottom=True)
        plt.tight_layout()
        plt.show()
    
class LiquidModelEncapsultor:
    def __init__(self, n_neurons:int|None, model_type:LiquidModel, neuron_type:LTC|CfC, verbose:int=1, seed:int=42) -> None:
        self.n_neurons = n_neurons
        self.verbose = verbose
        self.seed = seed
        
        self.model_is_built = False
        self.is_trained = False
        
        self.model_type = model_type
        self.neuron_type = neuron_type

    def _build_model(self) -> None:
        if self.model_is_built:
            raise ValueError('Model is already built.')
        elif (not hasattr(self, 'n_features_in')) or (not hasattr(self, 'n_features_out')):
            raise ValueError('Model does not have information about the data to be built.')
        elif self.n_neurons is None:
            raise ValueError('Number of model neurons has not been provided. Did you forget to load a model?')
        else:
            self.model:LiquidModel = self.model_type(self.n_neurons, self.neuron_type, self.n_features_in, self.n_features_out, self.seed)
            self.model_is_built = True
    
    def _batch_data(self, X_train) -> Tensor:
        return X_train
    
    def _window_data(self, data_to_window:Tensor, window_size:int) -> Tensor:
        data_to_window_np = data_to_window.numpy()
        data_length = len(data_to_window_np)
        windowed_data = []

        for i in range(data_length):
            if (data_length - i) < window_size:
                # print(f"Ended on iteration: {i} with last point taken on {i+window_size-1}")
                break
            
            windowed_data.append(data_to_window_np[i:i+window_size])

        windowed_data_ = np.array(windowed_data)
        
        data_to_window = Tensor(windowed_data_)
        return data_to_window
        
    def fit(self, X_train:Tensor, y_train:Tensor, epochs:int, clip_value:int|float=1, learning_rate:int|float=0.01, n_jobs:int=8, data_is_batched:bool=False, data_is_windowed:bool=False, window_size:int|None=None, batch_size:int|None=None) -> None:
        self.n_features_in:int = X_train.shape[-1]
        self.n_features_out:int = y_train.shape[-1]
        self.window_size = window_size
        self.batch_size = batch_size
        
        if data_is_windowed:
            self.window_size = X_train.shape[1]
            self.window_test = False
        else:
            if self.window_size is None:
                raise ValueError("Data is not specified to be already windowed and the window size has not been set.")
            else:
                self.window_test = True
                X_train = self._window_data(X_train, self.window_size)
                y_train = self._window_data(y_train, self.window_size)

        if data_is_batched:
            self.batch_size = X_train.shape[0]
            self.batch_test = False
        else:
            if self.batch_size is None:
                raise ValueError("Data is not specified to be already batched and the batch size has not been set.")
            else:
                self.batch_test = True
                X_train = self._batch_data(X_train)
                y_train = self._batch_data(y_train)
        
        self._build_model()
        
        if not self.is_trained:
            self.losses = []
        

        optimizer = Adam(self.model.parameters(), lr=learning_rate) # Optimizer
        loss_function = MSELoss() # Loss function
        train_tensor = TensorDataset(X_train, y_train)
        train_loader = DataLoader(train_tensor, batch_size=self.batch_size, shuffle=False, num_workers=n_jobs, persistent_workers=True)
        
        for epoch in range(epochs):
            self.model.train() # Set in train mode
            for X_batch, y_batch in train_loader:
                optimizer.zero_grad()
                outputs = self.model.forward(X_batch)
                loss = loss_function(outputs, y_batch.float())
                loss.backward()
                optimizer.step()
                self.losses.append(loss.item())
            print(f'Epoch {epoch + 1}/{epochs}, Loss: {self.losses[-1]:.4f}', end='\r')


        self.is_trained = True
        self.epochs = epochs
        self.clip_value = clip_value
        self.learning_rate = learning_rate
        self.n_features_in = self.n_features_in
        self.n_features_out = self.n_features_out

    def predict(self, X_test:Tensor) -> Tensor:
        if self.window_test:
            X_test = self._window_data(X_test, self.window_size)

        if self.batch_test:
            X_test = self._batch_data(X_test)
                
        with torch.no_grad():
            prediction = self.model.forward(X_test)
        return prediction
    
    def _set_up_parameters(self, metadata:dict[str, object|int|float]) -> None:
        for param, value in metadata.items():
            self.__setattr__(param, value)

    def save_model(self, model_path:Path) -> None:
        metadata = {key:value for key, value in self.__dict__.items() if not key.startswith('__') and not callable(value)}
        if not os.path.exists(model_path.parent):
            os.makedirs(model_path.parent)
        torch.save({'model_state_dict':self.model.state_dict(), 'metadata':metadata}, model_path)

    def load_model(self, model_path) -> Self:
        checkpoint = torch.load(model_path)
        self._set_up_parameters(checkpoint['metadata'])
        self.model_is_built = False
        self._build_model()
        self.model.load_state_dict(checkpoint['model_state_dict'])

        return self