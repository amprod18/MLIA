from torch.nn import MSELoss, AdaptiveAvgPool1d, Linear, Module, functional
from ncps.wirings import AutoNCP, FullyConnected
from ncps.torch import LTC, CfC
import torch
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import Adam, RMSprop

import logging
logging.getLogger("pytorch_lightning").setLevel(logging.WARNING)

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler

import os
from pathlib import Path
from deprecation import deprecated

from torch import Tensor
from typing import Any, Self
from numpy import ndarray
from pandas import DataFrame
from typing import Protocol
# TODO: Implement logging system

class LiquidModel(Module):
    def draw_model(self) -> None:
        ...
        
    def forward(self, x) -> Tensor:
        return x


# TODO: Move to a forecast data-preprocess script
# FIXME 
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


class LiquidNeuralNetwork(Module):
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
        
    def fit(self, X_train:Tensor, y_train:Tensor, epochs:int, shuffle_training_data:bool=False, learning_rate:int|float=0.01, n_jobs:int=8, data_is_batched:bool=False, data_is_windowed:bool=False, window_size:int|None=None, batch_size:int|None=None) -> None:
        self.n_features_in:int = X_train.shape[-1]
        self.n_features_out:int = y_train.shape[-1]
        self.window_size = window_size
        self.batch_size = batch_size
        self.epochs = epochs
        self.shuffle_training_data = shuffle_training_data
        self.learning_rate = learning_rate
        self.data_is_windowed = data_is_windowed
        self.data_is_batched = data_is_batched
        
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
        train_loader = DataLoader(train_tensor, batch_size=self.batch_size, shuffle=shuffle_training_data, num_workers=n_jobs, persistent_workers=True)
        
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

    def predict(self, X_test:Tensor) -> Tensor:
        if (self.window_test) and (self.window_size is not None):
            X_test = self._window_data(X_test, self.window_size)

        if self.batch_test:
            X_test = self._batch_data(X_test)
                
        with torch.no_grad():
            prediction = self.model.forward(X_test)
        return prediction
    
    def score(self, X_test:Tensor, y_test:Tensor) -> float:
        if (self.window_test) and (self.window_size is not None):
            X_test = self._window_data(X_test, self.window_size)

        if self.batch_test:
            X_test = self._batch_data(X_test)
            
        prediction = self.predict(X_test)
        return functional.mse_loss(prediction, y_test.float()).item()
    
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