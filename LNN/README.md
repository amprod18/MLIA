# Neural Circuit Policies (PyTorch)

<div align="center"><img src="https://raw.githubusercontent.com/mlech26l/ncps/master/docs/img/banner.png" width="800"/></div>

## 📜 Papers

[Neural Circuit Policies Enabling Auditable Autonomy (Open Access)](https://publik.tuwien.ac.at/files/publik_292280.pdf).  
[Closed-form continuous-time neural networks (Open Access)](https://www.nature.com/articles/s42256-022-00556-7)

Neural Circuit Policies (NCPs) are designed sparse recurrent neural networks loosely inspired by the nervous system of the organism [C. elegans](http://www.wormbook.org/chapters/www_celegansintro/celegansintro.html).
The goal of this package is to making working with NCPs in PyTorch as easy as possible.

[📖 NCP Docs](https://ncps.readthedocs.io/en/latest/index.html)

```python
from LNN.LNN import LiquidNeuralNetwork

epochs = 400
clip_value = 1
learning_rate = 0.01
n_neurons = 20

lnn = LiquidNeuralNetwork(n_neurons=n_neurons)
lnn.fit(X_train, y_train, epochs=epochs, clip_value=clip_value, learning_rate=learning_rate, n_jobs=8)
prediction = lnn.predict(data_x_test)
```

## Usage: Models and Wirings

The package provides a model combining the liquid time-constant (LTC) and the closed-form continuous-time (CfC) models together in a 'sklearn-ish' way.

```python
from LNN.LNN import LiquidNeuralNetwork

lnn = LiquidNeuralNetwork(n_neurons=20, model_type='LTC')
lnn = LiquidNeuralNetwork(n_neurons=20, model_type='CfC')
```

The LNNs defined above consider an NCP connection between layers in contrast to LSTM, GRUs, and other RNNs.
The distinctiveness of the NCP wiring is its structured and statistial-based connections. The LNN model constructs itself as follows:

```python
from ncps.torch import CfC, LTC
from ncps.wirings import AutoNCP

wiring = AutoNCP(28, 4) # 28 neurons, 4 outputs
input_size = 20
rnn1 = LTC(units) # fully-connected LTC
rnn2 = CfC(units) # fully-connected CfC
rnn3 = LTC(wiring) # NCP wired LTC
rnn4 = CfC(wiring) # NCP wired CfC
```

Fore more information about the NPC wiring algorithm refer either to the [NCP paper](https://publik.tuwien.ac.at/files/publik_292280.pdf) or to the [technical documentation](/docu/Documentación%20Técnca%20NCPs.docx) of the module.
