import logging
import pickle
from typing import Optional, Mapping, Any

import haiku as hk
import jax
import jax.numpy as jnp
import jax.random as jr
import jax.nn as jnn
import haiku.initializers as hki
import einops
import functools as ft

import numpy as np
import optax
import pandas as pd

#from statsmodels.tsa.deterministic import CalendarFourier, DeterministicProcess

from sklearn.preprocessing import OrdinalEncoder, MinMaxScaler


def make_multistep_target(ts, steps):
    return pd.concat(
        {f'y_step_{i + 1}': ts.shift(-i)
         for i in range(steps)},
        axis=1)

def make_lags(ts, lags):
    return pd.concat(
        {
            f'y_lag_{i}': ts.shift(i)
            for i in range(1, lags + 1)
        },
        axis=1)


def load_dataset(ftrain='./data/train.csv', ftest='./data/test.csv', fstores='./data/stores.csv', 
                    fholidays='./data/holidays_events.csv', foil='./data/oil.csv', ftransact='./data/transactions.csv'):
    train = pd.read_csv(ftrain, index_col=0)
    test = pd.read_csv(ftest, index_col=0)
    stores = pd.read_csv(fstores)
    holidays = pd.read_csv(fholidays)
    oil = pd.read_csv(foil)
    transactions = pd.read_csv(ftransact)

    holidays['date'] = pd.to_datetime(holidays['date'], format = "%Y-%m-%d")
    oil['date'] = pd.to_datetime(oil['date'], format = "%Y-%m-%d")
    transactions['date'] = pd.to_datetime(transactions['date'], format = "%Y-%m-%d")
    train['date'] = pd.to_datetime(train['date'], format = "%Y-%m-%d")
    test['date'] = pd.to_datetime(test['date'], format = "%Y-%m-%d")

    object_cols = [cname for cname in train.columns 
               if train[cname].dtype == "object" 
               and cname != "date"]
    num_cols = [cname for cname in train.columns 
            if train[cname].dtype in ['int64', 'float64']]
    
    ordinal_enc = OrdinalEncoder()
    train[object_cols] = ordinal_enc.fit_transform(train[object_cols])

    scaler = MinMaxScaler(feature_range=(0, 1))

    for col in num_cols:
        scaled_data = scaler.fit_transform(train[col].values.reshape(-1, 1))
        train[col] = pd.Series(scaled_data.flatten())

    train_data = train.groupby(['date']).agg({'sales':'mean', 'onpromotion':'mean'})
    x_train = train_data.copy()
    y_train = train_data.sales.copy()

    # Transforming into time series data


# Model Definition
def layer_norm(x: jnp.ndarray, name: Optional[str] = None) -> jnp.ndarray:
    return hk.LayerNorm(axis=-1, create_scale=True, create_offset=True, eps=1e-6, name=name)(x)

class Time2Vec(hk.Module):
    def __init__(self, kernel_size=1):
        super().__init__()
        self.k = kernel_size

    def __call__(self, inputs):
        ii1 = inputs.shape[1]
        init = hki.RandomUniform(0, 1.0 / ii1)
        bias = hk.get_parameter('wb', shape=(ii1,), init=init) * inputs + hk.get_parameter('bb', shape=(ii1,), init=init)
        wa = hk.get_parameter('wa', shape=(1, ii1, self.k), init=init)
        ba = hk.get_parameter('ba', shape=(1, ii1, self.k), init=init)
        dp = jnp.dot(inputs, wa) + ba
        weights = jnp.sin(dp)

        ret = jnp.concatenate([jnp.expand_dims(bias, axis=-1), weights], -1)
        ret = einops.rearrange(ret, "t b c -> t (b c)")
        return ret


class AttentionBlock(hk.Module):
    def __init__(self, num_heads, head_size, ff_dim=None, dropout=0.0):
        super().__init__()
        if ff_dim is None:
            self.ff_dim = head_size
        else:
            self.ff_dim = ff_dim
        self.dropout = dropout
        self.num_heads = num_heads
        self.head_size = head_size

    def __call__(self, inputs, is_training=True):
        dropout = self.dropout if is_training else 0
        out_features = inputs.shape[-1]

        x = hk.MultiHeadAttention(num_heads=self.num_heads, key_size=self.head_size, w_init_scale=1.0)(inputs, inputs, inputs)
        x = hk.BatchNorm(True, True, decay_rate=0.9, eps=1e-6, scale_init=hki.Constant(1.0), offset_init=hki.Constant(1e-8))(x, is_training)
        x = hk.dropout(hk.next_rng_key(), dropout, x)
        x = layer_norm(x)

        x = hk.Conv1D(output_channels=self.ff_dim, kernel_shape=1, padding="same")(x)
        x = hk.BatchNorm(True, True, decay_rate=0.9, eps=1e-6, scale_init=hki.Constant(1.0), offset_init=hki.Constant(1e-8))(x, is_training)
        x = jnn.gelu(x)
        x = hk.Conv1D(output_channels=out_features, kernel_shape=1, padding="same")(x)
        x = hk.BatchNorm(True, True, decay_rate=0.9, eps=1e-6, scale_init=hki.Constant(1.0), offset_init=hki.Constant(1e-8))(x, is_training)
        x = hk.dropout(hk.next_rng_key(), dropout, x)
        x = jnn.gelu(x)

        return layer_norm(x + inputs)


class TimeDistributed(hk.Module):
    def __init__(self, module, batch_first=True):
        super().__init__()
        self.batch_first = batch_first
        self.module = module

    def __call__(self, x):
        module = self.module
        if len(x.shape) <= 2:
            return module(x)

        x_reshape = einops.rearrange(x, "b c h -> (b c) h")

        y = module(x_reshape)

        return jnp.where(self.batch_first, jnp.reshape(y, newshape=(x.shape[0], -1, y.shape[-1])), jnp.reshape(y, newshape=(-1, x.shape[1], y.shape[-1])))


class TransformerThunk(hk.Module):
    def __init__(self, time2vec_dim=1, num_heads=2, head_size=128, ff_dim=None, num_layers=1, dropout=0):
        super().__init__()
        self.time2vec_dim = time2vec_dim
        if ff_dim is None:
            self.ff_dim = head_size
        else:
            self.ff_dim = ff_dim
        self.dropout = dropout
        self.num_heads = num_heads
        self.head_size = head_size
        self.num_layers = num_layers

    def __call__(self, inputs, is_training=True):
        time2vec = Time2Vec(kernel_size=self.time2vec_dim)
        time_embedding = TimeDistributed(time2vec)(inputs)

        x = jnp.concatenate([inputs, time_embedding], axis=-1)
        
        for i in range(self.num_layers):
            x = AttentionBlock(num_heads=self.num_heads, head_size=self.head_size, ff_dim=self.ff_dim, dropout=self.dropout)(x, is_training)
        t = einops.rearrange(x, 't c b -> t (c b)')
        out = hk.Linear(1)(t)
        out = jnn.sigmoid(out)
        return hk.get_parameter('scl', shape=(1,), init=hki.Constant(1.0)) * out + hk.get_parameter('offs', shape=(1,), init=hki.Constant(1e-8))


load_dataset()