import jax
#from jax import vmap
from jax import numpy as jnp
#import torch
from jax import random
from flax import linen as nn
#from flax.training import train_state, checkpoints
#import optax
from typing import Tuple, Callable, NamedTuple, Any
import numpy as np
from activation import KeLu




"""
global_key = jax.random.PRNGKey(1)
x = jax.random.normal(global_key, (10,10))
y = jax.random.normal(global_key, (1, 100, 100,1)).squeeze(-1)

"""

"""
l = nn.BatchNorm(use_running_average = False)
p = l.init(global_key, np.random.randn(5,10,10,5))
m = np.random.randn(5, 10,10,5)
m.std(axis=(0,1,2))
m.mean(axis=(0,1,2))
x, batch_states = l.apply(p, m, mutable =["batch_stats"])
p["batch_stats"] = batch_states["batch_stats"]
batch_states["batch_stats"]
batch_states
"""
class patcher(nn.Module):

    dtype: Any
    activation:Callable = nn.gelu
    embedding_dim:int = 768
    patch_size:int = 8
    debug_mode:bool = False
    precision:Any = jax.lax.Precision("bfloat16")
    
    @nn.compact    
    def __call__(self, 
                 x, 
                 train:bool = False):
        x = nn.Conv(self.embedding_dim, 
                            kernel_size = (self.patch_size, self.patch_size), 
                            strides = self.patch_size, dtype = self.dtype, precision = self.precision,
                            kernel_init =  nn.initializers.variance_scaling(scale = 0.9, mode="fan_in", distribution="uniform"))(x)
        x = self.activation(x)
        x = nn.BatchNorm(use_running_average = not train, dtype = self.dtype)(x)
        if self.debug_mode:
            self.sow('intermediates', 'patcher_features', x)
        return x

"""
patch_ = patcher(activation = nn.relu, dtype = jnp.float32, debug_mode = True)
params = patch_.init(global_key, np.random.randn(224,224, 3))
Y = patch_.apply(params, np.random.randn(1, 224,224, 3),  mutable=['batch_stats', "intermediates"], train = True)
Y[1]["intermediates"]
"""

class block(nn.Module):
    
    dtype:Any 
    kernel_size:int = 5
    embedding_dim:int = 768
    activation:Callable = nn.gelu
    debug_mode:bool = False
    precision:Any = jax.lax.Precision("bfloat16")

    @nn.compact
    def __call__(self, 
                 input:jnp.ndarray, 
                 train:bool = False):
        x = nn.Conv(features = self.embedding_dim, 
                kernel_size = (self.kernel_size, self.kernel_size), 
                feature_group_count = self.embedding_dim, 
                dtype = self.dtype, 
                precision = self.precision,
                kernel_init =  nn.initializers.variance_scaling(scale = 0.9, mode="fan_in", distribution="uniform"))(input)
        x = self.activation(x)
        x = nn.BatchNorm(use_running_average = not train, name="batch_1", dtype = self.dtype)(x)
        x += input
        x = nn.Conv(features = self.embedding_dim, 
                    kernel_size = (1,1), 
                    dtype = self.dtype, 
                    precision = self.precision,
                    kernel_init =  nn.initializers.variance_scaling(scale = 0.9, mode="fan_in", distribution="uniform"))(x)
        x = self.activation(x)
        x = nn.BatchNorm(use_running_average = not train, name = "batch_2", dtype = self.dtype)(x)
        if self.debug_mode:
            self.sow('intermediates', 'feature_block', x)
        return x

"""
import numpy as np
b = block(dtype = jnp.bfloat16, debug_mode = True, precision = jax.lax.Precision("bfloat16"))        
params_ = b.init(key, np.random.randn(1, 32,32,768), train = False)
params_
params_["intermediates"]
params = params_["params"]
batch_stats = params_['batch_stats']
x_, q = b.apply({"params":params, "batch_stats":batch_stats}, np.random.randn(10, 32,32,768), train = True, mutable = ["batch_stats", "intermediates"])
q["intermediates"]["feature_block"]
batch_stats = q["batch_stats"]
"""

class output(nn.Module):
    dtype:Any
    num_classes:int = 1000
    debug_mode:bool = False
    precision:Any = jax.lax.Precision("bfloat16")

    @nn.compact
    def __call__(self, x):
        x = nn.avg_pool(x, (x.shape[1], x.shape[2]))
        x = nn.Dense(self.num_classes, 
                     dtype = self.dtype, 
                     precision = self.precision,
                     kernel_init =  nn.initializers.variance_scaling(scale = 1.0, mode="fan_in", distribution="uniform")
                     )(x.squeeze())
        if self.debug_mode:
            self.sow('intermediates', 'feature_output', x)
        return x


"""
p = output(dtype = jnp.bfloat16, precision = jax.lax.Precision("float32"))
param = p.init(global_key, np.random.randn(5, 224, 224, 3))
p.apply(param, np.random.randn(5, 224, 224, 3))
params = o.init(global_key, np.random.randn(1, 32, 32, 768))
_, params_ = o.apply(params,  np.random.randn(1, 32, 32, 768), mutable = ["intermediates", "batch_stats"], train = True)
jax.tree_util.tree_map(lambda x: x.std(0).mean(), params_["intermediates"])
p.tabulate(global_key, np.random.randn(5, 224, 224, 3))
"""

class mixer(nn.Module):
    
    dtype_:Any
    n_blocks:int = 24
    kernel_size:int = 4
    embedding_dim:int = 768
    patch_size:int = 4
    num_classes:int = 1000
    activation_map:Callable = KeLu
    debug_mode:bool = False
    precision: Any = jax.lax.Precision("bfloat16")

    def setup(self):
        self.blocks = [
            block(
                self.dtype_,
                self.kernel_size,
                self.embedding_dim,
                self.activation_map,
                self.debug_mode,
                self.precision,
            ) for _ in range(self.n_blocks)
        ]

        self.output = output(self.dtype_, self.num_classes, self.debug_mode, self.precision)

        self.pathcher = patcher(
            self.dtype_,
            self.activation_map,
            self.embedding_dim,
            self.patch_size,
            self.debug_mode,
            self.precision
        )

    def __call__(self, x, train:bool = False):
        x = self.pathcher(x, train)
        for layer in self.blocks:
            x = layer(x, train)
        return self.output(x)


"""
global_key = jax.random.PRNGKey(1)
o = mixer(dtype_ = jnp.bfloat16, num_classes = 10 ,n_blocks = 30, debug_mode = True, activation_map = KeLu, embedding_dim =128, precision = jax.lax.Precision("bfloat16"))
x = jax.random.normal(global_key, (10,32,32,3))
params_ = o.init(global_key, x)   
output = o.apply(params_, x)
batch_stats = params_["batch_stats"]
x, params_ = o.apply(params_, x, train = True, mutable = ["intermediates","batch_stats"])
params_.keys()
jax.tree_util.tree_map(lambda x: x.std(-1).mean(), params_["intermediates"])

"""

"""
import flax
import optax
from flax import linen as nn



x = np.random.randn(2,10)
y = np.random.randint(0, 10, size = 2)

y_oh = nn.one_hot(y, 10)
-(jnp.log(nn.sigmoid(x))*y_oh+jnp.log(1-nn.sigmoid(x))*(1-y_oh))
optax.losses.sigmoid_binary_cross_entropy(x, )
"""


