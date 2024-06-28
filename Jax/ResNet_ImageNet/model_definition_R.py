import os
#os.environ['CUDA_VISIBLE_DEVICES'] = '1'


import jax
#from jax import vmap, jit, pmap
from jax import numpy as jnp
#import torch
#from jax import random
from flax import linen as nn
#from flax.training import train_state, checkpoints
#import optax
from typing import Tuple, Callable, NamedTuple, Any, Union
import numpy as np
#from activation import KeLu
from dataclasses import dataclass, field

#KeLu = lambda x: x

key = jax.random.PRNGKey(10)
 

## TODO : Variance scaling is important!
## TODO: introduce SOW -- to look at the intermediate layer activations
## TODO: Better to do training for seperately for gelu as well
## variance_scaler = nn.initializers.variance_scaling(scale = 0.01, distribution = "normal", mode = "fan_in")


class block_0(nn.Module):

    dtype: Any
    #activation:Callable = KeLu
    precision:Any = jax.lax.Precision("bfloat16")

    @nn.compact
    def __call__(self, 
                 x:jnp.ndarray, 
                 train:bool = True):
        x = nn.Conv(features=64, 
                    kernel_size=(7,7), 
                    strides = (2,2),
                    padding='same', 
                    precision = self.precision, dtype = self.dtype)(x)
        x = nn.max_pool(x, window_shape = (2,2), strides = (2,2))
        return x
class block_1(nn.Module):
    dtype: Any
    activation: Callable = KeLu
    precision:Any = jax.lax.Precision("bfloat16")
    dims:list[Union[list, None]]  = field(default_factory = lambda: [64, 256])
    debug:bool = False

    @property
    def kwargs(self): 
        return {"dtype":self.dtype, "precision": self.precision}


    @nn.compact
    def __call__(self, x, train = True):

        bottleneck_dim, output_dim = self.dims

        x = nn.BatchNorm(use_running_average = not train, dtype = self.dtype)(x)
        x = self.activation(x)
        x_ = nn.Conv(features = output_dim , kernel_size = (1,1), padding='same', dtype = self.dtype)(x)
        x = nn.Sequential([nn.Conv(features = bottleneck_dim, 
                                   kernel_size = (1,1), 
                                   padding='same', use_bias = False,
                                   **self.kwargs),                                   
                           nn.BatchNorm(use_running_average = not train, dtype = self.dtype),
                           self.activation, 
                           nn.Conv(features = bottleneck_dim, kernel_size = (3,3), padding='same', use_bias = False, **self.kwargs),
                           nn.BatchNorm(use_running_average = not train, dtype = self.dtype),
                           self.activation, 
                           ], name = "sequential_block")(x)
        x = nn.Conv(features = output_dim , kernel_size = (1,1), padding='same',  name = "Output", **self.kwargs)(x)
        x = jnp.add(x, x_)
        if self.debug:
                self.sow('intermediates', 'features', x)
        return x


"""
b = block_1(dtype = jnp.bfloat16, dims = [64, 256], debug = True)
params =b.init(key, jnp.ones((1,56,56,64)))

t = b.apply(params, jnp.ones((1,56,56,64)),mutable=['batch_stats', "intermediates"], train = False)[1]["intermediates"]

jax.tree_util.tree_map(lambda x: x.var(),t)

tabulate_fn = nn.tabulate(block_1(dtype = jnp.bfloat16), jax.random.key(0))
print(tabulate_fn(jnp.ones((1,56,56,64))))

"""

class block_2(nn.Module):
    dtype: Any
    activation: Callable = KeLu
    precision:Any = jax.lax.Precision("bfloat16")
    dims:list[Union[list, None]]  = field(default_factory = lambda: [64, 256])
    debug:bool = False

    
    @property
    def kwargs(self): 
        return {"dtype":self.dtype, "precision": self.precision}
    
    @nn.compact
    def __call__(self, x_, train = True):


        bottleneck_dim, output_dim = self.dims

        x = nn.BatchNorm(use_running_average = not train, dtype = self.dtype)(x_)
        x = self.activation(x)
        x = nn.Sequential([nn.Conv(features = bottleneck_dim, 
                                kernel_size=(1,1),
                                padding='same',
                                use_bias = False,
                                **self.kwargs),
                           nn.BatchNorm(use_running_average = not train, 
                                        dtype = self.dtype),
                           self.activation, 
                           nn.Conv(features = bottleneck_dim, 
                                   kernel_size=(3,3), 
                                   padding='same', 
                                   use_bias = False,
                                   **self.kwargs),
                           nn.BatchNorm(use_running_average = not train, 
                                        dtype = self.dtype),
                           self.activation, 
                           ])(x)
        x = nn.Conv(features = output_dim, 
                    kernel_size=(1,1), 
                    padding='same',  
                    **self.kwargs)(x)
        x = jnp.add(x, x_)
        if self.debug:
            self.sow('intermediates', 'features', x)
        return x

"""
params = block_2(dtype = jnp.bfloat16, dims = [None, 64, 256]).init(key, jnp.ones((1,56,56,64)))
block_2(dtype = jnp.bfloat16).apply(params, jnp.ones((1,56,56,256)),mutable=['batch_stats'], train = True)[0].shape
tabulate_fn = nn.tabulate(block_2(dtype = jnp.bfloat16), jax.random.key(0))
print(tabulate_fn(jnp.ones((1,56,56,256))))

"""


class block_3(nn.Module):
    dtype: Any
    activation: Callable = KeLu
    precision:Any = jax.lax.Precision("bfloat16")
    dims:list[Union[list, None]]  = field(default_factory = lambda: [64, 256])
    debug:bool = False
    
    @property
    def kwargs(self): 
        return {"dtype":self.dtype, "precision": self.precision}

    @nn.compact
    def __call__(self, x_, train = True):
        
        bottleneck_dim, output_dim = self.dims

        x = nn.BatchNorm(use_running_average = not train, dtype = self.dtype)(x_)
        x = self.activation(x)
        x = nn.Sequential([nn.Conv(features =  bottleneck_dim, kernel_size=(1,1), padding='same', use_bias = False,
                                   **self.kwargs),
                           nn.BatchNorm(use_running_average = not train, dtype = self.dtype),
                           self.activation, 
                           nn.Conv(features = bottleneck_dim, kernel_size=(3,3), strides = (2,2), padding='same', use_bias = False, **self.kwargs),
                           nn.BatchNorm(use_running_average = not train, dtype = self.dtype),
                           self.activation, 
                           ])(x)
        x = nn.Conv(features = output_dim, kernel_size=(1,1), padding='same', **self.kwargs)(x)
        x_ = nn.max_pool(x_, window_shape =(2,2), strides = (2,2), padding = "same")
        x = jnp.add(x, x_)
        if self.debug:
            self.sow('intermediates', 'features', x)
        return x
"""
params = block_3(dtype = jnp.bfloat16, dims = [64, 256]).init(key, jnp.ones((1,56,56,256)))
block_3(dtype = jnp.bfloat16).apply(params, jnp.ones((1,56,56,256)),mutable=['batch_stats'], train = True)[0].shape
tabulate_fn = nn.tabulate(block_3(dtype = jnp.bfloat16), jax.random.key(0))
print(tabulate_fn(jnp.ones((1,56,56,256))))

"""

class output(nn.Module):

    dtype: Any
    activation: Callable = KeLu
    precision:Any = jax.lax.Precision("bfloat16")
    output_dim:int = 1000
    dropout_rate:float = 0.4
    debug: bool = False


    @nn.compact
    def __call__(self, x, train = True):
        x = nn.BatchNorm(use_running_average = not train, dtype = self.dtype)(x)
        x = self.activation(x)
        x = x.mean([-2, -3])        
        x = nn.Dense(self.output_dim, precision = self.precision, dtype = self.dtype)(x)
        
        if self.debug:
            self.sow('intermediates', 'features', x)
        return nn.Dropout(rate = self.dropout_rate, deterministic = not train)(x)

""""
model = output(**{"dtype":jnp.bfloat16})
params = model.init(key, jnp.ones((10,10,10)), train = False)
main_key, params_key, dropout_key = jax.random.split(key=key, num=3)
model.apply(params, jnp.ones((10,10,10)), rngs = {"dropout": dropout_key}, train =True, mutable=['batch_stats'])
"""
class Resnet50(nn.Module):

    dtype: Any
    output_dim:int = 1000
    activation: Callable = KeLu
    precision:Any = jax.lax.Precision("bfloat16")
    dropout_rate:float = 0.4
    debug: bool = False

    @property
    def kwargs(self): 
        return {"dtype":self.dtype, "precision": self.precision, "activation": self.activation, "debug": self.debug}


    def setup(self):
        self.layers = nn.Sequential([
            block_0(**{"dtype":self.dtype, "precision":self.precision}),
            block_1(dims = [64, 256],**self.kwargs),
            block_2(dims = [64, 256],**self.kwargs),
            block_3(dims = [64, 256],**self.kwargs),
            block_1(dims = [128, 512],**self.kwargs),
            block_2(dims = [128, 512],**self.kwargs),
            block_2(dims = [128, 512],**self.kwargs),
            block_3(dims = [128, 512],**self.kwargs),
            block_1(dims = [256, 1024],**self.kwargs),
            block_2(dims = [256, 1024],**self.kwargs),
            block_2(dims = [256, 1024],**self.kwargs),
            block_2(dims = [256, 1024],**self.kwargs),
            block_2(dims = [256, 1024],**self.kwargs),
            block_3(dims = [256, 1024],**self.kwargs),
            block_1(dims = [512, 2048],**self.kwargs),
            block_2(dims = [512, 2048],**self.kwargs),
            block_2(dims = [512, 2048],**self.kwargs),
        ])

        self.output_layer = output(output_dim = self.output_dim, dropout_rate = self.dropout_rate, **self.kwargs)

    def __call__(self, x, train = True):
        x = self.layers(x, train)
        return self.output_layer(x, train = train)
    
"""
model = Resnet50(dtype = jnp.bfloat16, debug = True)
params = model.init(key, jnp.ones((1,32,32,3)), train = False)
#tabulate_fn = nn.tabulate(Resnet50(dtype = jnp.bfloat16), jax.random.key(0))
#print(tabulate_fn(jnp.ones((1,32,32,3))))

main_key, params_key, dropout_key = jax.random.split(key=key, num=3)
t = model.apply(params, jnp.ones((50,224,224,3)), rngs = {"dropout": dropout_key}, train =True, mutable=['batch_stats', "intermediates"])
jax.tree_util.tree_map(lambda x: x.std(-1).mean().block_until_ready(), t[1]["intermediates"])
"""
