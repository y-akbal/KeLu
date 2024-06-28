import os
os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.97'
"""os.environ['XLA_FLAGS'] = (
    #'--xla_gpu_enable_triton_softmax_fusion=true '
    #'--xla_gpu_triton_gemm_any=True '
    #'--xla_gpu_enable_async_collectives=true '
    #'--xla_gpu_enable_latency_hiding_scheduler=true '
    #'--xla_gpu_enable_highest_priority_async_stream=true '
    #'--xla_gpu_simplify_all_fp_conversions=true'
    ##"--XLA_PYTHON_CLIENT_MEM_FRACTION=.95"
)"""

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import jax
#import numpy as np
from jax import numpy as jnp
from jax import random, jit, grad, value_and_grad
from flax import linen as nn
from flax.training import train_state, checkpoints
import optax
import wandb
from model_definition import mixer
from typing import Any, Callable
from dataclasses import dataclass
from get_data_tf import return_train_set, return_test_set
#from get_data_pt import return_dataset
from activation import KeLu
import tensorflow as tf
from wandb_logger import wandb_loss_logger

tf.config.experimental.set_visible_devices([], 'GPU')

import numpy as np
import optax




"""# Some fake params.
params = {'w': jax.numpy.ones(10)}

# Use optax.adam, but tell optax that we'd like to move the adam hyperparameters into the optimizer's state.
opt = optax.inject_hyperparams(optax.sgd)(learning_rate=1e-4)
opt_state = opt.init(params)

# We can now set the learning rate however we want by directly mutating the state.
opt_state.hyperparams['learning_rate'] = 1e-5
opt.update(params, opt_state)
"""

@dataclass
class TraininingArgs:
    ## Project Details
    project_name:str = "KeLu_vs_Gelu_CIFAR100_Deeper"
    group_name:str = "gelu"
    name:str = "gelu_deeper"
    ## Train Kwargs
    epochs:int = 200
    training_precision = jnp.bfloat16
    ## Model Kwargs
    n_blocks:int = 12
    kernel_size:int = 5
    embedding_dim:int = 384
    activation:Callable = nn.gelu

    patch_size:int = 2
    precision:str = "bfloat16"
    ## dataset_kwargs
    dataset:str = "cifar100"
    num_classes:int = 100
    batch_size:int = 256
    ## Jax_internals ##
    global_key:int = 0
    ## Optimizer details ##
    lr:float = 0.01
    weight_decay:float = 1e-2
    warm_up_steps:int = 10

    def project_details(self):
        return {"project_name":self.project_name, 
                "group_name":self.group_name,
                "name":self.name
                }

    def model_kwargs(self):
        return {
            "kernel_size":self.kernel_size,
            "embedding_dim":self.embedding_dim,
            "activation_map":self.activation,
            "patch_size":self.patch_size,                
            "n_blocks":self.n_blocks,
            "num_classes":self.num_classes,
                "precision": jax.lax.Precision(self.precision),
            }
    def dataset_kwargs(self):
        return {"dataset":"cifar10",
                "num_classes":self.num_classes
            }
    def train_kwargs(self):
        return {"epochs":self.epochs, 
                "batch_size":self.batch_size,
                "training_precision":self.training_precision
            }
    def optimizer_kwargs(self):
        return {"lr":self.lr,
                "weight_decay":self.weight_decay,
                "warm_up_steps":self.warm_up_steps
            }

#from jax.tree_util import tree_map

def return_model(seed = 0, **kwargs):
    model = mixer(jnp.bfloat16,
                  **kwargs,
                  )
    global_key = jax.random.PRNGKey(seed)
    x = jnp.ones((1,32,32,3))
    params_ = model.init(global_key, x, train = False)
    return params_["params"], params_["batch_stats"], model

"""
x = jnp.array(np.random.randn(150, 32, 32, 3)).astype(jnp.float16)

@jit
def forward(params, batch_stats, x):
    return model.apply({"params":params, "batch_stats":batch_stats},x)

import time 
a = time.time()
q = forward(params, batch_stats, x).block_until_ready()
print(time.time()-a)
#jax.config.update('jax_disable_most_optimizations', False)
"""

class TrainState(train_state.TrainState):
    batch_stats: Any


@jit
def train_step(state, x_batched, y_batched):
    
    def loss_fn(params):
        logits, updates = state.apply_fn({"params":params, "batch_stats":state.batch_stats}, 
                            x_batched, 
                            train = True,                               
                            mutable = ["batch_stats"])
       
        # Tf datasets already mixes the labels in one hot encoded form -- so nor worryrry!!!!        
        #y_batched_ = jax.nn.one_hot(y_batched, num_classes=10)

        ### We do here a bit label smooting this is good we believe!!!
        #labels_onehot = (1-smoooting_fac)*labels_onehot + smoooting_fac/num_classes
        logits = jnp.array(logits, dtype = jnp.float32)
        return optax.losses.sigmoid_binary_cross_entropy(logits, y_batched).mean(), (updates, logits)

    grad_fn = jax.value_and_grad(loss_fn, has_aux = True)

    (loss, (updates, logits)), grads = grad_fn(state.params) 
    ## Scale the grads
    #grads = optax.scale_gradient(grads, 5)
    grad_norm = optax.global_norm(grads)

    state = state.apply_gradients(grads=grads).replace(batch_stats = updates['batch_stats']) ## Fix here
    metrics = {'loss': loss,
               'accuracy': jnp.mean(jnp.argmax(logits, -1) == y_batched.argmax(-1)),
               'grad_norm': grad_norm
               }
    return state, metrics

@jit
def val_step(state, x_batched, y_batched):
    
    def loss_fn(params):
        logits, updates = state.apply_fn({"params":params, "batch_stats":state.batch_stats}, x_batched, 
                              train = False,                               
                              mutable = ["batch_stats"])
        
        num_classes = logits.shape[1]
        
        labels_onehot = jax.nn.one_hot(y_batched, num_classes=num_classes)

        ### We do here a bit label smooting this is good we believe!!!

        return optax.softmax_cross_entropy(logits, labels_onehot).mean(), logits

    loss, logits = loss_fn(state.params)
    metrics = {'loss': loss,
               'accuracy': jnp.mean(jnp.argmax(logits, -1) == y_batched)}
    return state, metrics



def main():

    args = TraininingArgs()
    model_kwargs = args.model_kwargs()

    logger = wandb_loss_logger(**{**args.project_details(), 
                                  **args.model_kwargs()})

 
    train, val = return_train_set(**{"batch_size":args.batch_size}), return_test_set(**{"batch_size":args.batch_size})
    #train, val = return_dataset()

    params, batch_stats, model = return_model(**model_kwargs)
    
    state = TrainState.create(apply_fn=model.apply,
                              params=params,
                              batch_stats = batch_stats,
                              tx=optax.chain(#optax.clip(10),
                                 optax.clip_by_global_norm(5.5),
                                 optax.inject_hyperparams(optax.adamw)(learning_rate=0.0005, weight_decay = 0.01)
                                ))
    
    lr_scheduler = optax.warmup_cosine_decay_schedule(0.00001, 0.01, 15, 150, end_value=0.0003, exponent=1.0)
    #lr_scheduler = lambda t: np.interp([t], [0, args.epochs*3//5, args.epochs*4//5, args.epochs], [0, args.lr, args.lr/20.0, 0])[0]
    #lr_scheduler = optax.join_schedules([scheduler_1, scheduler_2], [2])
    #params, batch_stats, moptax.join_schedulesodel  = return_model(**vars(args))
    #y_pred = model.apply({"params":params, "batch_stats":state.batch_stats}, x, train = False)
    #jnp.mean(jnp.argmax(y_pred, -1) == y.argmax(-1))
    for i in range(args.epochs):
        state.opt_state[1][1]["learning_rate"] = lr_scheduler(i)
        acc_train = 0.0
        loss_train = 0.0
        counter_train = 0
        grad_norm = 0

        acc_val = 0.0
        loss_val = 0.0
        counter_val = 0

        #train_tqdm = tqdm(train)
        for j, (x,y) in enumerate(train):
            x,y = map(lambda t: jnp.asarray(t), [x,y])
            state.opt_state[1][1]["learning_rate"] =  lr_scheduler(i + (j + 1)/len(train))
            #x = jnp.transpose(x, (0, 2, 3, 1))
            state, metrics = train_step(state, x, y)
            acc_train += metrics["accuracy"]
            loss_train += metrics["loss"]
            grad_norm += metrics["grad_norm"]
            counter_train += 1
        #state.opt_state[1].hyperparams["learning_rate"] = 0.1/(i+1)

        for x,y in val:
            x,y = map(lambda t: jnp.asarray(t), [x,y])
            #x = jnp.transpose(x, (0, 2, 3, 1))
            state, metrics = val_step(state, x, y)
            acc_val += metrics["accuracy"]
            loss_val += metrics["loss"]
            counter_val += 1

        lr = float(state.opt_state[1][1]["learning_rate"])

        print(f"Epoch {i}: Lr = {lr}, T.acc.: {acc_train/counter_train}, loss: {loss_train/counter_train}, Val.acc. {acc_val/counter_val}, Val.loss {loss_val/counter_val}, grad_norm {grad_norm/counter_train}")
        
        ## Wandb log--
        logger.log({"T.acc.": acc_train/counter_train, 
                    "Train_loss": loss_train/counter_train, 
                    "Val.acc": acc_val/counter_val, 
                    "Val.loss":loss_val/counter_val, 
                    "grad_norm":grad_norm/counter_train
                    })
    return None




if __name__ == '__main__':
    main()
