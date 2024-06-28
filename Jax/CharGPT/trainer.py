import sys
sys.path.insert(0, '/home/sahmaran/Desktop/Git_Repos/KeLu/Jax')
import os
os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.95'
#os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import tensorflow as tf
tf.config.experimental.set_visible_devices([], 'GPU')
import jax
import numpy as np
from jax import numpy as jnp
from jax import jit
import optax
from flax import linen as nn
from flax.training import train_state, checkpoints
import optax
#from typing import Any, Callable
from dataclasses import dataclass
from activation import KeLu
from wandb_logger import wandb_loss_logger

from char_gpt import Model, generate, decode
from data_loader import return_dataset



"""
#jax.config.update('jax_disable_most_optimizations', False)
"""


@dataclass
class TraininingArgs:
    # Jax internals
    global_key:int = 0
    ## Project Details
    project_details= {
    "project_name":"KeLu_vs_Gelu_CharGPT",
    "group_name": "KeLu",
    "name": "gelu_deeper",
    }
    ## Train Kwargs
    train_kwargs = {
        "training_precision":jnp.bfloat16,
        "lr":0.001,
        "weight_decay": 1e-2,
        "warm_up_steps": 10,
        "gradient_accumulation": 3,
    }
    ## Model Kwargs
    model_kwargs = {
        "activation": KeLu,
        "vocab_size": 117,
        "ffn_dropout_rate": 0.2,
        "attn_dropout_rate":0.2,
        "use_bias": True,
        "embedding_dim": 384,
        "max_len":128,
        "num_blocks": 12,
        "num_heads":6,
        "precision": jax.lax.Precision("bfloat16"),
    }
    dataset_kwargs = {
        "path": os.path.join("/home/sahmaran/Desktop/Git_Repos/KeLu/Jax/CharGPT","test.txt"),
        "batch_size":768,
        "context_length":64,
        "epochs":3,
    }

def return_model(seed = 0, **kwargs):
    model = Model(jnp.bfloat16,
                  **kwargs,
                  )
    key = jax.random.PRNGKey(0)
    x = jax.numpy.array([[2 for i in range(10)]])
    params_ = model.init({"params":key, "dropout":key}, x = x, causal= True)
    return params_["params"], model



@jit
def train_step(state, x, dropout_key, label_smoothing:bool = False):
    x_batched, y_batched = x[:, :-1], x[:, 1:]
    def loss_fn(params):
        dropout_train_key = jax.random.fold_in(key=dropout_key, data=state.step)
        
        logits = state.apply_fn({"params":params}, 
                            x_batched, 
                            training = True,
                            rngs = {"dropout":dropout_train_key}
                            )
        num_classes = logits.shape[-1]
      
        y_batched_ = jax.nn.one_hot(y_batched, num_classes = num_classes)
        ### We do here a bit label smooting this is good we believe!!!
        if label_smoothing:
            y_batched_ = (1-0.1)*y_batched_ + 0.1/num_classes
        logits = jnp.array(logits, dtype = jnp.float32)
        return optax.softmax_cross_entropy(logits, y_batched_).mean(), logits

    grad_fn = jax.value_and_grad(loss_fn, has_aux = True)

    (loss, logits), grads = grad_fn(state.params) 
    state = state.apply_gradients(grads=grads)## Fix here
    metrics = {'loss': loss, "grads":optax.global_norm(grads)}
    return state, metrics

@jit
def val_step(state, x):
    x_batched, y_batched = x[:, :-1], x[:, 1:]
    def loss_fn(params):
        
        logits = state.apply_fn({"params":params}, 
                            x_batched, 
                            training = False)
        num_classes = logits.shape[-1]
      
        y_batched_ = jax.nn.one_hot(y_batched, num_classes = num_classes)

        logits = jnp.array(logits, dtype = jnp.float32)
        return optax.softmax_cross_entropy(logits, y_batched_).mean(), logits

    loss, logits = loss_fn(state.params)
    metrics = {'loss': loss}
    return metrics

class TrainState(train_state.TrainState):
    key: jax.Array
    ## This time this dude is empty

def main():
    
    args = TraininingArgs()
    model_kwargs = args.model_kwargs
    logger = wandb_loss_logger(**{**args.project_details, 
                                  **args.model_kwargs})
    
    params, model = return_model(**model_kwargs)
    train, val, tokenizer, detokenizer = return_dataset(**args.dataset_kwargs) ## find here vocab_size!!!

    

    state = TrainState.create(apply_fn=model.apply,
                              params=params,
                              key = jax.random.key(seed=0),
                              tx=optax.adamw(learning_rate=3e-4))
                            
    
    train_kwargs = args.train_kwargs
    loss_train = 4.76 ## log(117)
    loss_validation = 4.76
    for j, x in enumerate(train):
            
        x_ = jnp.asarray(x)
        x_.shape
        state, metrics = train_step(state, x_, jax.random.PRNGKey(j)) ## we need a dropout here!!!
        
        loss_train = 0.1*loss_train + 0.9*metrics["loss"]

        logger.log({"Train_loss": loss_train})
        
        if j % 1000 == 1:
            print(loss_train)

            
    print("A full sentence being generated...")
    tokenized_array = generate(model, 
                               {"params":state.params}, 
                               tokenizer, 
                               prompt= "PKK",
                               max_len = 1024, 
                               seed = 10)
        
    result = decode(tokenized=tokenized_array, 
                    detokenizer=detokenizer)
    print(f"GENERATED SENTENCE:", result)
        
        
    counter = 0
    for x_ in val:
        x_ = jnp.asarray(x_)
        metrics = val_step(state, x_)
        print(metrics)
        loss_validation += (metrics["loss"] - loss_validation)/(counter+1)
        counter += 1
        
    print(loss_validation)
    logger.log({"Validation_loss": loss_validation/counter})

                    
    return None




if __name__ == '__main__':
    main()
