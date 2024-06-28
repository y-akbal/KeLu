import jax
from jax import numpy as jnp
#from jax import grad, vmap, jit
#import flax
from flax import linen as nn
from typing import Any, Callable, Optional
import numpy as np


class FFN(nn.Module):
    
    dtype: Any
    activation:Callable = nn.gelu
    ffn_dropout_rate:float = 0.2
    embedding_dim:int = 768
    precision:Any = jax.lax.Precision("bfloat16")
    use_bias: bool = True

    @classmethod
    def from_kwargs(cls, **kwargs):
        att = ["dtype", "activation", "ffn_dropout_rate", "embedding_dim", "precision", "use_bias"]
        cls.att = {a:kwargs[a] for a in att}
        return cls(**cls.att)
    
    @nn.compact      
    def __call__(self, x, training = False):
    
        x = nn.Sequential([
            nn.Dense(4 * self.embedding_dim, dtype = self.dtype, precision = self.precision, use_bias = self.use_bias),
            self.activation,
            nn.Dense(self.embedding_dim, dtype = self.dtype, precision = self.precision, use_bias = self.use_bias),
            nn.Dropout(self.ffn_dropout_rate, deterministic = not training)])(x)
        return x

"""
m = FFN(jnp.bfloat16, bias = False)    
params = m.init(jax.random.PRNGKey(0), jnp.ones((1, 1, 768)))
m.apply(params, 2*jax.random.normal(jax.random.PRNGKey(0), (1, 1, 768)))
print(m.tabulate(jax.random.PRNGKey(0), 2*jax.random.normal(jax.random.PRNGKey(0), (1, 1, 768))))
"""   
class output(nn.Module):
    
    dtype: Any
    vocab_size:int = 55000
    embedding_dim:int = 768
    precision: int = jax.lax.Precision("bfloat16")
    

    @classmethod
    def from_kwargs(cls, **kwargs):
        att = ["dtype", "vocab_size", "precision", "embedding_dim"]
        cls.att = {a:kwargs[a] for a in att}
        return cls(**cls.att)
    
    @nn.compact      
    def __call__(self, x, training:bool = None):
        x = nn.LayerNorm(self.embedding_dim)(x)
        x = nn.Dense(self.vocab_size, dtype = self.dtype, precision = self.precision)(x)
        return x
    
"""
m = output(jnp.bfloat16)    
params = m.init(jax.random.PRNGKey(0), jnp.ones((1, 10, 32)))
m.apply(params, 2*jax.random.normal(jax.random.PRNGKey(0), (1, 10, 32)))
m.tabulate(jax.random.PRNGKey(0), 2*jax.random.normal(jax.random.PRNGKey(0), (1, 1, 32)))
"""    

class multi_headattention(nn.Module):
    
    dtype: Any
    num_heads:int = 4
    attn_dropout_rate:float = 0.2
    precision:Any = jax.lax.Precision("bfloat16")
    use_bias:bool = False

    @classmethod
    def from_kwargs(cls, **kwargs):
        att = ["dtype","num_heads", "attn_dropout_rate", "precision", "use_bias"]
        cls.att = {a:kwargs[a] for a in att}
        return cls(**cls.att)
    
    @nn.compact      
    def __call__(self, x, mask = None, deterministic = True):
        attn_func = nn.MultiHeadDotProductAttention(dtype = self.dtype, 
                                                    num_heads = self.num_heads, 
                                                    dropout_rate = self.attn_dropout_rate, 
                                                    precision = self.precision)
        
        return attn_func(inputs_q = x, mask = mask, deterministic = deterministic)
    
    

"""
head = multi_headattention(**{"num_heads":5, "dtype":jnp.float16, "attn_dropout_rate":0.1, "precision":jax.lax.Precision("bfloat16")})
x = jax.random.normal(jax.random.key(0), (10,10))
params = head.init(jax.random.key(0), x)
jax.tree_util.tree_map(lambda x: x.shape, params["params"])

layer = nn.LayerNorm()
params = layer.init(jax.random.key(0), x)
layer.apply(params, jnp.ones(10,))

ffn = FFN(jnp.float32)
params =ffn.init(key1, x, training = True)
ffn.apply(params,x, training = True, rngs={'dropout': key1})
e = nn.Embed(100, 2)
params = e.init(jax.random.key(0), jnp.asarray([1,2,3]))
e.apply(params, jnp.asarray([1,2]))
"""

"""
h = nn.MultiHeadAttention(num_heads = 4, use_bias = False, dropout_rate = 0.2) 
key = jax.random.PRNGKey(0)
params = h.init({"params":key, "dropout":key}, np.random.randn(1, 2, 12), deterministic = False)
x = np.random.randn(1,32,12)
mask = nn.make_causal_mask(np.random.randn(1,32))
h.apply(params, x, mask = mask,rngs={"dropout":jax.random.PRNGKey(122)}, deterministic = False).shape
"""
class Block(nn.Module):

    dtype: Any
    activation:Callable = nn.gelu
    ffn_dropout_rate:float = 0.2
    attn_dropout_rate:float = 0.2
    embedding_dim:int = 768
    num_heads:int = 8
    use_bias:bool = False
    precision:Any = jax.lax.Precision("bfloat16")

    @property
    def model_kwargs(self):
        __attrbs__ = ["dtype", "activation", "ffn_dropout_rate", 
                      "attn_dropout_rate","embedding_dim", "num_heads", 
                      "use_bias", "precision"]
        return {attribute:vars(self)[attribute] for attribute in __attrbs__}
    
    @classmethod
    def from_kwargs(cls, **kwargs):
        att = ["dtype", "activation", "ffn_dropout_rate", 
                      "attn_dropout_rate","embedding_dim", "num_heads", 
                      "use_bias", "precision"]
        cls.att = {a:kwargs[a] for a in att}
        return cls(**cls.att)


    @nn.compact         
    def __call__(self, x, mask = None , training = False):
        ## Init submodules
        ## hope rope layer will come here soon!!!
        
        att_head = multi_headattention.from_kwargs(**self.model_kwargs)
        ## 
        ffn = FFN.from_kwargs(**self.model_kwargs)
        ## we are doing here pre-norm stuff
        x = x + att_head(nn.LayerNorm(self.embedding_dim)(x), mask = mask, deterministic = not training)
        ##
        x = x + ffn(nn.LayerNorm(self.embedding_dim)(x), training = training)
        return x



"""
h = nn.Sequential([Block(dtype = jnp.float32, num_heads = 2, embedding_dim = 4) for i in range(2)])
key = jax.random.PRNGKey(1)
np.random.seed(0)
y = np.random.randn(1, 3, 4)
params = h.init(x =y, rngs = key)
mask = nn.make_causal_mask(np.random.randn(1, 3,4)[:,:,0])
mask = None
h.apply(params,y,mask = mask,training = False, rngs = {"dropout":jax.random.PRNGKey(225)})
"""

class Model(nn.Module):
    dtype: Any
    activation:Callable = nn.gelu
    vocab_size:int = 512
    ffn_dropout_rate:float = 0.2
    attn_dropout_rate:float = 0.2
    use_bias:bool = True
    embedding_dim:int = 768
    max_len:int = 512
    num_blocks:int = 12
    num_heads:int = 8
    precision:Any = jax.lax.Precision("bfloat16")

    @property
    def model_kwargs(self):
        __attrbs__ = ["dtype", "activation", "ffn_dropout_rate", 
                      "attn_dropout_rate","embedding_dim", "num_heads", 
                      "use_bias", "precision", "vocab_size"]
        return {attribute:vars(self)[attribute] for attribute in __attrbs__}


    @nn.compact
    def __call__(self, x, training = True, causal = True):

        B, L = x.shape

        mask = None ## in the case that no mask is used!!!
        if causal:
            mask = nn.make_causal_mask(x)
        

        embeddings_ = self.param('embedding', nn.initializers.xavier_uniform(), (self.vocab_size, self.embedding_dim))
        bias_ = self.param('embedding_bias', nn.initializers.zeros_init(), (1,self.vocab_size))

        ## Here we need to absolute positional embeddings may be rope - would be good!!!
        ## place holder for absolute positional embeddings!!!

        position_embeddings = nn.Embed(self.max_len, self.embedding_dim)(jnp.arange(L))
        token_embeddings = embeddings_.take(x, axis = 0)

        embedded_sentence = position_embeddings + token_embeddings
        attention_blocks = nn.Dropout(self.ffn_dropout_rate, deterministic = not training)(embedded_sentence)
        blocks = [
            Block.from_kwargs(**self.model_kwargs) for i in range(self.num_blocks)
        ]
        for block in blocks:
            attention_blocks = block(attention_blocks, mask = mask,training = training)

        #x = output.from_kwargs(**self.model_kwargs)(attention_blocks)        
        attention_blocks = nn.LayerNorm(self.embedding_dim)(attention_blocks)

        return attention_blocks @ embeddings_.transpose(-1,-2)+bias_
"""
model = Model(dtype = jnp.bfloat16, embedding_dim = 4, vocab_size = 3, num_heads = 2, num_blocks = 2)
key = jax.random.PRNGKey(0)
params = model.init({"params":key, "dropout":key}, x = jax.numpy.array([[0]]), causal= True)
params
output_logits = model.apply(params, jax.numpy.array([[0,1,2,1,0,0,0,0,1,0]]), 
                            training = False, rngs = {"dropout":jax.random.PRNGKey(22)}, 
                            causal = True)
q = nn.Embed(10, 2)
p = q.init(key, jnp.array([1,2]))
q.apply(p, jnp.arange(11))
"""
"""
from data_loader import return_dataset

def split(x):
    x, y = np.split(x, 2, axis =-2)
    return x.squeeze(), y.squeeze() 

for i in return_dataset(batch_size=32):
    x, y = split(i)
    y = model.apply(params, x, training = False, rngs = {"dropout":jax.random.PRNGKey(22)}, causal = True)

y.argmax(-1) == 
"""

def generate(model, 
            params, 
            tokenizer:dict, 
            prompt:str = "This shall too pass", 
            max_len:int = 256,
            seed = 0):    
    ## Sure not the best way autoregressive getting but would be pretty ok!

    key = jax.random.PRNGKey(seed)

    vocab = jnp.array([i for i, _ in enumerate(tokenizer.keys())])

    tokens = [tokenizer[char] for char in prompt]
    
    ## Here you can stop tracking gradients!!! for performance thing!!!
    def sub_generate(key:jnp.array, 
                     x:jnp.array,
                     )->jnp.array:
        "This is goin to be ok"
        "341200000 -> 3412(1 is to be predicted)0000 -- probably 0 might have been different!!"
        output_logits = model.apply(params, x, training = False, causal = True)
        next_token_prob = jax.nn.softmax(output_logits, axis = -1)[0, -1, :]
       
        next_token = jax.random.choice(key = key, 
                                       a = vocab,
                                       shape = (1,), 
                                       p = next_token_prob, 
                                       axis = 0)
        
        return next_token[0].astype(int)

    for i in range(len(prompt), max_len):
        key = jax.random.fold_in(key, i) ## Generate a new key!
        tokens_ = jnp.array([tokens[-64:]])
        x = sub_generate(key, tokens_) ## Get the next token!!!
        tokens.append(x)
        

    return np.array(tokens).squeeze()

def decode(tokenized:np.ndarray, detokenizer:dict)->str:
    str_ = ""
    for char in map(lambda x: detokenizer[x], tokenized.tolist()):
        str_ += char
    return str_

    


"""
tokenizer = {"a":0, "b":1, "c":2}
detokenizer = {i:j for j,i in tokenizer.items()}
a = generate(model, params, tokenizer, prompt = "cbab", max_len=123, seed= 0)
decode(a, detokenizer)
"""    