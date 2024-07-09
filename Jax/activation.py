### Define KeLu here!!!

from jax import jit
from jax import numpy as jnp
from jax._src.numpy import util as numpy_util
from jax._src.typing import Array, ArrayLike
from jax import lax
from functools import partial
   

@partial(jit, static_argnums = 1)
def KeLu(x_: ArrayLike, a:float = 3.5) -> Array:
    """
    x<-a, 0, x>a, x, else 0.5*x*(1+x/a+(1/jnp.pi)*jnp.sin(x*jnp.pi/a)
    """
    numpy_util.check_arraylike("KeLu", x_)
    x = jnp.asarray(x_)
    return  lax.select(x < -a, jnp.zeros(x.shape).astype(x_.dtype), lax.select(x > a, x, 0.5*x*(1+x/a+(1/jnp.pi)*jnp.sin(x*jnp.pi/a).astype(x_.dtype))))


if __name__ == "__main__":
    pass
