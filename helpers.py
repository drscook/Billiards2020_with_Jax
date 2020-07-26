from math import log10, ceil
import numpy as np
import scipy.stats as stats
try:
    import jax
    import jaxlib
    import jax.numpy as jnp
    jax.config.update('jax_enable_x64', True)
    print(f"Using jax {jax.__version__} and jaxlib {jaxlib.__version__}")
    JAX_AVAILABLE = True
except:
    print('Error importing jax; importing numpy as jnp instead')
    import numpy as jnp
    JAX_AVAILABLE = False


##################### Utility Functions #####################
def arrify(x):
    try:
        x.shape
    except:
        x = jnp.asarray(x) 
    if x.ndim == 0:
        x = x.reshape(-1)
    return x

def metric_prefix(x, sigfigs=3):
    s = round(sigfigs)
    assert s > 0, f'sigfigs must round to > 0, got {sigfigs}'
    prefixes = [(12,'T'), (9,'G'), (6,'M'), (3,'k'), (0,''), (-3,'m'), (-6,'µ'), (-9,'n'), (-12,'p')]
    a = ceil(log10(x))
    for b, pref in prefixes:
        if a > b:
            c = s - a + b
            y = round(x/(10**b), c)
            if c <= 0:
                num = int(y)
            else:
                num = format(y, f'.{c}f')
            return f'{num} {pref}'


##################### OVERLY FANCY CUSTOM TIMER #####################
import functools
import time
from dataclasses import field#, dataclass
from datadict import dataclass
from typing import Any, Callable, ClassVar, Dict, Optional

class TimerError(Exception):
    """A custom exception used to report errors in use of Timer class"""

@dataclass
class Timer:
    """Time your code using a class, context manager, or decorator"""

    timers: ClassVar[Dict[str, float]] = dict()
    name: Optional[str] = None
    # text: str = "Mean elapsed time: {:0.4f} seconds over {} loops"
    text: str = "Mean elapsed time: {}s over {} loops"
    logger: Optional[Callable[[str], None]] = print
    _start_time: Optional[float] = field(default=None, init=False, repr=False)
    loop_time: float = -1.0
    quiet: bool = False

    def __post_init__(self) -> None:
        """Initialization: add timer to dict of timers"""
        if self.name:
            self.timers.setdefault(self.name, 0)
        self.loops = 0
        self.elapsed_time = 0.0

    def __mean_time__(self) -> float:
        if self.loops <= 0:
            self.mean_time = 0.0
        else:
            self.mean_time = self.elapsed_time / self.loops
        return self.mean_time

    def start(self) -> None:
        """Start a new timer"""
        if self._start_time is not None:
            raise TimerError(f"Timer is running. Use .stop() to stop it")
        self._start_time = time.perf_counter()

    def stop(self) -> float:
        """Stop the timer"""
        if self._start_time is None:
            raise TimerError(f"Timer is not running. Use .start() to start it")

        # Calculate elapsed time
        self.elapsed_time += (time.perf_counter() - self._start_time)
        self.loops += 1
        self._start_time = None
        if self.name:
            self.timers[self.name] = self.__mean_time__()
        return self.__mean_time__()

    def __report__(self) -> None:
        # Report elapsed time
        decimals = 2
        if self.logger:
            self.logger(self.text.format(metric_prefix(self.__mean_time__()), self.loops))

    def end(self) -> float:
        """Stop the timer, and report the elapsed time"""
        if self._start_time is not None:
            self.stop()
        if not self.quiet:
            self.__report__()
        return self.__mean_time__()

    def __enter__(self) -> "Timer":
        """Start a new timer as a context manager"""
        self.start()
        return self

    def __exit__(self, *exc_info: Any) -> None:
        """Stop the context manager timer"""
        self.end()

    def __call__(self, func) -> "Decorator":
        """Support using Timer as a decorator w/added looping functionality"""
        @functools.wraps(func)
        def wrapper_timer(*args, **kwargs):
            if self.loop_time > 0:
                # Execute function once to clear initial overhead before timing
                out = func(*args, **kwargs)
                while self.elapsed_time < self.loop_time:
                    self.start()
                    func(*args, **kwargs)
                    self.stop()
                self.end()
                return out
            else:
                with self:
                    return func(*args, **kwargs)
        return wrapper_timer


##################### Numpy & Jax Standardization #####################
def outer(x, y, func=jnp.multiply):
    a, b = arrify(x), arrify(y)
    a, b = a.reshape(list(a.shape) + [1] * b.ndim), b.reshape(a.ndim * [1] + list(b.shape))
    return func(a, b)

def put(arr, vals=0, idx=None):
    try:
        arr = arr.at[idx].set(vals)  # jax
    except:
        arr[idx] = vals  # numpy
    return arr

def add(arr, vals=0, idx=None):
    try:
        arr = arr.at[idx].add(vals)  # jax
    except:
        arr[idx] += vals  # numpy
    return arr

def mul(arr, vals=1, idx=None):
    try:
        arr = arr.at[idx].mul(vals)  # jax
    except:
        arr[idx] *= vals  # numpy
    return arr

def minimum(arr, vals=np.PINF, idx=None):
    try:
        arr = arr.at[idx].min(vals)  # jax
    except:
        arr[idx] = jnp.minimum(arr[idx], vals)  # numpy
    return arr

def maximum(arr, vals=np.NINF, idx=None):
    try:
        arr = arr.at[idx].max(vals)  # jax
    except:
        arr[idx] = jnp.maximum(arr[idx], vals)  # numpy
    return arr

class jax_RandomState():
    """
    Jax rng object with same syntax as np.randomState
    """
    def __init__(self, seed=42):
        self.seed = seed
        self.key = key = jax.random.PRNGKey(seed)
        self.tol = 1e-5

    def randint(self, low, high=None, size=1, dtype=None):
        if high is None:
            high = low
            low = 0
        if low > high:
            print("specified low > high; I'll swap them")
            low, high = high, low
        return jax.random.randint(self.key, minval=low, maxval=high,
                                  shape=arrify(size), dtype=dtype)

    def uniform(self, low=0, high=1, size=1, dtype=None):
        return jax.random.uniform(self.key, minval=low, maxval=high,
                                  shape=arrify(size))#, dtype=dtype)

    def normal(self, loc=0, scale=1, size=1):
        x = jax.random.normal(self.key, shape=arrify(size))
        if abs(scale - 1.0) > self.tol:
            x *= scale
        if abs(loc) > self.tol:
            x += loc
        return x

    def multivariate_normal(self, mean=0, cov=None, size=1, dtype=None):
        mean = arrify(mean)
        dim = mean.shape[0]
        if cov is None:
            cov = jnp.eye(dim)
            print(f"cov not specified; using identity matrix")
        cov = self._arrayify(cov)
        sh = cov.shape
        if (len(sh) != 2) or (sh[0] != dim) or (sh[0] != sh[1]):
            raise Exception(f"cov has shape {sh}; should be ({dim}, {dim})")
        if dim == 1:
            return self.normal(loc=mean[0], scale=cov[0,0], size=size, dtype=dtype)
        else:
            return jax.random.multivariate_normal(self.key, mean=mean, cov=cov,
                                                  shape=arrify(size), dtype=dtype)

    def choice(self, a, replace=True, p=None, size=1):
        try:
            return jax.random.choice(self.key, a, replace=replace, p=p, shape=arrify(size))
        except Exception as e:
            print(f"You're running jax {jax.__version__}, but random.choice appears to be implemented in jax 0.1.71.  Sadly, jax > 0.1.69 can not utilize colab GPU (as of 2020-07-11).")
            print(e)
        return 

    def permutation(self, x):
        return jax.random.permutation(self.key, x)


##################### Smart Chooser Between Numpy & Jax #####################
def pick_jnp(num_part=1, seed=42, force_numpy=False, force_jax=False):
    """
    Determines whether to use jax or numpy.  Can force it, or allow detection.
    """
    global jnp, USE_JAX, nx, jx
    if (force_jax is True) and (JAX_AVAILABLE is True):
        print("Given force_jax = True")
        USE_JAX = True
    elif force_numpy is True:
        print("Given force_numpy = True")
        USE_JAX = False
    elif JAX_AVAILABLE is False:
        print("Jax not available")
        USE_JAX = False
    else:
        max_time = 4
        import numpy as np
        @Timer(name='np', loop_time=max_time, quiet=True)
        def f_np(x):
            return outer(x, x, np.subtract)

        import jax.numpy as jnp
        @Timer(name='jax', loop_time=max_time, quiet=True)
        def f_jax(x):
            return outer(x, x, jnp.subtract)

        nx = np.random.uniform(size=num_part); jx = jnp.array(nx)
        f_np(nx); f_jax(jx)
        nt, jt = Timer.timers['np'], Timer.timers['jax']
        USE_JAX = jt < nt
        print(f"For {num_part} particles, numpy time = {metric_prefix(nt)}s vs jax time = {metric_prefix(jt)}s")
    
    if USE_JAX is True:
        print("importing jax.numpy as jnp.")
        import jax.numpy as jnp
        rng = jax_RandomState(seed)
    else:
        print("importing numpy as jnp.")
        import numpy as jnp
        rng = np.random.RandomState(seed)
    return rng