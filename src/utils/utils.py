from contextlib import contextmanager
from time import time


@contextmanager
def timer(name: str):
    t0 = time()
    print(f"[{name}] start")
    yield
    print(f"[{name}] done in {time() - t0:.0f} s")
