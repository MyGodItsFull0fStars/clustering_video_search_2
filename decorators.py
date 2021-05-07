from functools import wraps
import time


def time_decorator(my_func):
    @wraps(my_func)
    def timed(*args, **kw):
        t_start = time.time()
        output = my_func(*args, **kw)
        t_end = time.time()

        print('"{}" took {:.3f} ms to execute\n'.format(my_func.__name__, (t_end - t_start) * 1000))
        return output

    return timed
