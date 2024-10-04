
from functools import wraps, partial
import time
import concurrent
from collections import OrderedDict
from concurrent.futures import ProcessPoolExecutor
from tqdm.auto import tqdm


def timeit(func, history):
    @wraps(func)
    def timeit_wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        total_time = end_time - start_time
        # first item in the args, ie `args[0]` is `self`
        # print(f'Function {func.__name__}{args} {kwargs} Took {total_time:.4f} seconds')
        history.append(total_time)
        return result
    return timeit_wrapper

# def get_timer(history):
#     return partial(_timeit, history=history)

def launch_jobs(n_workers:int, func, list_param_dict):
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        print('launching jobs')
        print(executor)
        try:
            futures = [executor.submit(func, params) for params in reversed(list_param_dict)]
            counter = len(futures)
            print(f"received {counter} job(s)")
            for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc='jobs'):
                future.result()
                counter -= 1
                print(f'Remaining job(s): {counter}')
        except Exception as e:
            print(e)
            raise e