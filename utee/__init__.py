from functools import wraps, partial
import time
import concurrent
import datetime
import uuid
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
        history.append(total_time)
        return result

    return timeit_wrapper


def get_time_uuid():
    now = datetime.datetime.now()
    time_str = now.strftime("%H%M%S")
    random_uuid = str(uuid.uuid4()).split("-")[1]
    return f"{time_str}{random_uuid}"


def launch_jobs(n_workers: int, func, list_param_dict):
    """
    Launches multiple parallel jobs using a ProcessPoolExecutor.

    Args:
        n_workers (int): The maximum number of worker processes to use for parallel execution.
        func (callable): The function to be executed for each job. This function should accept a single argument, which will be a dictionary from `list_param_dict`.
        list_param_dict (list[dict]): A list of dictionaries, where each dictionary contains the parameters to be passed to the `func` for a single job. The jobs are submitted in reverse order of this list.

    Raises:
        Exception: If any exception occurs during the execution of the jobs, it is caught, printed, and then re-raised.
    """
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        print("launching jobs")
        print(executor)
        try:
            futures = [
                executor.submit(func, params) for params in reversed(list_param_dict)
            ]
            counter = len(futures)
            print(f"received {counter} job(s)")
            for future in tqdm(
                concurrent.futures.as_completed(futures),
                total=len(futures),
                desc="jobs",
            ):
                future.result()
                counter -= 1
                print(f"Remaining job(s): {counter}")
        except Exception as e:
            print(e)
            raise e
