import abc
import math
import time


def _to_ms(ns):
    """
    """
    _NSS_PER_MS = 1e6
    return int(ns // _NSS_PER_MS)


def timer(label=''):
    def on_decoration(function):
        def on_call(*args, **kargs):
            # Time the function
            start = time.perf_counter_ns()
            result = function(*args, **kargs)
            elapsed_ns = time.perf_counter_ns() - start

            # Update the stats
            on_call.__total_elapsed_ns += elapsed_ns
            on_call.__total_calls_num += 1
            if elapsed_ns < on_call.__min_elapsed_ns:
                on_call.__min_elapsed_ns = elapsed_ns
            if on_call.__max_elapsed_ns < elapsed_ns:
                on_call.__max_elapsed_ns = elapsed_ns

            # Report the stats
            min_ms = _to_ms(ns=on_call.__min_elapsed_ns)
            elapsed_ms = _to_ms(ns=elapsed_ns)
            mean_ms = _to_ms(ns=on_call.__total_elapsed_ns / on_call.__total_calls_num)
            max_ms = _to_ms(ns=on_call.__max_elapsed_ns)

            logger = args[0].get_logger()
            logger.debug(f'Report timing for an operation: {{'
                         f'"tag": "{label}", '
                         f'"min_ms": {min_ms}, '
                         f'"elapsed_ms": {elapsed_ms}, '
                         f'"mean_ms": {mean_ms}, '
                         f'"max_ms": {max_ms}, '
                         f'"num": {on_call.__total_calls_num}'
                         f'}}')

            return result

        on_call.__min_elapsed_ns = math.inf
        on_call.__total_elapsed_ns = 0
        on_call.__total_calls_num = 0
        on_call.__max_elapsed_ns = -math.inf
        return on_call

    return on_decoration


class Timeable(metaclass=abc.ABCMeta):
    """
    Provides a common interface to time operations
    """

    def __init__(self, logger):
        self.__logger = logger

    def get_logger(self):
        return self.__logger
