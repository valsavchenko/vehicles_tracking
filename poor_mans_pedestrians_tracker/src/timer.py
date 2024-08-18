import time


def timer(label='', trace=True):
    def on_decorator(function):
        def on_call(*args, **kargs):
            start = time.perf_counter()
            result = function(*args, **kargs)
            elapsed = time.perf_counter() - start

            on_call.__total_elapsed_time += elapsed
            on_call.__total_calls_num += 1
            if trace:
                logger = args[0].get_logger()
                msec_per_sec = 1000
                elapsed_ms = int(elapsed * msec_per_sec)
                average_ms = int(on_call.__total_elapsed_time / on_call.__total_calls_num * msec_per_sec)
                logger.debug(f'{label}: {elapsed_ms} ms.; {average_ms} ms. per {on_call.__total_calls_num} calls')

            return result

        on_call.__total_elapsed_time = 0
        on_call.__total_calls_num = 0
        return on_call

    return on_decorator
