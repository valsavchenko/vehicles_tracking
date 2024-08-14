import time


def timer(label='', trace=True):
    def on_decorator(function):
        def on_call(*args, **kargs):
            start = time.perf_counter()
            result = function(*args, **kargs)
            elapsed = time.perf_counter() - start

            on_call.__total_elapsed_time += elapsed
            on_call.__total_invocations_count += 1
            if trace:
                milliseconds_per_second = 1000
                elapsed_ms = int(elapsed * milliseconds_per_second)
                average_ms = int(
                    on_call.__total_elapsed_time / on_call.__total_invocations_count * milliseconds_per_second)
                print(f'{label}: {elapsed_ms} ms.; {average_ms} ms. per {on_call.__total_invocations_count} calls')

            return result

        on_call.__total_elapsed_time = 0
        on_call.__total_invocations_count = 0
        return on_call

    return on_decorator
