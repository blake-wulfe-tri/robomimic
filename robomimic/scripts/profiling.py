import cProfile
import pstats
import time

import tqdm

import tensorflow as tf


def time_dataloader(
    loader,
    prof_filepath,
    max_iters=50,
    verbose=True,
    profile=False,
):
    start = time.time()

    if verbose:
        loader = tqdm.tqdm(loader, desc="Data Loader:")

    if profile:
        pr = cProfile.Profile()
        pr.enable()
    for i, batch in enumerate(loader):
        # breakpoint()
        if i == max_iters - 1:
            break

    end = time.time()

    if profile:
        pr.disable()
        pr.dump_stats(prof_filepath)

    duration = end - start
    if verbose:
        print(f"{i+1} batches took {duration:0.4f} seconds")
    return duration


# import tensorflow as tf
# import time
# import tqdm


# @tf.function
# def tf_time_dataloader(loader, max_iters):
#     for i, batch in enumerate(loader.take(max_iters)):
#         # just a pass for now; you can perform operations here if required
#         tf.print(i)
#         breakpoint()


# def time_tf_dataloader(
#     loader,
#     max_iters=50,
#     verbose=True,
# ):
#     start = time.time()
#     tf_time_dataloader(loader, max_iters)
#     end = time.time()
#     duration = end - start
#     if verbose:
#         print(f"{max_iters} batches took {duration:0.4f} seconds")
#     return duration
