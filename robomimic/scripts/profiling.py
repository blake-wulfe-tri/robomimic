from contextlib import contextmanager
import cProfile
import time

import tensorflow as tf
import tqdm


@contextmanager
def cprofiler_context(output_file, do_profile=True):
    if do_profile:
        pr = cProfile.Profile()
        pr.enable()
    yield
    if do_profile:
        pr.disable()
        pr.dump_stats(output_file)


@contextmanager
def tfprofiler_context(output_file, do_profile=True):
    if do_profile:
        tf.profiler.experimental.start(output_file)
    yield
    if do_profile:
        tf.profiler.experimental.stop()


def get_profiler_context_manager(profiler):
    if profiler == "cprofile":
        return cprofiler_context
    elif profiler == "tfprofile":
        return tfprofiler_context
    else:
        raise ValueError()


def time_dataloader(
    loader,
    prof_filepath,
    max_iters=50,
    verbose=True,
    profile=False,
    profiler="cprofile",
):
    if verbose:
        loader = tqdm.tqdm(loader, desc="Data Loader:")
    profiler_context_manager = get_profiler_context_manager(profiler)
    start = time.time()
    with profiler_context_manager(prof_filepath, do_profile=profile):
        for i, batch in enumerate(loader):
            if i == max_iters - 1:
                break
    end = time.time()
    duration = end - start
    if verbose:
        print(f"{i+1} batches took {duration:0.4f} seconds")
    return duration
