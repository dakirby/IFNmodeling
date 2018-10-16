import os
from multiprocessing import cpu_count


# LIBRARY GLOBAL MODS
def set_deep_threads():
    system_threads = cpu_count()
    threads_optimized = {4: "4",    # for laptop
                         16: "16",   # for Santa Fe
                         64: "64",   # for workstation
                         80: "2"}   # for cluster
    threads_int = threads_optimized[system_threads]
    print("init_multiprocessing.py - Setting os.environ['OPENBLAS_NUM_THREADS'] (and others) to {}".format(threads_int))
    os.environ['MKL_NUM_THREADS'] = threads_int
    os.environ["OMP_NUM_THREADS"] = threads_int
    os.environ["NUMEXPR_NUM_THREADS"] = threads_int
    os.environ["OPENBLAS_NUM_THREADS"] = threads_int
    #os.environ["MKL_THREADING_LAYER"] = "sequential"  # this should be off if NUM_THREADS is not 1
    return 0

set_deep_threads()  # this must be set before importing numpy for the first time (during execution)