"""Release disk-backed array pages between bounded-size processing batches."""
import mmap
import numpy as np


def release_pages(array):
    base = array
    while getattr(base, 'base', None) is not None and not isinstance(base, np.memmap):
        base = base.base
    if isinstance(base, np.memmap):
        if base.flags.writeable:
            base.flush()
        if hasattr(base._mmap, 'madvise'):
            base._mmap.madvise(mmap.MADV_DONTNEED)
