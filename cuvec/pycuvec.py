"""Thin wrappers around `cuvec` C++/CUDA module"""
import array
import logging
from collections.abc import Sequence
from textwrap import dedent

import numpy as np

from .cuvec import (
    Vector_b,
    Vector_B,
    Vector_c,
    Vector_d,
    Vector_e,
    Vector_f,
    Vector_h,
    Vector_H,
    Vector_i,
    Vector_I,
    Vector_q,
    Vector_Q,
)

log = logging.getLogger(__name__)
# u: non-standard np.dype('S2'); l/L: inconsistent between `array` and `numpy`
typecodes = ''.join(i for i in array.typecodes if i not in "ulL") + "e"
vec_types = {
    np.dtype('int8'): Vector_b,
    np.dtype('uint8'): Vector_B,
    np.dtype('S1'): Vector_c,
    np.dtype('int16'): Vector_h,
    np.dtype('uint16'): Vector_H,
    np.dtype('int32'): Vector_i,
    np.dtype('uint32'): Vector_I,
    np.dtype('int64'): Vector_q,
    np.dtype('uint64'): Vector_Q,
    np.dtype('float16'): Vector_e,
    np.dtype('float32'): Vector_f,
    np.dtype('float64'): Vector_d}


def cu_zeros(shape, dtype="float32"):
    """
    Returns a new `<cuvec.Vector_*>` of the specified shape and data type.
    """
    return vec_types[np.dtype(dtype)](shape if isinstance(shape, Sequence) else (shape,))


def cu_copy(arr):
    """
    Returns a new `<cuvec.Vector_*>` with data copied from the specified `arr`.
    """
    res = cu_zeros(arr.shape, arr.dtype)
    np.asarray(res).flat = arr.flat
    return res


_Vector_types = tuple(vec_types.values())
_Vector_types_s = tuple(map(str, vec_types.values()))


def is_raw_cuvec(cuvec):
    """
    Returns `True` when given the output of
    CPython API functions returning `PyCuVec<T> *` PyObjects.

    This is needed since conversely `isinstance(cuvec, CuVec)` may be `False`
    due to external libraries
    `#include "pycuvec.cuh"` making a distinct type object.
    """
    return isinstance(cuvec, _Vector_types) or str(type(cuvec)) in _Vector_types_s


class CuVec(np.ndarray):
    """
    A `numpy.ndarray` compatible view with a `cuvec` member containing the
    underlying `cuvec.Vector_*` object (for use in CPython API function calls).
    """
    def __new__(cls, arr):
        """arr: `cuvec.CuVec`, raw `cuvec.Vector_*`, or `numpy.ndarray`"""
        if is_raw_cuvec(arr):
            log.debug("wrap raw %s", type(arr))
            obj = np.asarray(arr).view(cls)
            obj.cuvec = arr
            return obj
        if isinstance(arr, CuVec) and hasattr(arr, 'cuvec'):
            log.debug("new view")
            obj = np.asarray(arr).view(cls)
            obj.cuvec = arr.cuvec
            return obj
        if isinstance(arr, np.ndarray):
            log.debug("copy")
            return copy(arr)
        raise NotImplementedError(
            dedent("""\
            Not intended for explicit construction
            (do not do `cuvec.CuVec((42, 1337))`;
            instead use `cuvec.zeros((42, 137))`"""))

    @property
    def __cuda_array_interface__(self):
        if not hasattr(self, 'cuvec'):
            raise AttributeError(
                dedent("""\
                `numpy.ndarray` object has no attribute `cuvec`:
                try using `cuvec.asarray()` first."""))
        res = self.__array_interface__
        return {
            'shape': res['shape'], 'typestr': res['typestr'], 'data': res['data'], 'version': 3}


def zeros(shape, dtype="float32"):
    """
    Returns a `cuvec.CuVec` view of a new `numpy.ndarray`
    of the specified shape and data type (`cuvec` equivalent of `numpy.zeros`).
    """
    return CuVec(cu_zeros(shape, dtype))


def copy(arr):
    """
    Returns a `cuvec.CuVec` view of a new `numpy.ndarray`
    with data copied from the specified `arr`
    (`cuvec` equivalent of `numpy.copy`).
    """
    return CuVec(cu_copy(arr))


def asarray(arr, dtype=None, order=None):
    """
    Returns a `cuvec.CuVec` view of `arr`, avoiding memory copies if possible.
    (`cuvec` equivalent of `numpy.asarray`).
    """
    if not isinstance(arr, np.ndarray) and is_raw_cuvec(arr):
        res = CuVec(arr)
        if dtype is None or res.dtype == np.dtype(dtype):
            return CuVec(np.asanyarray(res, order=order))
    return CuVec(np.asanyarray(arr, dtype=dtype, order=order))
