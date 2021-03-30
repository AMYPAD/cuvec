/**
 * Python template header wrapping `CuVec<T>`. Provides:
 *     struct PyCuVec<T> : PyObject;
 *     PyCuVec<T> *PyCuVec_zeros(std::vector<Py_ssize_t> shape);
 *     PyCuVec<T> *PyCuVec_zeros_like(PyCuVec<T> *other);
 *     PyCuVec<T> *PyCuVec_deepcopy(PyCuVec<T> *other);
 *     PyTypeObject PyCuVec_tp<T>.tp_obj;
 */
#ifndef _PYCUVEC_H_
#define _PYCUVEC_H_

#include "Python.h"
#ifndef _CUVEC_HALF
#ifndef CUVEC_DISABLE_CUDA
#include "cuda_fp16.h" // __half
#define _CUVEC_HALF __half
#else // CUVEC_DISABLE_CUDA
#ifdef __fp16
#define _CUVEC_HALF __fp16
#endif               // __fp16
#endif               // CUVEC_DISABLE_CUDA
#endif               // _CUVEC_HALF
#include "cuvec.cuh" // CuVec
#include <cstdlib>   // malloc, free
#include <sstream>   // std::stringstream
#include <typeinfo>  // typeid
#include <vector>    // std::vector

namespace cuvec {
template <typename T> struct PyType {
  static const char *format() { return typeid(T).name(); }
};
#define _PYCVEC_TPCHR(T, typestr)                                                                 \
  template <> struct PyType<T> {                                                                  \
    static const char *format() { return typestr; }                                               \
  }
_PYCVEC_TPCHR(char, "c");
_PYCVEC_TPCHR(signed char, "b");
_PYCVEC_TPCHR(unsigned char, "B");
#ifdef _Bool
_PYCVEC_TPCHR(_Bool, "?");
#endif
_PYCVEC_TPCHR(short, "h");
_PYCVEC_TPCHR(unsigned short, "H");
_PYCVEC_TPCHR(int, "i");
_PYCVEC_TPCHR(unsigned int, "I");
_PYCVEC_TPCHR(long long, "q");
_PYCVEC_TPCHR(unsigned long long, "Q");
#ifdef _CUVEC_HALF
_PYCVEC_TPCHR(_CUVEC_HALF, "e");
#endif
_PYCVEC_TPCHR(float, "f");
_PYCVEC_TPCHR(double, "d");
} // namespace cuvec

/** classes */
/// class PyCuVec<T>
template <class T> struct PyCuVec {
  PyObject_HEAD CuVec<T> vec; // PyObject_HEAD has an implicit `;` after it
  std::vector<Py_ssize_t> shape;
  std::vector<Py_ssize_t> strides;
};
namespace cuvec {
/// __init__
template <class T> int PyCuVec_init(PyCuVec<T> *self, PyObject *args, PyObject *kwargs) {
  PyObject *shape;
  static const char *kwds[] = {"shape", NULL};
  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O|i", (char **)kwds, &shape)) return -1;
  if (!PySequence_Check(shape)) {
    PyErr_SetString(PyExc_ValueError, "First argument must be shape (sequence)");
    return -1;
  }
  Py_ssize_t ndim = PySequence_Size(shape);
  if (ndim <= 0) return 0;
  self->shape.resize(ndim);
  PyObject *o;
  for (int i = 0; i < ndim; i++) {
    o = PySequence_ITEM(shape, i);
    if (!o) return -1;
    self->shape[i] = PyLong_AsSsize_t(o);
    Py_DECREF(o);
  }
  self->strides.resize(ndim);
  self->strides[ndim - 1] = (Py_ssize_t)sizeof(T);
  for (int i = ndim - 2; i >= 0; i--) self->strides[i] = self->shape[i + 1] * self->strides[i + 1];
  self->vec.resize(self->shape[0] * (self->strides[0] / sizeof(T)));
  return 0;
}
/// __del__
template <class T> void PyCuVec_dealloc(PyCuVec<T> *self) {
  self->vec.clear();
  self->vec.shrink_to_fit();
  self->shape.clear();
  self->shape.shrink_to_fit();
  self->strides.clear();
  self->strides.shrink_to_fit();
  Py_TYPE(self)->tp_free((PyObject *)self);
}
/// __name__
template <class T> const char *PyCuVec_t_str() {
  static char s[] = "PyCuVec_X";
  s[8] = cuvec::PyType<T>::format()[0];
  return s;
}
/// __str__
template <class T> PyObject *PyCuVec_str(PyCuVec<T> *self) {
  std::stringstream s;
  s << PyCuVec_t_str<T>() << "((";
  if (self->shape.size() > 0) s << self->shape[0];
  for (size_t i = 1; i < self->shape.size(); i++) s << ", " << self->shape[i];
  s << "))";
  std::string c = s.str();
  PyObject *ret = PyUnicode_FromString(c.c_str());
  return ret;
}
/// buffer interface
template <class T> int PyCuVec_getbuffer(PyObject *obj, Py_buffer *view, int flags) {
  if (view == NULL) {
    PyErr_SetString(PyExc_BufferError, "NULL view in getbuffer");
    view->obj = NULL;
    return -1;
  }

  PyCuVec<T> *self = (PyCuVec<T> *)obj;
  view->buf = (void *)self->vec.data();
  view->obj = obj;
  view->len = self->vec.size() * sizeof(T);
  view->readonly = 0;
  view->itemsize = sizeof(T);
  view->format = (char *)cuvec::PyType<T>::format();
  view->ndim = self->shape.size();
  view->shape = self->shape.data();
  view->strides = self->strides.data();
  view->suboffsets = NULL;
  view->internal = NULL;

  Py_INCREF(view->obj);
  return 0;
}
template <class T> void PyCuVec_releasebuffer(PyObject *obj, Py_buffer *view) {
  if (view == NULL) {
    PyErr_SetString(PyExc_BufferError, "NULL view in release");
    return;
  }
  // Py_DECREF(obj) is automatic
}
} // namespace cuvec
/// class
template <class T> struct PyCuVec_tp {
  PyBufferProcs as_buffer;
  PyTypeObject tp_obj;
  PyCuVec_tp()
      : as_buffer({
            (getbufferproc)cuvec::PyCuVec_getbuffer<T>,
            (releasebufferproc)cuvec::PyCuVec_releasebuffer<T>,
        }),
        tp_obj({
            PyVarObject_HEAD_INIT(NULL, 0) cuvec::PyCuVec_t_str<T>(), /* tp_name */
            sizeof(PyCuVec<T>),                                       /* tp_basicsize */
            0,                                                        /* tp_itemsize */
            (destructor)cuvec::PyCuVec_dealloc<T>,                    /* tp_dealloc */
            0,                                                        /* tp_print */
            0,                                                        /* tp_getattr */
            0,                                                        /* tp_setattr */
            0,                                                        /* tp_reserved */
            0,                                                        /* tp_repr */
            0,                                                        /* tp_as_number */
            0,                                                        /* tp_as_sequence */
            0,                                                        /* tp_as_mapping */
            0,                                                        /* tp_hash  */
            0,                                                        /* tp_call */
            (reprfunc)cuvec::PyCuVec_str<T>,                          /* tp_str */
            0,                                                        /* tp_getattro */
            0,                                                        /* tp_setattro */
            &as_buffer,                                               /* tp_as_buffer */
            Py_TPFLAGS_DEFAULT,                                       /* tp_flags */
            "Arguments\n---------\nshape  : tuple",                   /* tp_doc */
            0,                                                        /* tp_traverse */
            0,                                                        /* tp_clear */
            0,                                                        /* tp_richcompare */
            0,                                                        /* tp_weaklistoffset */
            0,                                                        /* tp_iter */
            0,                                                        /* tp_iternext */
            0,                                                        /* tp_methods */
            0,                                                        /* tp_members */
            0,                                                        /* tp_getset */
            0,                                                        /* tp_base */
            0,                                                        /* tp_dict */
            0,                                                        /* tp_descr_get */
            0,                                                        /* tp_descr_set */
            0,                                                        /* tp_dictoffset */
            (initproc)cuvec::PyCuVec_init<T>,                         /* tp_init */
        }) {
    tp_obj.tp_new = PyType_GenericNew;
    if (PyType_Ready(&tp_obj) < 0) fprintf(stderr, "error: count not finalise\n");
  }
};

/// Helper functions for creating `PyCuVec<T> *`s in C++ for casting to CPython API `PyObject *`s
template <class T> PyCuVec<T> *PyCuVec_new() {
  static PyCuVec_tp<T> PyCuVec_T;
  if (PyType_Ready(&PyCuVec_T.tp_obj) < 0) return NULL;
  return (PyCuVec<T> *)PyCuVec_T.tp_obj.tp_alloc(&PyCuVec_T.tp_obj, 0);
}
template <class T> PyCuVec<T> *PyCuVec_zeros(std::vector<Py_ssize_t> shape) {
  PyCuVec<T> *self = PyCuVec_new<T>();
  if (!self) return NULL;
  size_t ndim = shape.size();
  self->shape = shape;
  self->strides.resize(ndim);
  self->strides[ndim - 1] = (Py_ssize_t)sizeof(T);
  for (int i = ndim - 2; i >= 0; i--) self->strides[i] = self->shape[i + 1] * self->strides[i + 1];
  self->vec.resize(self->shape[0] * (self->strides[0] / sizeof(T)));
  return self;
}
template <class T> PyCuVec<T> *PyCuVec_zeros_like(PyCuVec<T> *other) {
  PyCuVec<T> *self = PyCuVec_new<T>();
  if (!self) return NULL;
  self->vec.resize(other->vec.size());
  self->shape = other->shape;
  self->strides = other->strides;
  return self;
}
template <class T> PyCuVec<T> *PyCuVec_deepcopy(PyCuVec<T> *other) {
  PyCuVec<T> *self = PyCuVec_new<T>();
  if (!self) return NULL;
  self->vec = other->vec;
  self->shape = other->shape;
  self->strides = other->strides;
  return self;
}

#endif // _PYCUVEC_H_
