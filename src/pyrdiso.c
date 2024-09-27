#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include "numpy/arrayobject.h"
#include <stdbool.h>

//
// https://docs.python.org/3/extending/index.html
// https://docs.python.org/3/extending/newtypes_tutorial.html
// https://www.intel.com/content/www/us/en/docs/onemkl/developer-reference-c/2023-0/pardiso-getdiag.html
//
// TODO: optional access to the factorized diagonal extraction MKL program
//       allow get_diag if iparm[55] = 1 on init, optional flag
//

//#include <stdio.h>
//#include <stdlib.h>
//#include <math.h>

#include "mkl_pardiso.h"
#include "mkl_types.h"
#include "mkl_sparse_handle.h"

// in utilities.c
int validate_csr_data(MKL_INT n, 
                      MKL_INT* ia, 
                      MKL_INT* ja, 
                      int one_based,
                      int mkl_structure, 
                      int verbose);

///////////////////////////////////////////////////////////////////////////////

/*** MODULE ERRORS OBJECT ***/

static PyObject *ModuleError;

#define ERRORMESSAGE(msg) \
  { PyErr_SetString(ModuleError, msg); \
    goto offload_and_return_error; } \

/*** INPUT ARGUMENT HELPERS ***/

bool np_is_nonempty_vector(PyArrayObject* a)
{
  if (!PyArray_ISBEHAVED_RO(a)) return false; // ?
  if (PyArray_NDIM(a) != 1) return false;
  const int nelem = PyArray_DIM(a, 0);
  return (nelem > 0);
}

bool np_is_nonempty_vector_of_type(PyArrayObject* a, int type)
{
  if (!PyArray_ISBEHAVED_RO(a)) return false; // ?
  if (PyArray_NDIM(a) != 1) return false;
  if (PyArray_TYPE(a) != type) return false;
  const int nelem = PyArray_DIM(a, 0);
  return (nelem > 0);
}

///////////////////////////////////////////////////////////////////////////////

void reset_pt_and_iparm_(void** pt, 
                         MKL_INT* iparm, 
                         int len) // len should be set to 64
{
  memset(pt, 0, len * sizeof(void*));
  memset(iparm, 0, len * sizeof(MKL_INT));

  iparm[0] = 1;         /* No solver default */
  iparm[1] = 2;         /* Fill-in reordering from METIS */
  iparm[3] = 0;         /* No iterative-direct algorithm */
  iparm[4] = 0;         /* No user fill-in reducing permutation */
  iparm[5] = 0;         /* Write solution into x */
  iparm[6] = 0;         /* Not in use */
  iparm[7] = 2;         /* Max numbers of iterative refinement steps */
  iparm[8] = 0;         /* Not in use */
  iparm[9] = 13;        /* Perturb the pivot elements with 1E-13 */
  iparm[10] = 1;        /* Use nonsymmetric permutation and scaling MPS */
  iparm[11] = 0;        /* Conjugate transposed/transpose solve */
  iparm[12] = 1;        /* Maximum weighted matching algorithm is switched-on (default for non-symmetric) */
  iparm[13] = 0;        /* Output: Number of perturbed pivots */
  iparm[14] = 0;        /* Not in use */
  iparm[15] = 0;        /* Not in use */
  iparm[16] = 0;        /* Not in use */
  iparm[17] = -1;       /* Output: Number of nonzeros in the factor LU */
  iparm[18] = -1;       /* Output: Mflops for LU factorization */
  iparm[19] = 0;        /* Output: Numbers of CG Iterations */
  iparm[34] = 1;        /* zero-based indexing */
}

typedef struct {
    PyObject_HEAD

    /* Type-specific fields go here. */

    int status;

    /* PARDISO Internal solver memory pointer pt, */
    /* 32-bit: int pt[64]; 64-bit: long int pt[64] */
    /* or void *pt[64] should be OK on both architectures */
    void* pt[64];

    /* PARDISO control parameters; associated with a particular setup call */
    MKL_INT iparm[64];
    MKL_INT maxfct, mnum, phase, error;
    MKL_INT mtype, nrhs, msglvl;
    MKL_INT n; 

    PyArrayObject* arr_data;      // a
    PyArrayObject* arr_indptr;    // ia
    PyArrayObject* arr_indices;   // ja

} CustomObject;

static int
CustomObject_init(CustomObject *self, PyObject *args, PyObject *kwds)
{
  self->status = -1;
  self->arr_data = NULL;
  self->arr_indptr = NULL;
  self->arr_indices = NULL;

  // printf("[%s::%s]\n", __FILE__, __func__);

  static char* kwlist[] = {
    "data", 
    "indptr", 
    "indices", 
    "shape", 
    "mtype", 
    "msglvl", 
    NULL
  };

  PyObject *arg_data = NULL, *arg_indptr = NULL, *arg_indices = NULL, 
           *arg_shape = NULL, *arg_mtype = NULL;

  int arg_msglvl = 0;

  if (!PyArg_ParseTupleAndKeywords(args, kwds, "O!O!O!O!O!|i", kwlist, 
                                   &PyArray_Type, &arg_data, 
                                   &PyArray_Type, &arg_indptr, 
                                   &PyArray_Type, &arg_indices,
                                   &PyTuple_Type, &arg_shape,
                                   &PyLong_Type, &arg_mtype,
                                   &arg_msglvl)) 
  {
    return 1;
  }

  if (PyTuple_GET_SIZE(arg_shape) != 2) {
    PyErr_SetString(ModuleError, "shape tuple input must have 2 elements");
    return 1;
  }

  const int shape_m = PyLong_AsLong(PyTuple_GET_ITEM(arg_shape, 0));
  const int shape_n = PyLong_AsLong(PyTuple_GET_ITEM(arg_shape, 1));

  // printf("[%s::%s::shape=(%i,%i)]\n", __FILE__, __func__, shape_m, shape_n);

  if (shape_m != shape_n) {
    PyErr_SetString(ModuleError, "shape tuple input must indicate a square matrix");
    return 1;
  }

  self->n = (MKL_INT) shape_m;
  self->mtype = (MKL_INT) PyLong_AsLong(arg_mtype);
  self->msglvl = (MKL_INT) arg_msglvl;

  self->nrhs = 1;             /* number of RHS to solve for (memory allocation presumably) */
  self->maxfct = 1;           /* Maximum number of numerical factorizations. */
  self->mnum = 1;             /* Which factorization to use (1..maxfct) */

  const bool symmetric_factorization = (self->mtype == -2);
  const bool symmetric_positive_factorization = (self->mtype == 2);
  const bool nonsymmetric_factorization = (self->mtype == 11);

  if (!symmetric_factorization && !nonsymmetric_factorization && !symmetric_positive_factorization) {
    PyErr_SetString(ModuleError, "Only supports: mtype=-2 (real symmetric indefinite), mtype=11 (real nonsymmetric), or mtype=2 (real symmetric positive)");
    return 1;
  }

  if (self->msglvl > 0) {
    printf("[%s::%s::mtype=%d]\n", __FILE__, __func__, self->mtype);
    printf("[%s::%s::msglvl=%d]\n", __FILE__, __func__, self->msglvl);
  }

  if (!np_is_nonempty_vector_of_type((PyArrayObject *) arg_data, NPY_DOUBLE) || 
      !np_is_nonempty_vector_of_type((PyArrayObject *) arg_indptr, NPY_INT32) || 
      !np_is_nonempty_vector_of_type((PyArrayObject *) arg_indices, NPY_INT32))
  {
    PyErr_SetString(ModuleError, "CSR data type (or shape) is incorrect for at least one of three arrays");
    return 1;
  }

  if (sizeof(MKL_INT) != 4) {
    PyErr_SetString(ModuleError, "Expected MKL_INT to be 32 bit.");
    return 1;
  }

  const int OBJECT_ARRAY_FLAGS = NPY_ARRAY_CARRAY | NPY_ARRAY_ENSURECOPY; // aligned, writeable, contiguous

  self->arr_data = PyArray_FROM_OTF(arg_data, NPY_DOUBLE, OBJECT_ARRAY_FLAGS);
  self->arr_indptr = PyArray_FROM_OTF(arg_indptr, NPY_INT32, OBJECT_ARRAY_FLAGS);
  self->arr_indices = PyArray_FROM_OTF(arg_indices, NPY_INT32, OBJECT_ARRAY_FLAGS);

  if (self->arr_data == NULL || self->arr_indptr == NULL || self->arr_indices == NULL) {
    Py_XDECREF(self->arr_data);
    Py_XDECREF(self->arr_indptr);
    Py_XDECREF(self->arr_indices);
    PyErr_SetString(ModuleError, "PyArray_FROM_OTF failed for at least one of the input arrays");
    return 1;
  }

  const int num_data = PyArray_SIZE((PyArrayObject *) self->arr_data);
  double* ptr_data = PyArray_DATA((PyArrayObject *) self->arr_data);          // a
  MKL_INT* ptr_indptr = PyArray_DATA((PyArrayObject *) self->arr_indptr);     // ia
  MKL_INT* ptr_indices = PyArray_DATA((PyArrayObject *) self->arr_indices);   // ja

  if (self->msglvl > 0)
    printf("[%s::%s::len(data)=%i]\n", __FILE__, __func__, num_data);

  // Input appears to be of correct type... pass it along to PARDISO...
  // Do not forget to set 0-based indexing (which will be the case for SCIPY CSRs)

  // memset(self->pt, 0, 64 * sizeof(void*));
  // memset(self->iparm, 0, 64 * sizeof(MKL_INT));

  reset_pt_and_iparm_(self->pt, self->iparm, 64);

  self->status = 0;
  self->error = 0;            /* Initialize error flag */
  self->phase = 0;

  double ddum;          /* Double dummy */
  MKL_INT idum;         /* Integer dummy. */

  // Failure during calls to PARDISO below will still result in object initialization
  // but there will be a nonzero error code; fetched as self->error, and the 
  // object cannot be used to solve equations.
  // self->phase also indicates at which step the error happened

  if (nonsymmetric_factorization)
  {
    self->phase = 11;

    PARDISO(self->pt, &self->maxfct, &self->mnum, &self->mtype, &self->phase,
            &self->n, ptr_data, ptr_indptr, ptr_indices, &idum, &self->nrhs, 
            self->iparm, &self->msglvl, &ddum, &ddum, &self->error);

    if (self->error != 0) {
      return self->status;
    }

    self->phase = 22;

    PARDISO(self->pt, &self->maxfct, &self->mnum, &self->mtype, &self->phase,
            &self->n, ptr_data, ptr_indptr, ptr_indices, &idum, &self->nrhs, 
            self->iparm, &self->msglvl, &ddum, &ddum, &self->error);

    if (self->error != 0) {
      return self->status;
    }

  }

  if (symmetric_factorization || symmetric_positive_factorization) 
  {
    self->phase = 11;

    PARDISO(self->pt, &self->maxfct, &self->mnum, &self->mtype, &self->phase,
            &self->n, ptr_data, ptr_indptr, ptr_indices, &idum, &self->nrhs, 
            self->iparm, &self->msglvl, &ddum, &ddum, &self->error);
    
    if (self->error != 0) {
      return self->status;
    }

    self->phase = 22;
    
    PARDISO(self->pt, &self->maxfct, &self->mnum, &self->mtype, &self->phase,
            &self->n, ptr_data, ptr_indptr, ptr_indices, &idum, &self->nrhs, 
            self->iparm, &self->msglvl, &ddum, &ddum, &self->error);

    if (self->error != 0) {
      return self->status;
    }

  }

  return self->status;

//offload_and_return_error:
//  return 1;
}

static void
CustomObject_dealloc(CustomObject *self)
{
  // printf("[%s::%s]\n", __FILE__, __func__);

  if (self->status == 0) {
    double ddum;          /* Double dummy */
    MKL_INT idum;         /* Integer dummy. */
    MKL_INT* ptr_indptr = PyArray_DATA((PyArrayObject *) self->arr_indptr);     // ia
    MKL_INT* ptr_indices = PyArray_DATA((PyArrayObject *) self->arr_indices);   // ja

    self->phase = -1;           /* Release internal memory. */

    PARDISO(self->pt, &self->maxfct, &self->mnum, &self->mtype, &self->phase,
            &self->n, &ddum, ptr_indptr, ptr_indices, &idum, &self->nrhs,
            self->iparm, &self->msglvl, &ddum, &ddum, &self->error);
  }

  if (self->status == 0) {
    Py_XDECREF(self->arr_data);
    Py_XDECREF(self->arr_indptr);
    Py_XDECREF(self->arr_indices);
  }

  Py_TYPE(self)->tp_free((PyObject *)self);
}


static PyObject*
CustomObject_solve(CustomObject *self, 
                   PyObject *args,
                   PyObject *kwds)
{
  static char* kwlist[] = {
    "rhs", 
    "transpose",    // optional (has default, if applicable)
    "refinements",  // optional (has default)
    NULL
  };

  PyObject *arg_rhs = NULL;
  int transpose_code = 0;
  int max_refinements = self->iparm[7];

  if (!PyArg_ParseTupleAndKeywords(args, kwds, "O!|ii", kwlist, 
                                   &PyArray_Type, &arg_rhs, 
                                   &transpose_code, 
                                   &max_refinements)) 
  {
    return NULL;
  }

  if (self->error != 0) {
    PyErr_SetString(ModuleError, "PARDISO factorization object has non-zero error status.");
    return NULL;
  }

  const bool symmetric_factorization = (self->mtype == -2);
  const bool symmetric_positive_factorization = (self->mtype == 2);
  const bool nonsymmetric_factorization = (self->mtype == 11);

  PyArrayObject* arr_rhs = PyArray_FROM_OTF(arg_rhs, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);

  if (arr_rhs == NULL) {
    PyErr_SetString(ModuleError, "Failed to obtain array object for RHS iput argument");
    return NULL;
  }

  const int num_data = PyArray_SIZE((PyArrayObject *) arr_rhs);
  double* ptr_rhs = PyArray_DATA((PyArrayObject *) arr_rhs);

  if (num_data != self->n) {
    Py_DECREF(arr_rhs);
    PyErr_SetString(ModuleError, "RHS input array has incorrect number of elements");
    return NULL;
  }

  const npy_intp output_array_dims[1] = {num_data};
  PyArrayObject *arr_solution = PyArray_SimpleNew(1, output_array_dims, NPY_DOUBLE);

  if (arr_solution == NULL) {
    Py_DECREF(arr_rhs);
    PyErr_SetString(ModuleError, "Failed to create array object for solution");
    return NULL;  
  }

  double* ptr_solution = PyArray_DATA((PyArrayObject *) arr_solution);

  double* ptr_data = PyArray_DATA((PyArrayObject *) self->arr_data);          // a
  MKL_INT* ptr_indptr = PyArray_DATA((PyArrayObject *) self->arr_indptr);     // ia
  MKL_INT* ptr_indices = PyArray_DATA((PyArrayObject *) self->arr_indices);   // ja

  self->iparm[7] = max_refinements;
  self->phase = 33;

  MKL_INT idum;

  if (symmetric_factorization || symmetric_positive_factorization)
  {
    PARDISO(self->pt, &self->maxfct, &self->mnum, &self->mtype, &self->phase,
            &self->n, ptr_data, ptr_indptr, ptr_indices, &idum, &self->nrhs, 
            self->iparm, &self->msglvl, ptr_rhs, ptr_solution, &self->error);

    if (self->error != 0) {
      Py_DECREF(arr_rhs);
      Py_DECREF(arr_solution);
      PyErr_SetString(ModuleError, "Solution phase call failure (symmetric matrix)");
      return NULL;
    }
  }

  if (nonsymmetric_factorization)
  {
    self->iparm[11] = transpose_code;

    PARDISO(self->pt, &self->maxfct, &self->mnum, &self->mtype, &self->phase,
            &self->n, ptr_data, ptr_indptr, ptr_indices, &idum, &self->nrhs, 
            self->iparm, &self->msglvl, ptr_rhs, ptr_solution, &self->error);

    if (self->error != 0) {
      Py_DECREF(arr_rhs);
      Py_DECREF(arr_solution);
      PyErr_SetString(ModuleError, "Solution phase call failure (nonsymmetric matrix)");
      return NULL;
    }
  }

  Py_DECREF(arr_rhs);

  //Py_RETURN_NONE;
  return (PyObject *) arr_solution;
}


static PyObject*
CustomObject_csr_data(CustomObject *self)
{
  PyObject* new_data = PyArray_NewLikeArray(self->arr_data, NPY_KEEPORDER, NULL, 1);
  PyObject* new_indptr = PyArray_NewLikeArray(self->arr_indptr, NPY_KEEPORDER, NULL, 1);
  PyObject* new_indices = PyArray_NewLikeArray(self->arr_indices, NPY_KEEPORDER, NULL, 1);

  if (new_data == NULL || new_indptr == NULL || new_indices == NULL) {
    Py_XDECREF(new_data);
    Py_XDECREF(new_indptr);
    Py_XDECREF(new_indices);  
    return NULL;
  }

  PyArray_CopyInto(new_data, self->arr_data);
  PyArray_CopyInto(new_indptr, self->arr_indptr);
  PyArray_CopyInto(new_indices, self->arr_indices);

  PyObject* outDict = Py_BuildValue("{s:O,s:O,s:O}",
                                    "data", new_data, 
                                    "indptr", new_indptr,
                                    "indices", new_indices);

  Py_DECREF(new_data);
  Py_DECREF(new_indptr);
  Py_DECREF(new_indices);

  return outDict;
}

static PyObject*
CustomObject_error(CustomObject *self)
{
  return PyLong_FromLong(self->error);
}

static PyObject*
CustomObject_phase(CustomObject *self)
{
  return PyLong_FromLong(self->phase);
}

static PyObject*
CustomObject_size(CustomObject *self)
{
  return PyLong_FromLong(self->n);
}

static PyObject*
CustomObject_status(CustomObject *self)
{
  return PyLong_FromLong(self->status);
}

static PyObject*
CustomObject_mtype(CustomObject *self)
{
  return PyLong_FromLong(self->mtype);
}

static PyObject*
CustomObject_msglvl(CustomObject *self, PyObject *args)
{
  int msglvl_ = (int) self->msglvl;
  if (!PyArg_ParseTuple(args, "|i", &msglvl_))
    return NULL;

  if (msglvl_ != (int) self->msglvl) {
    self->msglvl = (MKL_INT) msglvl_;
  }

  return PyLong_FromLong(self->msglvl);
}

static PyMethodDef CustomObjectMethods[] = {
    {"solve", (PyCFunction)CustomObject_solve, METH_VARARGS | METH_KEYWORDS, "PARDISO solve call given a RHS vector"},
    {"error", (PyCFunction)CustomObject_error, METH_NOARGS, "PARDISO error code at last phase"},
    {"phase", (PyCFunction)CustomObject_phase, METH_NOARGS, "PARDISO phase for last call"},
    {"size", (PyCFunction)CustomObject_size, METH_NOARGS, "Return matrix size n"},
    {"status", (PyCFunction)CustomObject_status, METH_NOARGS, "Status code for object"},
    {"mtype", (PyCFunction)CustomObject_mtype, METH_NOARGS, "Return matrix type code"},
    {"msglvl", (PyCFunction)CustomObject_msglvl, METH_VARARGS, "Return PARISO verbosity level (optionally set msglvl)"},
    {"csr_data", (PyCFunction)CustomObject_csr_data, METH_NOARGS, "Return copy of the CSR data (as items in a dict)"},
    {NULL, NULL, 0, NULL}  /* Sentinel */
};

static PyTypeObject CustomType = {
    .ob_base = PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name = "pyrdiso.CustomObject",
    .tp_doc = PyDoc_STR("Python object that stores (and uses) a PARDISO factorization of a sparse matrix (for solving specific right-hand-sides)."),
    .tp_basicsize = sizeof(CustomObject),
    .tp_itemsize = 0,
    .tp_flags = Py_TPFLAGS_DEFAULT,
    .tp_new = PyType_GenericNew,
    .tp_methods = CustomObjectMethods,
    .tp_init = (initproc) CustomObject_init,
    .tp_dealloc = (destructor) CustomObject_dealloc,
};

/*** EXTERNAL-FACING FUNCTIONS ***/

static PyObject*
pyrdiso_smoke(PyObject *self)
{
  printf("[%s] begin.\n", __func__);
  printf("sizeof(MKL_INT)=%li\n", sizeof(MKL_INT));
  printf("[%s] done.\n", __func__);
  Py_RETURN_NONE;
}


static PyObject*
pyrdiso_check_csr(PyObject *self, 
                  PyObject *args,
                  PyObject *kwds)
{
  static char* kwlist[] = {"indptr", "indices", "shape", "data", "check", "verbose", NULL};
  PyObject *arg_data = NULL, *arg_indptr = NULL, *arg_indices = NULL, *arg_shape = NULL;
  int which_structure = -1;
  int verbose = 0;

  if (!PyArg_ParseTupleAndKeywords(args, kwds, "O!O!O!|O!ii", 
                                   kwlist, 
                                   &PyArray_Type, &arg_indptr, 
                                   &PyArray_Type, &arg_indices, 
                                   &PyTuple_Type, &arg_shape,
                                   &PyArray_Type, &arg_data,
                                   &which_structure,
                                   &verbose)) {
    return NULL;
  }

  if (PyTuple_GET_SIZE(arg_shape) != 2) {
    PyErr_SetString(ModuleError, "shape tuple input must have 2 elements");
    return NULL;
  }

  const int shape_m = PyLong_AsLong(PyTuple_GET_ITEM(arg_shape, 0));
  const int shape_n = PyLong_AsLong(PyTuple_GET_ITEM(arg_shape, 1));

  if (shape_m != shape_n) {
    PyErr_SetString(ModuleError, "square matrix shape expected");
    return NULL;
  }

  int mkl_structure_ = MKL_GENERAL_STRUCTURE;
  switch (which_structure) {
    case 0: mkl_structure_ = MKL_GENERAL_STRUCTURE; break;
    case 1: mkl_structure_ = MKL_UPPER_TRIANGULAR; break;
    case 2: mkl_structure_ = MKL_LOWER_TRIANGULAR; break;
    case 3: mkl_structure_ = MKL_STRUCTURAL_SYMMETRIC; break;
    default: 
      PyErr_SetString(ModuleError, "check structure code not recognized");
      return NULL; 
  }

  // printf("which_structure=%i\n", which_structure);

  PyArrayObject* arr_data = NULL;
  if (arg_data != NULL)
    arr_data = PyArray_FROM_OTF(arg_data, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);

  PyArrayObject* arr_indptr = PyArray_FROM_OTF(arg_indptr, NPY_INT32, NPY_ARRAY_IN_ARRAY);
  PyArrayObject* arr_indices = PyArray_FROM_OTF(arg_indices, NPY_INT32, NPY_ARRAY_IN_ARRAY);

  int num_data = -1;
  double* ptr_data = NULL;

  if (arr_data != NULL) {
    num_data = PyArray_SIZE((PyArrayObject *) arr_data);
    ptr_data = PyArray_DATA((PyArrayObject *) arr_data);          // a

    // TODO: if data is given verify it has the correct length, and there are no NaNs

  }

  MKL_INT* ptr_indptr = PyArray_DATA((PyArrayObject *) arr_indptr);     // ia
  MKL_INT* ptr_indices = PyArray_DATA((PyArrayObject *) arr_indices);   // ja

  const int one_based_ = 0;

  int error_ = validate_csr_data(shape_m, 
                                 ptr_indptr, 
                                 ptr_indices, 
                                 one_based_, 
                                 mkl_structure_, 
                                 verbose);

  Py_XDECREF(arr_data);
  Py_XDECREF(arr_indptr);
  Py_XDECREF(arr_indices);

  return PyLong_FromLong(error_);
}

/*** METHODS TABLE ***/

static PyMethodDef pyrdiso_methods[] = 
{
    {"smoke", 
     (PyCFunction) pyrdiso_smoke, 
     METH_NOARGS,
     PyDoc_STR("Smoke test function / example PARDISO call")},

    {"check_csr", 
     (PyCFunction) pyrdiso_check_csr, 
     METH_VARARGS | METH_KEYWORDS,
     PyDoc_STR("Test function for parsing a CSR data structure")},

    {NULL, 
     NULL, 
     0, 
     NULL}  /* end-of-table marker */
};

/*** MODULE INITIALIZATION ***/

PyDoc_STRVAR(docstring, "Basic Python interface for MKL PARIDSO sparse linear solver (WIP).");

static struct PyModuleDef pyrdiso_module = {
    PyModuleDef_HEAD_INIT,
    "pyrdiso",   /* name of module */
    docstring,  /* module documentation, may be NULL */
    -1,         /* size of per-interpreter state of the module,
                   or -1 if the module keeps state in global variables. */
    pyrdiso_methods,
    NULL,
    NULL,
    NULL,
    NULL
};

PyMODINIT_FUNC
PyInit_pyrdiso(void)
{
  if (PyType_Ready(&CustomType) < 0)
    return NULL;

  PyObject* m = PyModule_Create(&pyrdiso_module);

  if (m == NULL)
    return m;

  Py_INCREF(&CustomType);
  if (PyModule_AddObject(m, "CustomObject", (PyObject *) &CustomType) < 0) {
    Py_DECREF(&CustomType);
    Py_DECREF(m);
    return NULL;
  }

  ModuleError = PyErr_NewException("pyrdiso.error", NULL, NULL);
  Py_XINCREF(ModuleError);

  if (PyModule_AddObject(m, "error", ModuleError) < 0) {
    Py_XDECREF(ModuleError);
    Py_CLEAR(ModuleError);
    Py_DECREF(m);
    return NULL;
  }

  import_array();

  return m;
}
