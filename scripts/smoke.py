import pyrdiso
import numpy as np
import scipy.sparse as scsp
from scipy import __version__ as scipy_version

print("NUMPY version", np.__version__)
print("SCIPY version", scipy_version)

assert pyrdiso.smoke() is None, "Expected None as returned value"

size = int(5)
indptr = np.array([1, 4, 6, 9, 12, 14], dtype=np.int32) - 1
indices = np.array([1, 2, 4, 1, 2, 3, 4, 5, 1, 3, 4, 2, 5], dtype=np.int32) - 1
data = np.array(
    [1.0, -1.0, -3.0, -2.0, 5.0, 4.0, 6.0, 4.0, -4.0, 2.0, 7.0, 8.0, -5.0],
    dtype=np.float64,
)

assert (
    pyrdiso.check_csr(
        indptr=indptr,
        indices=indices,
        shape=(size, size),
        check=pyrdiso.CHECK_GENERAL_STRUCTURE,
        data=data, # this is optional
        verbose=1, # this is 0 by default
    )
    == 0
)

scipy_object_ = scsp.csr_array((data, indices, indptr), shape=(size, size))
print(scipy_object_.toarray())  # show what the matrix is

factorized_ = pyrdiso.CustomObject(
    data=data,
    indptr=indptr,
    indices=indices,
    shape=(size, size),
    mtype=pyrdiso.MTYPE_REAL_NONSYMMETRIC,
)

assert factorized_.error() == 0

copy_of_ = factorized_.csr_data()

assert np.all(copy_of_["data"] == data)
assert np.all(copy_of_["indptr"] == indptr)
assert np.all(copy_of_["indices"] == indices)

print("[%s] done." % (__name__))
