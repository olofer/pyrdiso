import numpy as np
import scipy.sparse as scsp
import pyrdiso
import argparse


def make_random_coo(m: int, rho: float):
    nnz_ = int(m * m * rho)
    data_ = np.random.randn(nnz_)
    rows_ = np.random.randint(low=0, high=m, size=nnz_, dtype=np.int32)
    cols_ = np.random.randint(low=0, high=m, size=nnz_, dtype=np.int32)
    return scsp.coo_array((data_, (rows_, cols_)), shape=(m, m))


def trip_on_error(obj: pyrdiso.CustomObject):
    if obj_.error() != 0:
        print("ERROR %i: %s" % (obj.error(), pyrdiso.error_message_str(obj.error())))
        assert False


def make_upper_csr_proper(M: scsp.base) -> scsp.csr:
    coo_ = scsp.triu(M, k=1, format="coo")
    diag_ = M.diagonal()
    n = M.shape[0]
    assert M.shape[1] == n, "square matrix required"
    vals = np.concatenate([coo_.data, diag_])
    rows = np.concatenate([coo_.row, np.arange(n)], dtype=np.int32)
    cols = np.concatenate([coo_.col, np.arange(n)], dtype=np.int32)
    return scsp.csr_array((vals, (rows, cols)), shape=(n, n))


def make_dominant_csr(
    M: scsp.base, margin: float = 1.0, positive: bool = False
) -> scsp.csr:
    M_abs = M * M.sign()
    row_abs_sum = M_abs.sum(axis=1)
    diag_ = M.diagonal()
    diag_sign_ = np.sign(diag_)
    return (
        M + scsp.diags(-diag_ + (row_abs_sum + margin))
        if positive
        else M + scsp.diags(-diag_ + (row_abs_sum + margin) * diag_sign_)
    )


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--m", type=int, default=101, help="random sparse matrix size")
    parser.add_argument(
        "--rho",
        type=float,
        default=0.15,
        help="approximate random sparse matrix density (nnz/m/m)",
    )
    parser.add_argument(
        "--diag", type=float, default=None, help="add constant scalar value to diagonal"
    )
    parser.add_argument(
        "--random-diag",
        type=float,
        default=None,
        help="add (scaled) random values to diagonal",
    )
    parser.add_argument("--symmetric", action="store_true")
    parser.add_argument("--positive", action="store_true")
    parser.add_argument(
        "--dominance",
        type=float,
        default=None,
        help="assign a specific diagonal dominance (random sign, unless --positive)",
    )
    args = parser.parse_args()

    if args.symmetric:
        coo_ = make_random_coo(args.m, args.rho / 2.0)
        csr_ = coo_.tocsr()
        csr_ = (csr_ + csr_.T) / 2.0
    else:
        coo_ = make_random_coo(args.m, args.rho)
        csr_ = coo_.tocsr()

    if not args.diag is None:
        csr_ = csr_ + scsp.eye(args.m) * args.diag

    if not args.random_diag is None:
        csr_ = csr_ + args.random_diag * scsp.diags(np.random.randn(args.m))

    if not args.dominance is None:
        csr_ = make_dominant_csr(csr_, args.dominance, args.positive)

    if args.dominance is None and args.positive:
        csr_ = make_dominant_csr(csr_, 1.0, True)

    if not csr_.has_sorted_indices:
        csr_ = csr_.sorted_indices()

    assert csr_.has_sorted_indices, "CSR object does not have sorted indices"
    assert csr_.indptr.shape[0] == csr_.shape[0] + 1
    assert csr_.data.shape[0] == csr_.indices.shape[0]

    print("matrix density = %.3f" % (csr_.nnz / np.prod(csr_.shape)))

    assert (
        pyrdiso.check_csr(
            indptr=csr_.indptr,
            indices=csr_.indices,
            shape=csr_.shape,
            check=(
                pyrdiso.CHECK_STRUCTURAL_SYMMETRIC
                if args.symmetric
                else pyrdiso.CHECK_GENERAL_STRUCTURE
            ),
            verbose=1,
        )
        == 0
    )

    rhs_ = np.random.randn(args.m)

    # Built-in SCIPY.SPARSE SuperLU solution
    splu_solver = scsp.linalg.splu(csr_.tocsc())
    x_splu = splu_solver.solve(rhs_)

    err_splu = csr_ @ x_splu - rhs_
    print("max residual element: %e (splu)" % (np.max(np.abs(err_splu))))

    splu_solver_transposed = scsp.linalg.splu(csr_.T.tocsc())
    x_splu_transposed = splu_solver_transposed.solve(rhs_)

    err_splu_transposed = csr_.T @ x_splu_transposed - rhs_
    print(
        "max residual element: %e (splu transposed)"
        % (np.max(np.abs(err_splu_transposed)))
    )

    # Create PARDISO factorization
    obj_ = pyrdiso.CustomObject(
        data=csr_.data,
        indptr=csr_.indptr,
        indices=csr_.indices,
        shape=csr_.shape,
        mtype=pyrdiso.MTYPE_REAL_NONSYMMETRIC,
    )

    trip_on_error(obj_)

    x_pardiso = obj_.solve(rhs=rhs_, transpose=pyrdiso.SOLVE_NO_TRANSPOSE)
    trip_on_error(obj_)

    err_pardiso = csr_ @ x_pardiso - rhs_
    print("max residual element: %e (pardiso)" % (np.max(np.abs(err_pardiso))))

    x_pardiso_transposed = obj_.solve(rhs=rhs_, transpose=pyrdiso.SOLVE_TRANSPOSE)
    trip_on_error(obj_)

    err_pardiso_transposed = csr_.T @ x_pardiso_transposed - rhs_
    print(
        "max residual element: %e (pardiso transposed)"
        % (np.max(np.abs(err_pardiso_transposed)))
    )

    print(
        "max elem diff btw solutions:",
        np.max(np.abs(x_splu - x_pardiso)),
        "(splu vs pardiso)",
    )
    print(
        "max elem diff btw solutions:",
        np.max(np.abs(x_splu_transposed - x_pardiso_transposed)),
        "(transposed eq, splu vs pardiso)",
    )

    if args.symmetric:

        # Upper triangular part only, and with explicit diagonal zeros if needed
        csr_upper_ = make_upper_csr_proper(csr_)

        assert (
            pyrdiso.check_csr(
                indptr=csr_upper_.indptr,
                indices=csr_upper_.indices,
                shape=csr_upper_.shape,
                check=pyrdiso.CHECK_UPPER_TRIANGULAR,
                verbose=1,
            )
            == 0
        )

        # Create PARDISO symmetric indefinite factorization
        obj_sym_ = pyrdiso.CustomObject(
            data=csr_upper_.data,
            indptr=csr_upper_.indptr,
            indices=csr_upper_.indices,
            shape=csr_upper_.shape,
            mtype=pyrdiso.MTYPE_REAL_SYMMETRIC_INDEFINITE,
        )

        trip_on_error(obj_sym_)

        x_pardiso_sym = obj_sym_.solve(rhs=rhs_)

        trip_on_error(obj_sym_)

        err_pardiso_sym = csr_ @ x_pardiso_sym - rhs_
        print(
            "max residual element: %e (pardiso/sym)" % (np.max(np.abs(err_pardiso_sym)))
        )

        print(
            "max elem diff btw solutions:",
            np.max(np.abs(x_pardiso_sym - x_pardiso)),
            "(pardiso vs pardiso/sym)",
        )

    if args.symmetric and args.positive:

        # Create PARDISO symmetric positive definite factorization
        obj_pos_ = pyrdiso.CustomObject(
            data=csr_upper_.data,
            indptr=csr_upper_.indptr,
            indices=csr_upper_.indices,
            shape=csr_upper_.shape,
            mtype=pyrdiso.MTYPE_REAL_SYMMETRIC_POSITIVE,
        )

        trip_on_error(obj_pos_)

        x_pardiso_pos = obj_pos_.solve(rhs=rhs_)

        trip_on_error(obj_pos_)

        err_pardiso_pos = csr_ @ x_pardiso_pos - rhs_
        print(
            "max residual element: %e (pardiso/sym/pos)"
            % (np.max(np.abs(err_pardiso_pos)))
        )

        print(
            "max elem diff btw solutions:",
            np.max(np.abs(x_pardiso_pos - x_pardiso)),
            "(pardiso vs pardiso/sym/pos)",
        )

    print("done.")
