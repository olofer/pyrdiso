import os
import scipy.io as scio
import scipy.sparse as scsp
import numpy as np
import urllib.request
import argparse

from html.parser import HTMLParser
import time

import pyrdiso

from test_random_matrix import make_upper_csr_proper

URL_BASE = "https://math.nist.gov/pub/MatrixMarket2/Harwell-Boeing"

UMFPACK_AVAILABLE = True
try:
    import scikits.umfpack as umfpack
except ImportError:
    UMFPACK_AVAILABLE = False

"""
example index: https://math.nist.gov/MatrixMarket/data/Harwell-Boeing/cannes/cannes.html
example matrix: https://math.nist.gov/pub/MatrixMarket2/Harwell-Boeing/cannes/can_1054.mtx.gz
https://docs.scipy.org/doc/scipy/reference/generated/scipy.io.mmread.html

NOTE:
  python3 scripts/matrixmarket_benchmark.py --dataset lshape

There must be some setting relating to pivoting that will make PARDISO more reliable for 
this suite of test-matrices (and others). Defaults for UMFPACK/SUPERLU appears more robust.

"""


def superlu_solve(A, b):
    tic0 = time.perf_counter()
    obj_ = scsp.linalg.splu(A)
    tic1 = time.perf_counter()
    sol_ = obj_.solve(b)
    tic2 = time.perf_counter()
    res_ = A @ sol_ - b
    return tic1 - tic0, tic2 - tic1, np.max(np.abs(res_))


def umfpack_solve(A, b):
    scsp.linalg.use_solver(useUmfpack=True)
    tic0 = time.perf_counter()
    solver_ = scsp.linalg.factorized(A)
    tic1 = time.perf_counter()
    sol_ = solver_(b)
    tic2 = time.perf_counter()
    res_ = A @ sol_ - b
    return tic1 - tic0, tic2 - tic1, np.max(np.abs(res_))


def pardiso_solve(A, b, mtype=pyrdiso.MTYPE_REAL_NONSYMMETRIC, Aeval=None):
    tic0 = time.perf_counter()
    obj_ = pyrdiso.CustomObject(
        data=A.data,
        indptr=A.indptr,
        indices=A.indices,
        shape=A.shape,
        mtype=mtype,
    )
    assert obj_.error() == 0, "factorized object nonzero error code (%i)" % (obj_.error())
    tic1 = time.perf_counter()
    sol_ = obj_.solve(b)
    tic2 = time.perf_counter()
    res_ = A @ sol_ - b if Aeval is None else Aeval @ sol_ - b
    return tic1 - tic0, tic2 - tic1, np.max(np.abs(res_))


class MyHTMLParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.candidates = list()

    def handle_starttag(self, tag, attrs):
        if (
            tag == "a"
            and len(attrs) == 1
            and len(attrs[0]) == 2
            and attrs[0][0] == "href"
        ):
            if attrs[0][1].endswith(".html") and attrs[0][1][0] != "/":
                pre_ = attrs[0][1][:-5]
                if len(pre_) == 8:
                    self.candidates.append(pre_)

    def handle_endtag(self, tag):
        pass

    def handle_data(self, data):
        pass

    def download_candidates(self, url_base: str = ""):
        return [(url_base + s + ".mtx.gz") for s in self.candidates]


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", type=str, default="temp")
    parser.add_argument("--dataset", type=str, default="sherman")
    parser.add_argument("--repeats", type=int, default=10)
    args = parser.parse_args()

    folder_ = os.path.join(args.folder, args.dataset)

    if not os.path.isdir(folder_):
        print("creating: ", folder_)
        os.makedirs(folder_)

    file_html_ = os.path.join(folder_, args.dataset + ".html")

    if not os.path.isfile(file_html_):
        url_html_ = URL_BASE + "/" + args.dataset + "/" + args.dataset + ".html"
        print("downloading: ", file_html_)
        _, headers_ = urllib.request.urlretrieve(url_html_, file_html_)
        urllib.request.urlcleanup()
    else:
        print("cached:", file_html_)

    assert os.path.isfile(file_html_), "expected index file does not exist"

    with open(file_html_, "r") as file_:
        file_content_ = file_.read()

    assert isinstance(file_content_, str)

    html_parser = MyHTMLParser()
    html_parser.feed(file_content_)
    data_names_ = html_parser.download_candidates()

    for item_ in data_names_:
        item_url_ = URL_BASE + "/" + args.dataset + "/" + item_
        item_file_ = os.path.join(folder_, item_)
        if not os.path.isfile(item_file_):
            print(item_url_, " =>", item_file_)
            _, headers_ = urllib.request.urlretrieve(item_url_, item_file_)
            urllib.request.urlcleanup()
        else:
            print("cached:", item_file_)

    # Now the folder should be loaded with a set of *.mtx.gz files.
    # Load these using SCIPY.IO.MMREAD(..)

    print("=== EACH MATRIX GETS %i REPEATS PER SOLVER ===" % (args.repeats))

    def print_stats_(ttl_: str, stats_: list):
        print(
            ttl_,
            np.min([k[0] for k in stats_]),
            np.min([k[1] for k in stats_]),
            np.max([k[2] for k in stats_]),
        )

    # TODO: collect the benchmark results into dicts indexed by the benchmark item name
    #       summarize with grand totals per solver at the end ?!

    tic = time.perf_counter()

    for item_ in data_names_:
        item_file_ = os.path.join(folder_, item_)
        coo_data_ = scio.mmread(item_file_)

        """upper_part = scsp.triu(coo_data_, k=1)
        lower_part = scsp.tril(coo_data_, k=-1)
        diag_part = coo_data_.diagonal()
        print(upper_part.count_nonzero(), lower_part.count_nonzero())"""

        sym_residual = coo_data_ - coo_data_.T
        nnz_A_m_At = sym_residual.count_nonzero()

        print(
            item_,
            coo_data_.shape,
            coo_data_.row.dtype,
            coo_data_.col.dtype,
            coo_data_.data.dtype,
            "symmetric" if nnz_A_m_At == 0 else "non-symmetric",
        )

        if coo_data_.shape[0] != coo_data_.shape[1]:
            continue

        csr_ = coo_data_.tocsr()
        csc_ = coo_data_.tocsc()
        rhs_ = np.random.randn(coo_data_.shape[0])

        assert (
            pyrdiso.check_csr(
                indptr=csr_.indptr,
                indices=csr_.indices,
                shape=csr_.shape,
                check=pyrdiso.CHECK_GENERAL_STRUCTURE,
            )
            == 0
        )

        try:
            splu_stats = [superlu_solve(csc_, rhs_) for _ in range(args.repeats)]
            print_stats_("SUPERLU    ", splu_stats)
        except Exception as E:
            print(E)

        try:
            pyrdiso_stats = [pardiso_solve(csr_, rhs_) for _ in range(args.repeats)]
            print_stats_("PARDISO    ", pyrdiso_stats)
        except Exception as E:
            print(E)

        if nnz_A_m_At == 0:
            csr_upper_ = make_upper_csr_proper(csr_)
            try:
                pyrdiso_stats_sym = [
                    pardiso_solve(
                        csr_upper_,
                        rhs_,
                        mtype=pyrdiso.MTYPE_REAL_SYMMETRIC_INDEFINITE,
                        Aeval=csr_,
                    )
                    for _ in range(args.repeats)
                ]
                print_stats_("PARDISO/SYM", pyrdiso_stats_sym)
            except Exception as E:
                print(E)

        if UMFPACK_AVAILABLE:
            try:
                umfpack_stats = [umfpack_solve(csc_, rhs_) for _ in range(args.repeats)]
                print_stats_("UMFPACK    ", umfpack_stats)
            except Exception as E:
                print(E)

    toc = time.perf_counter()

    elapsed = toc - tic
    print("total benchmark runtime: %.2f sec" % (elapsed))

    print("=== DONE ===")
