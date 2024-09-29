# pyrdiso

Interface to `MKL`'s `PARDISO` sparse linear solver. Create custom `python` objects that hold either a symmetric indefinite, symmetric positive, or a general non-symmetric matrix factorization. This can then be used to solve for a specific right-hand side as needed (or for the transposed equation, if applicable). Intended to work with `scipy.sparse` matrices, mainly of the `CSR` type (see example scripts).

## Installation and basic smoke-test

1. Install `MKL` (and the development files) in whatever way works.
2. Rebuild and install this repository: `pip install -e .`
3. Check health of install: `python scripts/smoke.py`

No error messages? Then OK.

Installation/testing has been successful with `Ubuntu` versions `20.04`, and `22.04` (under `WSL2`).

### More serious testing

4. Run `python scripts/test_random_matrix.py --random-diag 0.1`
5. Run `python scripts/test_random_matrix.py --random-diag 0.1 --symmetric`
6. Run `python scripts/test_random_matrix.py --random-diag 0.1 --symmetric --positive`

## Examples:

See `scripts/` folder for various tests and use-cases.

To auto-download and time the factorization and solution times for various sparse matrices run this:
```
python3 scripts/matrixmarket_benchmark.py --repeats 50
```
which will also compare to the `scipy.sparse` default solver `SUPERLU`, and if installed also to `UMFPACK` (also through `scipy.sparse`).

## Development notes

It is useful to run `python setup.py build_ext --force --inplace` to see the build commands and progress in detail. 

## References

- https://docs.python.org/3/extending/index.html
- https://numpy.org/doc/stable/reference/c-api/index.html
- https://www.intel.com/content/www/us/en/docs/onemkl/developer-reference-c/2023-0/sparse-matrix-checker-routines.html
- https://www.intel.com/content/www/us/en/docs/onemkl/developer-reference-c/2023-0/onemkl-pardiso-parameters-in-tabular-form.html
- https://www.intel.com/content/www/us/en/docs/onemkl/developer-reference-c/2023-0/sparse-matrix-storage-formats.html
