#include <stdio.h>

#include "mkl_types.h"
#include "mkl_sparse_handle.h"

#if !defined(MKL_ILP64)
#define IFORMAT "%i"
#else
#define IFORMAT "%lli"
#endif

// mkl_structure = { MKL_GENERAL_STRUCTURE, MKL_UPPER_TRIANGULAR, MKL_LOWER_TRIANGULAR, MKL_STRUCTURAL_SYMMETRIC }

int validate_csr_data(MKL_INT n, 
                      MKL_INT* ia, 
                      MKL_INT* ja, 
                      int one_based,
                      int mkl_structure, 
                      int verbose)
{
  sparse_checker_error_values check_err_val;
  sparse_struct pt;

  sparse_matrix_checker_init(&pt);

  pt.n = n;
  pt.csr_ia = ia;
  pt.csr_ja = ja;
  pt.indexing         = (one_based > 0 ? MKL_ONE_BASED : MKL_ZERO_BASED);
  pt.matrix_structure = mkl_structure;
  pt.print_style      = MKL_C_STYLE;
  pt.message_level    = (verbose > 0 ? MKL_PRINT : MKL_NO_PRINT);
  pt.matrix_format    = MKL_CSR;

  check_err_val = sparse_matrix_checker(&pt);

  const int error = (check_err_val == MKL_SPARSE_CHECKER_SUCCESS ? 0 : 1);

  if (verbose > 0 && error != 0) {
    printf(
      "Matrix check details: (" IFORMAT ", " IFORMAT ", " IFORMAT ")\n", 
      pt.check_result[0], pt.check_result[1], pt.check_result[2]
    );
  }

  if (check_err_val == MKL_SPARSE_CHECKER_NONTRIANGULAR && verbose > 0)
    printf("Matrix check result: MKL_SPARSE_CHECKER_NONTRIANGULAR\n");

  if (check_err_val == MKL_SPARSE_CHECKER_SUCCESS && verbose > 0)
    printf("Matrix check result: MKL_SPARSE_CHECKER_SUCCESS\n");

  if (check_err_val == MKL_SPARSE_CHECKER_NON_MONOTONIC && verbose > 0)
    printf("Matrix check result: MKL_SPARSE_CHECKER_NON_MONOTONIC\n");

  if (check_err_val == MKL_SPARSE_CHECKER_OUT_OF_RANGE && verbose > 0)
    printf("Matrix check result: MKL_SPARSE_CHECKER_OUT_OF_RANGE\n");

  if (check_err_val == MKL_SPARSE_CHECKER_NONORDERED && verbose > 0)
    printf("Matrix check result: MKL_SPARSE_CHECKER_NONORDERED\n");

  return error;
}
