from .pyrdiso import *

MTYPE_REAL_SYMMETRIC_INDEFINITE = int(-2)
MTYPE_REAL_SYMMETRIC_POSITIVE = int(2)
MTYPE_REAL_NONSYMMETRIC = int(11)

SOLVE_NO_TRANSPOSE = int(0)
SOLVE_TRANSPOSE = int(2)


def error_message_str(error: int) -> str:
    conversion_dict_ = {
        0: "No error",
        -1: "Input inconsistent",
        -2: "Not enough memory",
        -3: "Reordering problem",
        -4: "Zero pivot, numerical factorization or iterative refinement problem",
        -5: "Unclassified (internal) error",
        -6: "Reordering failed (matrix types 11 and 13 only)",
        -7: "Diagonal matrix is singular",
        -8: "32-bit integer overflow problem",
        -9: "Not enough memory for OOC",
        -10: "Problems with opening OOC temporary files",
        -11: "Read/write problems with the OOC data file",
    }
    return conversion_dict_.get(error, "Unknown error code")


CHECK_GENERAL_STRUCTURE = int(0)
CHECK_UPPER_TRIANGULAR = int(1)
CHECK_LOWER_TRIANGULAR = int(2)
CHECK_STRUCTURAL_SYMMETRIC = int(3)
