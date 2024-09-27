import setuptools
import numpy as np
import os

if os.getenv("MKLROOT") is None:
    MKL_ROOT = "/opt/intel/oneapi/mkl/latest"
else:
    MKL_ROOT = os.getenv("MKLROOT")

setuptools.setup(
    name="pyrdiso",
    version="0.10",
    description="Yet another python interface to the MKL PARDISO sparse linear solver",
    ext_modules=[
        setuptools.Extension(
            name="pyrdiso.pyrdiso",
            sources=["src/pyrdiso.c", "src/utilities.c"],
            include_dirs=[np.get_include(), os.path.join(MKL_ROOT, "include")],
            define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
            extra_compile_args=["-std=c99", "-O2", "-Wall"],
            # library_dirs=[os.path.join(MKL_ROOT, "lib")],
            # runtime_library_dirs=[os.path.join(MKL_ROOT, "lib")],
            extra_link_args=["-lm", "-lgomp"],
            extra_objects=[
                os.path.join(MKL_ROOT, "lib", "libmkl_core.so"),
                os.path.join(MKL_ROOT, "lib", "libmkl_rt.so"),
                os.path.join(MKL_ROOT, "lib", "libmkl_sequential.so"),
                os.path.join(MKL_ROOT, "lib", "libmkl_intel_lp64.so"),
            ],
        ),
    ],
)
