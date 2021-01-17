#!/usr/bin/env python3
import logging
import re
import sys
from pathlib import Path

from setuptools import find_packages, setup
from setuptools_scm import get_version

__version__ = get_version(root=".", relative_to=__file__)

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("cuvec.setup")

build_ver = ".".join(__version__.split('.')[:3]).split(".dev")[0]
setup_kwargs = {"use_scm_version": True, "packages": find_packages(exclude=["tests"])}

try:
    from miutil import cuinfo
    from skbuild import setup as sksetup
    nvcc_arches = map(cuinfo.compute_capability, range(cuinfo.num_devices()))
    nvcc_arches = {"%d%d" % i for i in nvcc_arches if i >= (3, 5)}
except Exception as exc:
    log.warning("could not detect CUDA architectures:\n%s", exc)
    setup(**setup_kwargs)
else:
    for i in (Path(__file__).resolve().parent / "_skbuild").rglob("CMakeCache.txt"):
        i.write_text(re.sub("^//.*$\n^[^#].*pip-build-env.*$", "", i.read_text(), flags=re.M))
    sksetup(
        cmake_source_dir="cuvec", cmake_languages=("C", "CXX", "CUDA"),
        cmake_minimum_required_version="3.18", cmake_args=[
            f"-DCUVEC_BUILD_VERSION={build_ver}", f"-DPython3_ROOT_DIR={sys.prefix}",
            "-DCMAKE_CUDA_ARCHITECTURES=" + " ".join(sorted(nvcc_arches))], **setup_kwargs)
