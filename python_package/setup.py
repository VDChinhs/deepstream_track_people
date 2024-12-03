from setuptools import setup, Extension
import subprocess
from setuptools.command.build_ext import build_ext
import pybind11

ext_modules = [
    Extension(
        "landmarkcus",  # Name of the Python module
        ["/home/jetsonvy/DucChinh/python_package/landmark.cpp"],  # Source file
        include_dirs=[
            pybind11.get_include(),  # pybind11 headers
            "/opt/nvidia/deepstream/deepstream-6.0/sources/includes/",  # DeepStream headers
            "/usr/local/cuda-10.2/targets/aarch64-linux/include/",  # CUDA headers
            "/usr/include/glib-2.0",  # Glib headers
            "/usr/lib/aarch64-linux-gnu/glib-2.0/include/"  # Glib config headers
        ],
        libraries=["nvinfer", "nvds_meta", "glib-2.0"],  # Add necessary libraries
        library_dirs=[
            "/opt/nvidia/deepstream/deepstream-6.0/lib/",  # Path to DeepStream libraries
            "/usr/lib/aarch64-linux-gnu"  # Glib library path
        ],
        language="c++"
    ),
]

setup(
    name="nvds_custom",
    version='0.1',
    author='VDChinhs',
    ext_modules=ext_modules,
    zip_safe=False,
)