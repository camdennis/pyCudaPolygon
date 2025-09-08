import os
import subprocess
from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext

class CMakeBuild(build_ext):
    def run(self):
        build_dir = os.path.join(self.build_temp, 'build')
        os.makedirs(build_dir, exist_ok=True)

        # Set the source directory for CMake (project root directory)
        source_dir = os.path.abspath(os.path.dirname(__file__))  # Get the absolute path of the current directory where setup.py is located

        # Run cmake to configure the build
        print(f"Running CMake in {source_dir}")
        subprocess.check_call(['cmake', source_dir], cwd=build_dir)

        # Build with make
        print("Running make")
        subprocess.check_call(['make'], cwd=build_dir)

        # Continue with the default build process
        super().run()

setup(
    name='pyCudaPolygon',
    version='1.0',
    packages=['pyCudaPolygon'],
    ext_modules=[Extension(
        name='pyCudaElasto',
        sources=['src/pybindWrapper.cpp'],  # Point to your C++/CUDA binding source
    )],
    cmdclass={'build_ext': CMakeBuild},
    install_requires=[
        'numpy',  # Example of Python dependencies
        'scipy',
    ],
    include_package_data=True,
    zip_safe=False,
)
