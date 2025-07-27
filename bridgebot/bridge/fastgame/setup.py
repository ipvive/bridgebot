# src/bridgebot/bridge/fastgame/setup.py
from setuptools import setup, Extension
import numpy

setup(
    name="fastgame",
    version="1.0",
    description="The fastgame C extension",
    # This provides the C compiler with the necessary numpy headers
    include_dirs=[numpy.get_include()],
    ext_modules=[
        Extension(
            # The compiled module will be a top-level one named _fastgame
            "_fastgame",
            sources=["fastgame.c"],
        )
    ],
)
