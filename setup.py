from setuptools import setup, find_packages, Extension
import numpy as np

setup(
    ext_modules=[
        Extension(
            "bridgebot.bridge._fastgame",
            sources=["bridgebot/bridge/fastgame/fastgame.c"],
            include_dirs = [np.get_include()],
        )
    ]
)
