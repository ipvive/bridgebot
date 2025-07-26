import setuptools

setuptools.setup(
    name='bridgebot',
    version='0.1.0',
    install_requires=[
        'tensorflow_datasets>=2.1.0',
    ],
    packages=setuptools.find_packages(),
)
