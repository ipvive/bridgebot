from setuptools import Extension, setup

def build(setup_kwargs):
    """
    This function is mandatory in order to build the extensions.
    """
    setup_kwargs.update(
        {
            "ext_modules": [
                Extension(
                    "_fastgame",
                    ["src/bridgebot/bridge/fastgame/fastgame.c"],
                )
            ]
        }
    )

