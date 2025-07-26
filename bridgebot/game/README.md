# Building the module
    
    python3 -m venv .env
    . ./.env/bin/activate
    pip install maturin
    maturin develop

# Using the module

    $ python3

    Python 3.10.12 (main, Jun 11 2023, 05:26:28) [GCC 11.4.0] on linux
    Type "help", "copyright", "credits" or "license" for more information.
    >>> import game
    >>> d = game.Deal.random_deal()
    >>> d.show()
        'Deal { dealer: East, players: [Player { _name: "Rodwell" }, ...

# Running some tests

    maturin develop
    python3 wrapper_test.py
