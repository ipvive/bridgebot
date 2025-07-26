from absl import logging
from reason import reason


all_proofs = reason.load("ncard/1card/1card.reason")


def node(name):
    try:
        return all_proofs.entities[name]
    except:
        logging.error("%s not in %s", name, all_proofs.entities.keys())


def no_dangling_references(node):
    pass  # FIXME


all_validations = {"dangling": lambda node: no_dangling_references(node)}


def validate(node, validations=all_validations):
    success = True
    for name, validation in validations.items():
        if not validation(node):
            success = False
    return success

        
