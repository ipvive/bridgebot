from collections import namedtuple
import numpy as np


class TypedValue:
    def __init__(self, type_, value):
        self.type_ = type_
        self.value = value

    def __mul__(self, rhs):
        return TypedValue(self.type_ * rhs.type_, (self.value, rhs.value))


class reals:
    def __le__(self, rhs):
        return self  # (sic)

    def __ge__(self, rhs):
        return self  # (sic)


def constant(type_, value):
    return type_.constant(value)


class Axis:
    def __init__(self, index, size, embedding_type):
        self.index = index
        self.size = size
        self.embedding_type = embedding_type


class Dim:
    def __init__(self, name, labels):
        self.name = name
        self.labels = labels

    def constant(self, value):
        return TypedValue(self, self.parse(value))

    def parse(self, value_like):
        return self.labels.index(value_like)

    def __mul__(self, dim):
        assert dim != self
        return Space(None, [self, dim])

    def __repr__(self):
        label_reprs = [d.__repr__() for d in self.labels]
        return f"Space(name={self.name.__repr__()}, labels={label_reprs})"


class Space:
    def __init__(self, name, dims):
        self.name = name
        self.dims = dims

    def constant(self, value):
        if isinstance(value, dict):
            pv = tuple(d.parse(value[d.name]) for d in self.dims)
        elif isinstance(value, TypedValue):
            assert value.type_ == self
            return value
        else:
            pv = tuple(d.parse(v) for d,v in zip(self.dims, value))
        return TypedValue(self, pv)

    def __str__(self):
        return f"Space(name={self.name}, labels={[d for d in self.dims]})"

    def __repr__(self):
        dim_reprs = [d.__repr__() for d in self.dims]
        return f"Space(name={self.name.__repr__()}, labels={dim_reprs})"


class Tensor:
    def __init__(self, name, space):
        self.name = name
        self.space = space

    def constant(self, value_like):
        pv = [self.space.constant(v) for v in value_like]
        return TypedValue(self, pv)

    def parse(self, value_like):
        pass


class Map:
    def __init__(self, name, domain, range_):
        self.name = name
        self.domain = domain
        self.range = range_


class Collection:
    def __init__(self, name, items):
        self._value_t = namedtuple(name, [item.name for item in items])
        self._type_t = self._value_t(items)

    def __getitem__(self, key):
        if isinstance(key, str):
            return self._type_t.__getattr__(key)
        else:
            sel = [self._type_t.__getattr__(k) for k in key]
            return Collection(None, sel)


Observable = namedtuple('Observable', ['dims'])


class AbsoluteObservable(Observable):
    def parse(self, data):
        shape = tuple(len(d.labels) for d in self.dims)
        if data is not None:
            return np.array(data).reshape(shape)
        else:
            return np.full(shape, np.nan)   

    def unparse(self, tensor):
        pass #TODO


class OneHotObservable(Observable):
    """Returns the positions of the observable, or None."""
    def parse(self, data):
        if data is not None:
            if not (isinstance(data, list) or isinstance(data, tuple)):
                data = (data,) 
            try:
                assert len(self.dims) == len(data)
            except:
                pdb.set_trace()
            ixs = {dim.name: dim.labels.index(datum)
                    if datum in dim.labels else datum
                    for dim, datum in zip(self.dims, data)}
            return ixs
        else:
            return (None,) * len(self.dims)

    def unparse(self, ixs):
        assert len(self.dims) == len(ixs)
        return tuple(dim.labels[idx] if idx is not None else None for idx in ixs)


