import collections.abc
from itertools import repeat

def _ntuple_same(n):
    def parse(x):
        if isinstance(x, int):
            return tuple(repeat(x, n))
        elif isinstance(x, collections.abc.Iterable):
            assert len(set(x))==1, 'the size of kernel must be the same for each side'
            return tuple(repeat(x[0], n))
    return parse

def _to_ntuple(n):
    def parse(x):
        if isinstance(x, int):
            return tuple(repeat(x, n))
        elif isinstance(x, collections.abc.Iterable):
            if len(set(x))==1:
                return tuple(repeat(x[0], n))
            else:
                assert len(x)==n , 'wrong format'
                return x
    return parse

_pair_same = _ntuple_same(2)
_triple_same = _ntuple_same(3)

_to_pair = _to_ntuple(2)
_to_triple = _to_ntuple(3)
