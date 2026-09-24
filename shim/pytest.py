"""Minimal stand-in for pytest (raises, mark.parametrize), used only when real pytest is unavailable."""
import contextlib


class _Info:
    value = None


@contextlib.contextmanager
def raises(exc):
    info = _Info()
    try:
        yield info
    except exc as e:
        info.value = e
        return
    raise AssertionError(f"DID NOT RAISE {exc.__name__}")


class _Mark:
    @staticmethod
    def parametrize(names, values):
        def deco(f):
            f._params = ([n.strip() for n in names.split(",")], values)
            return f
        return deco


mark = _Mark()
