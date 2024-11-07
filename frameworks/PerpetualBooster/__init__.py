from importlib.metadata import version as libversion


def version():
    return libversion('perpetual')


def run(*args, **kwargs):
    from .exec import run
    return run(*args, **kwargs)
