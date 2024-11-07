
def version():
    from perpetual import __version__
    return __version__


def run(*args, **kwargs):
    from .exec import run
    return run(*args, **kwargs)
