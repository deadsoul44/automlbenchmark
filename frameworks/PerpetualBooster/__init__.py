from importlib.metadata import version as lib_version
from amlb.utils import call_script_in_same_dir


def setup(*args, **kwargs):
    call_script_in_same_dir(__file__, "setup.sh", *args, **kwargs)


def version():
    return lib_version('perpetual')


def run(*args, **kwargs):
    from .exec import run
    return run(*args, **kwargs)
