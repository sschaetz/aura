import re
from pathlib import Path

supported_base_list = ['cuda', 'metal', 'opencl']

def get_alang_header(base):
    """Get the GPU kernel header for a specific alang base."""
    base = base.lower()

    if base not in supported_base_list:
        raise ValueError('Unknown aura base, choose among' +
                str(supported_base_list))

    files = [Path(__file__).resolve().parent.parent.parent / \
                'include' / 'boost' / 'aura' / 'base' / 'alang.hpp',
            Path(__file__).resolve().parent.parent.parent / \
                'include' / 'boost' / 'aura' / 'base' / base / 'alang.hpp']
    header = ''
    for f in files:
        with f.open() as fh:
            s = fh.read()
            begin = '// PYTHON-BEGIN'
            end = '// PYTHON-END'
            header += s[s.find(begin)+len(begin):s.rfind(end)]
    return header


def transform_kernel_string(k):
    """This function transforms a kernel string to make it compatible with
    Python's .format() syntax. Kernels are C code. The syntax to replace
    strings using Python's .format() is the same as the C syntax to declare
    scope. To write proper C code and avoid {{ // scope }}, we use <<<var>>>
    to define variables that should be replaced. So this function applies the
    following transforms:
    { -> {{
    } -> }}
    <<< -> {
    >>> -> }
    Arguments:
    k                 ---- Kernel string that is transformed.
    """

    # From http://stackoverflow.com/a/6117124
    rep = {"{": "{{", "}": "}}", "<<<": "{", ">>>": "}"}

    rep = dict((re.escape(k), v) for k, v in rep.items())
    pattern = re.compile("|".join(rep.keys()))

    return pattern.sub(lambda m: rep[re.escape(m.group(0))], k)


def array_to_c(arr):
    """Helper function to convert a python array to a string
    string that defines the same array for in a C/C++ program
    """
    cstr = "{" + to_c_string(float(arr[0]))
    for element in arr[1::]:
        cstr = cstr + ",\n " + to_c_string(float(element))
    return cstr + "}"


def to_c_string(value):
    """Shim function that converts a value or array to a string"""
    if hasattr(value, 'to_base_units'):
        return array_to_c(value.m)
    elif isinstance(value, np.ndarray):
        return array_to_c(value)
    elif isinstance(value, float):
        return str(value) + 'f'
    return str(value)

