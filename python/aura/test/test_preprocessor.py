from pathlib import Path
import pytest
import sys

LIB_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(LIB_DIR))

from aura.preprocessor import *

def test_get_alang_header():
    for base in ['cuda', 'metal', 'opencl']:
        assert len(get_alang_header(base)) > 0

def test_get_wrong_alang_header():
    with pytest.raises(ValueError) as excinfo:
        get_alang_header('does-not-exist')
    assert 'Unknown aura base' in str(excinfo.value)

def test_transform_kernel_string():
    kernel_string = '''
    const int cool_stuff = <<<insert_cool_stuff_here>>>;
    int mykernel()
    {

    }

    '''
    kernel_string_expected = '''
    const int cool_stuff = {insert_cool_stuff_here};
    int mykernel()
    {{

    }}

    '''
    assert kernel_string_expected == transform_kernel_string(kernel_string)


