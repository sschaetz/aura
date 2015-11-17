import pyparsing as pp

# Inspired from http://pyparsing.wikispaces.com/file/view/oc.py/150660287/oc.py

# Generic definitions.
LPAR, RPAR, LBRACK, RBRACK, LBRACE, RBRACE, SEMI, COMMA = \
        map(pp.Suppress, "()[]{};,")
CONST = pp.Keyword('const')
PTR = pp.Keyword('*')

NAME = pp.Word(pp.alphas+"_", pp.alphanums+"_")
TYPE = pp.Optional(CONST) + NAME + pp.Optional(PTR)

# Function argument traits.
READONLY, WRITEONLY, READWRITE, \
FIBERINMESH, FIBERINBUNDLE, BUNDLEINMESH, \
NUMFIBERSINMESH, NUMFIBERSINBUNDLE, NUMBUNDLESINMESH = \
    map(pp.Keyword,
        ["readonly", "writeonly", "readwrite",
         "fiber_in_mesh", "fiber_in_bundle", "bundle_in_mesh",
         "num_fibers_in_mesh", "num_fibers_in_bundle",
         "num_bundles_in_mesh"])
# Function elements.
function_argument_traits = \
    (TYPE + pp.Optional(READONLY ^ WRITEONLY ^ READWRITE)) ^ \
    (FIBERINMESH ^ FIBERINBUNDLE ^ BUNDLEINMESH ^
     NUMFIBERSINMESH ^ NUMFIBERSINBUNDLE ^ NUMBUNDLESINMESH)
function_attribute_host_callable = pp.Keyword("host_callable")
function_attribute = pp.Optional(function_attribute_host_callable)
function_name = NAME
function_arg = pp.Group(NAME + LBRACK + function_argument_traits + RBRACK)

# Return value.
function_return = pp.Word('->') + TYPE

# Function interface.
function_interface = function_attribute + function_name + \
        LPAR + pp.Optional(pp.Group(pp.delimitedList(function_arg))) + RPAR + \
        function_return

aulang = function_interface

exp = """
host_callable myfunction(
    asdf [float readonly],
    asdf2 [float2 readonly],
    foo [num_fibers_in_mesh])
    -> void
"""

func0 = """
host_callable
function1() => void
{
}
"""



print(exp, "->", aulang.parseString(exp).dump())
for k in aulang.parseString(exp).keys():
    print(k)




