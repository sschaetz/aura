import pyparsing as pp
import pprint

# -----------------------------------------------------------------------------
# AST
# -----------------------------------------------------------------------------

class aura_tree:
    """Aura language syntax tree.
    """
    class function:
        """Represents a function in the syntax tree.
        """
        def __init__(self):
            self.attributes = []
            self.arguments = []
            self.return_type = ''

        def __repr__(self):
            s = 'attributes ['
            for a in self.attributes:
                s += ' ' + a + ' '
            s += ']\n'
            s += 'arguments ['
            for a in self.arguments:
                a.__repr__()
            s += ']\n'
            s += 'returntype ' + self.return_type
            return s

        class argument:
            """Represents an argument to a function.
            """
            def __init__(self, name, typeid, attributes):
                self.name = name
                self.typeid = typeid
                self.attributes = attributes

            def __repr__(self):
                s = self.name + self.typeid
                s += 'attributes ['
                for a in self.attributes:
                    s += ' ' + a + ' '
                s += ']'
                return s

    def __init__(self):
        # List of functions.
        self.functions = {}
        # Current function.
        self.cf = ''

    def add_function(self, toks):
        self.functions[toks[0]] = self.function()
        self.cf = toks[0]

    def add_argument(self, toks):
        toks = toks[0]
        name = toks[0]
        typeid = toks[1]
        if len(toks) > 2:
            attributes = toks[2:]
        else:
            attributes = []

        self.functions[self.cf].arguments.append(
            self.function.argument(name, typeid, attributes))
        pass

    def add_return_type(self, toks):
        self.functions[self.cf].return_type = toks[0]

    def __repr__(self):
        for x in self.functions.items():
            return x.__repr__()

def parse(string):
    ast = aura_tree()
    # --------------------------------------------------------------------------
    # Grammar
    # --------------------------------------------------------------------------

    # Inspired from
    # http://pyparsing.wikispaces.com/file/view/oc.py/150660287/oc.py

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
    function_name = NAME.copy().addParseAction(ast.add_function)
    function_arg = \
        pp.Group(NAME + LBRACK + function_argument_traits + RBRACK). \
        addParseAction(ast.add_argument)

    # Return value.
    function_return = \
        (pp.Suppress(pp.Word('->')) + TYPE).addParseAction(ast.add_return_type)

    # Function interface.
    function_interface = function_attribute + function_name + \
            LPAR + \
            pp.Optional(pp.Group(pp.delimitedList(function_arg))) + \
            RPAR + \
            function_return

    aulang = function_interface
    return aulang.parseString(string), ast



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


parsed, ast = parse(exp)
pp = pprint.PrettyPrinter()
pp.pprint(ast)



