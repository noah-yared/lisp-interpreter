"""
LISP Interpreter
"""

#!/usr/bin/env python3

import sys

sys.setrecursionlimit(20_000)


#############################
# Scheme-related Exceptions #
#############################

class SchemeError(Exception):
    """
    A type of exception to be raised if there is an error with a Scheme
    program.  Should never be raised directly; rather, subclasses should be
    raised.
    """
    pass


class SchemeNameError(SchemeError):
    """
    Exception to be raised when looking up a name that has not been defined.
    """
    pass


class SchemeSyntaxError(SchemeError):
    """
    Exception to be raised if there is a syntax error encountered when parsing 
    source.
    """
    pass


class SchemeEvaluationError(SchemeError):
    """
    Exception to be raised if there is an error during evaluation other than a
    SchemeNameError.
    """
    pass


############################
# Tokenization and Parsing #
############################


def number_or_symbol(value):
    """
    Helper function: given a string, convert it to an integer or a float if
    possible; otherwise, return the string itself

    >>> number_or_symbol('8')
    8
    >>> number_or_symbol('-5.32')
    -5.32
    >>> number_or_symbol('1.2.3.4')
    '1.2.3.4'
    >>> number_or_symbol('x')
    'x'
    """
    try:
        return int(value)
    except ValueError:
        try:
            return float(value)
        except ValueError:
            return value


def parse_semicolons(source: str) -> list[str]:
    """
    Removes comments from source code.
    Args: source - input string
    Returns: list of code segments without comments
    """
    scs_removed, curr = [], 0
    while curr < len(source):
        tok_end = source.find(";", curr)
        if tok_end == -1:  # no more semicolons to parse out
            scs_removed.append(source[curr:])
            return scs_removed
        scs_removed.append(source[curr:tok_end])
        eol = source.find("\n", tok_end)
        if eol == -1:  # on last line
            return scs_removed
        curr = eol + 1
    return scs_removed


def tokenize(source: str):
    """
    Splits an input string into meaningful tokens (left parens, right parens,
    other whitespace-separated values).  Returns a list of strings.

    Arguments:
        source (str): a string containing the source code of a Scheme
                      expression

    >>> source = "bare-name"
    >>> tokenize(source)
    ['bare-name']
    """
    # Remove comments / semicolons
    scs_parsed = parse_semicolons(source)

    # Split on whitespace
    ws_parsed = []
    for token in scs_parsed:
        ws_parsed.extend(token.split())

    # Handle parentheses
    tokens = []
    for token in ws_parsed:
        left_idx, right_idx = 0, len(token) - 1
        while left_idx != len(token) and token[left_idx] in ("()"):
            tokens.append(token[left_idx])  # opening parens token
            left_idx += 1
        if left_idx == len(token):
            continue
        while token[right_idx] in ("()"):
            right_idx -= 1
        tokens.append(token[left_idx : right_idx + 1])
        tokens.extend(token[right_idx + 1 :])  # closing parens tokens

    return tokens


def find_matching_paren(tokens: list[str], left_idx: int, right_idx: int):
    """
    Finds the matching closing parenthesis for a given opening parenthesis.
    Args: tokens - list of tokens, substr - substring to find, left_idx - left index,
    right_idx - right index
    Returns: index of matching closing parenthesis
    """
    i, stack = left_idx, ["("]
    while stack and i <= right_idx:
        if tokens[i] == ")":
            stack.pop()
        elif tokens[i] == "(":
            stack.append("(")
        i += 1
    # print (f"start is {left_idx - 1}, end is {i}")
    if stack:
        raise SchemeSyntaxError("Missing parentheses")
    return i - 1


def parse_indices(tokens, left_idx, right_idx):
    """
    Parses a substring of the tokens list
    Args: tokens - list of tokens, left_idx - left index, right_idx - right index
    Returns: parsed expression
    """
    if left_idx == right_idx:
        if tokens[left_idx] not in "()":
            return number_or_symbol(tokens[left_idx])
        raise SchemeSyntaxError("Lone parentheses")
    if tokens[left_idx] == "(" and tokens[right_idx] == ")":
        parsed_expr, i = [], left_idx + 1
        while i < right_idx:
            if tokens[i] == "(":
                subexpr_end = find_matching_paren(tokens, i + 1, right_idx - 1)
                parsed_expr.append(parse_indices(tokens, i, subexpr_end))
                i = subexpr_end + 1
            else:
                if tokens[i] == ")":
                    raise SchemeSyntaxError("Mismatching parentheses")
                parsed_expr.append(number_or_symbol(tokens[i]))
                i += 1
        return parsed_expr
    raise SchemeSyntaxError("Invalid character")


def parse(tokens: list[str]):
    """
    Parses a list of tokens, constructing a representation where:
        * symbols are represented as Python strings
        * numbers are represented as Python ints or floats
        * S-expressions are represented as Python lists

    Arguments:
        tokens (list): a list of strings representing tokens

    >>> tokenize("(cat (dog (tomato)))")
    ['(', 'cat', '(', 'dog', '(', 'tomato', ')', ')', ')']

    >>> tokenize("(foo (bar 3.14))")
    ['(', 'foo', '(', 'bar', '3.14', ')', ')']
    """
    return parse_indices(tokens, 0, len(tokens) - 1)


######################
# Built-in Functions #
######################


def calc_sub(*args):
    """
    Subtracts a list of numbers.
    Args: args - list of numbers
    Returns: difference of numbers
    """
    if not args:
        return 0
    if len(args) == 1:
        return -args[0]
    first_num, *rest_nums = args
    return first_num - scheme_builtins["+"](*rest_nums)


def calc_mul(*args):
    """
    Multiplies a list of numbers.
    Args: args - list of numbers
    Returns: product of numbers
    """
    if not args:
        return 1
    if len(args) == 1:
        return args[0]
    first_num, *rest_nums = args
    return first_num * calc_mul(*rest_nums)


def calc_div(*args):
    """
    Divides a list of numbers.
    Args: args - list of numbers
    Returns: quotient of numbers
    """
    if not args:
        raise SchemeEvaluationError
    if len(args) == 1:
        return args[0]
    *rest_nums, last_num = args
    return calc_div(*rest_nums) / last_num


def equals(*args) -> bool:
    """
    Check if all args are equivalent
    Args: args - list of expressions to evaluate
    Returns: true if each expression  evaluates to same value
    """
    if len(args) <= 1:
        raise SchemeEvaluationError
    if len(args) == 2:
        return args[0] == args[1]
    first_expr, *rest_exprs = args
    return first_expr == rest_exprs[0] and equals(*rest_exprs)

def lt_eq(*args) -> bool:
    """
    Check if args are in nondecreasing order
    """
    if len(args) <= 1:
        raise SchemeEvaluationError
    if len(args) == 2:
        return args[0] <= args[1]
    first_expr, *rest_exprs = args
    return first_expr <= rest_exprs[0] and lt_eq(*rest_exprs)

def lt(*args) -> bool:
    """
    Check if args are in strictly increasing order
    """
    if len(args) <= 1:
        raise SchemeEvaluationError
    if len(args) == 2:
        return args[0] < args[1]
    first_expr, *rest_exprs = args
    return first_expr < rest_exprs[0] and lt(*rest_exprs)


def gt(*args) -> bool:
    """
    Check if args are in strictly decreasing order
    """
    if len(args) <= 1:
        raise SchemeEvaluationError
    if len(args) == 2:
        return args[0] > args[1]
    first_expr, *rest_exprs = args
    return first_expr > rest_exprs[0] and gt(*rest_exprs)

def gt_eq(*args) -> bool:
    """
    Check if args are in nonincreasing order
    """
    if len(args) <= 1:
        raise SchemeEvaluationError
    if len(args) == 2:
        return args[0] >= args[1]
    first_expr, *rest_exprs = args
    return first_expr >= rest_exprs[0] and gt_eq(*rest_exprs)

def negate(*args):
    """
    Negates given expression
    """
    if len(args) != 1:
        raise SchemeEvaluationError
    return not evaluate(*args)

def length(expr):
    """
    Returns the length of a linked list
    """
    if not is_list(expr):
        raise SchemeEvaluationError("no length for non-list")
    if isinstance(expr, EmptyList):
        return 0
    return len(list(expr)) - 1

def list_ref(expr, index):    
    if not isinstance(expr, (EmptyList, Pair)):
        raise SchemeEvaluationError("cannot index non-list")
    if index < 0:
        raise SchemeEvaluationError("do not support negative indices!")
    if isinstance(expr, EmptyList) or index >= len(list(expr)) - 1:
        raise SchemeEvaluationError("index out of bounds")
    for i, elt in enumerate(expr):
        if i == index:
            return elt
    raise SchemeEvaluationError("index out of bounds")


def append_item(*args):
    if len(args) == 0:
        return EmptyList()
    
    linked_list = EmptyList()
    for i in range(len(args) - 1, -1, -1):
        if isinstance(args[i], EmptyList):
            continue
        if not is_list(args[i]):
            raise SchemeEvaluationError("Argument must all be lists for append")
        for elt in reversed(list(args[i])[:-1]):
            linked_list = Pair(elt, linked_list)
    return linked_list

def is_list(expr):
    """
    Check if expr is a list
    """
    if isinstance(expr, EmptyList):
        return True
    if not isinstance(expr, Pair):
        return False
    cdr = expr.cdr
    while isinstance(cdr, Pair):
        cdr = cdr.cdr
    return isinstance(cdr, EmptyList)


scheme_builtins = {
    "+": lambda *args: sum(args),
    "-": calc_sub,
    "*": calc_mul,
    "/": calc_div,
    "equal?": equals,
    "<=": lt_eq,
    "<": lt,
    ">=": gt_eq,
    ">": gt,
    "#t": True,
    "#f": False,
    "not": negate,
    "list?": is_list,
    "length": length,
    "list-ref": list_ref,
    "append": append_item
}


###############
# Frame Class #
###############


class Frame:
    """
    A class representing a frame in the Scheme interpreter.
    """

    def __init__(self, bindings: dict, parent):
        # Store bindings and parent frame
        self.bindings = bindings
        self.parent = parent

    def __contains__(self, key):
        return key in self.bindings or key in self.parent

    def make_assignment(self, key, val):
        self.bindings[key] = val

    def __getitem__(self, key):
        try:
            if key not in self.bindings:
                return self.parent[key]
            return self.bindings[key]
        except Exception as exc:
            raise SchemeNameError("variable does not exist in frame") from exc
        
    def __setitem__(self, key, val):
        frame = self
        while isinstance(frame, Frame):
            if key in frame.bindings:
                frame.bindings[key] = val
                return val
            frame = frame.parent
        raise SchemeNameError("variable does not exist in frame")

    def __iter__(self):
        # Current frame bindings
        for var in self.bindings:
            yield var
        # Parent frame bindings
        variables = set(self.bindings)
        for var in self.parent:
            if var not in variables:
                yield var
                variables.add(var)


###################
# Function class #
###################
class Function:
    """
    A class representing a function in the Scheme interpreter.
    """

    def __init__(self, params, expression, parent):
        self.enclosing_frame = parent
        self.params = params
        self.expression = expression

    def __call__(self, *args):
        if len(args) != len(self.params):
            raise SchemeEvaluationError
        bindings = dict(zip(self.params, args))
        calling_frame = Frame(bindings, self.enclosing_frame)
        return evaluate(self.expression, frame=calling_frame)
    

##############
# Pair Class #
##############

class Pair:
    """
    A class representing a pair in the Scheme interpreter.
    """

    def __init__(self, first, second):
        self.car = first
        self.cdr = second

    def __iter__(self):
        yield self.car
        if isinstance(self.cdr, Pair):
            yield from self.cdr
        else:
            yield self.cdr

class EmptyList:
    """
    A class representing an empty list in the Scheme interpreter.
    """
    def __eq__(self, other): # make sure that other EmptyList instances are equal
        return isinstance(other, EmptyList)


##############
# Evaluation #
##############


def or_args(*args, frame=None):
    """
    Evaluates a list of expressions and returns the first truthy value
    as True with short-circuiting (otherwise returns False)
    """
    for expr in args:
        if evaluate(expr, frame=frame):
            return True
    return False


def and_args(*args, frame=None):
    """
    Evaluates a list of expressions and returns the first falsey value
    as False with short-circuiting (otherwise returns True)
    """
    for expr in args:
        if not evaluate(expr, frame=frame):
            return False
    return True


def make_initial_frame():
    """
    Creates an initial frame with an empty bindings dictionary and the built-in
    functions.
    Returns: initial frame
    """
    return Frame({}, scheme_builtins)


def handle_define(tree, frame):
    """
    Handle define statements
    """
    if isinstance(tree[1], list):
        # Function definition
        key = tree[1][0]
        value = Function(tree[1][1:], tree[2], parent=frame)
        frame.make_assignment(key, value)
    else:
        # Variable definition
        key = tree[1]
        value = evaluate(tree[2], frame=frame)
        frame.make_assignment(tree[1], value)
    return value

def handle_lambda(tree, frame):
    """
    Handle lambda expressions
    """
    if isinstance(tree[1], list):
        return Function(tree[1], tree[2], frame)
    raise SchemeSyntaxError("Invalid lambda expression")


def handle_if(tree, frame):
    """
    Handle if statements
    """
    return (
        evaluate(tree[2], frame=frame)
        if (eval:=evaluate(tree[1], frame=frame)) and eval != "#f"
        else evaluate(tree[3], frame=frame)
    )


def handle_and(tree, frame):
    """
    Handle and special form
    """
    if len(tree) <= 2:
        raise SchemeEvaluationError
    return and_args(*[tree[i] for i in range(1, len(tree))], frame=frame)


def handle_or(tree, frame):
    """
    Handle or special form
    """
    if len(tree) <= 2:
        raise SchemeEvaluationError
    return or_args(*[tree[i] for i in range(1, len(tree))], frame=frame)


def handle_cons(tree, frame):
    """
    Handle cons special form
    """
    if len(tree) != 3:
        raise SchemeEvaluationError
    return Pair(evaluate(tree[1], frame=frame), evaluate(tree[2], frame=frame))


def handle_car(tree, frame):
    """
    Handle car special form
    """
    if len(tree) != 2:
        raise SchemeEvaluationError
    try:
        return evaluate(tree[1], frame=frame).car
    except AttributeError as exc:
        raise SchemeEvaluationError("car of non-cons cell") from exc
    

def handle_cdr(tree, frame):
    """
    Handle cdr special form
    """
    if len(tree) != 2:
        raise SchemeEvaluationError
    try:
        return evaluate(tree[1], frame=frame).cdr
    except AttributeError as exc:
        raise SchemeEvaluationError("cdr of non-cons cell") from exc
    
def handle_list(tree, frame):
    """
    Handle list special form
    """
    if len(tree) == 1:
        return EmptyList()
    # push evaluations onto a stack
    stack = [evaluate(tree[i], frame=frame) for i in range(1, len(tree))]
    # Build cdr iteratively from stack
    cdr = EmptyList()
    while len(stack) != 1:
        cdr = Pair(stack.pop(), cdr)
    return Pair(stack[0], cdr)

def handle_begin(tree, frame):
    """
    Handle begin special form
    """
    if len(tree) == 1:
        raise SchemeEvaluationError("No arguments")
    for i in range(1, len(tree) - 1):
        evaluate(tree[i], frame=frame)
    return evaluate(tree[len(tree) - 1], frame=frame) # return the evaluation of the last arg

def handle_del(tree, frame):
    """
    Handle del special form
    """
    if len(tree) != 2:
        raise SchemeEvaluationError("incorrect number of arguments (expected 1)")
    var = tree[1]
    if var not in frame.bindings:
        raise SchemeNameError(f"{var=} is not bound locally")
    return frame.bindings.pop(var)


def handle_let(tree, frame):
    """
    Handle let special form
    """
    if len(tree) != 3:
        raise SchemeEvaluationError("incorrect number of arguments (expected 2)")
    new_bindings = {var: evaluate(value, frame=frame) for [var, value] in tree[1]}
    new_frame = Frame(new_bindings, frame)
    return evaluate(tree[2], frame=new_frame)
    
def handle_setbang(tree, frame):
    """
    Handle set! special form
    """
    if len(tree) != 3: 
        raise SchemeEvaluationError("incorrect number of arguments (expected 2)")
    frame[tree[1]] = (evaluation:=evaluate(tree[2], frame=frame))
    return evaluation

def handle_fncall(tree, frame):
    """
    Handle function calls
    """
    if isinstance(tree[0], (int, float)):
        raise SchemeEvaluationError(f"{tree[0]}({type(tree[0])}) is not callable")
    try:
        callable_fn = evaluate(tree[0], frame=frame)
        return callable_fn(
            *[evaluate(tree[i], frame=frame) for i in range(1, len(tree))]
        )
    except TypeError as exc:
        raise SchemeEvaluationError(f"{tree[0]} is not callable") from exc


def evaluate(tree, frame=None):
    """
    Evaluate the given syntax tree according to the rules of the Scheme
    language.

    Arguments:
        tree (type varies): a fully parsed expression, as the output from the
                            parse function
    """
    # print(f"tree: {tree}")
    if frame is None:
        frame = make_initial_frame()
    
    # Handle literals and variables
    if isinstance(tree, (int, float)):
        return tree
    if isinstance(tree, str):
        if tree not in frame:
            raise SchemeNameError(f"{tree} is not in frame")
        return frame[tree]
    
    if isinstance(tree, list):
        # Empty List:
        if not tree:
            return EmptyList()
        
        # map special forms to their corresponding handlers
        special_forms = {
            "define": handle_define,
            "lambda": handle_lambda,
            "if": handle_if,
            "and": handle_and,
            "or": handle_or,
            "cons": handle_cons,
            "car": handle_car,
            "cdr": handle_cdr,
            "list": handle_list,
            "begin": handle_begin,
            "del": handle_del,
            "let": handle_let,
            "set!": handle_setbang,
        }

        if isinstance(tree[0], str) and tree[0] in special_forms:
            return special_forms[tree[0]](tree, frame) # handle special form
        return handle_fncall(tree, frame) # assume its a function call

    raise SchemeEvaluationError("what did you do???")


def evaluate_file(file_name, frame=None):
    with open(file_name, "r") as f:
        source = f.read()
    return evaluate(parse(tokenize(source)), frame=frame)

if __name__ == "__main__":
    new_frame = make_initial_frame()
    for file_name in sys.argv[1:]:
        evaluate_file(file_name, frame=new_frame)
