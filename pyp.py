#!/usr/bin/env python3
import argparse
import ast
import inspect
import sys
from typing import Any, List, Optional, Set, Tuple

__all__ = ["pypprint"]


def pypprint(*args, **kwargs):  # type: ignore
    """Replacement for ``print`` that special-cases dicts and iterables.

    - Dictionaries are printed one line per key-value pair, with key and value colon-separated.
    - Iterables (excluding strings) are printed one line per item
    - Everything else is delegated to ``print``

    """
    from typing import Iterable

    if len(args) != 1:
        print(*args, **kwargs)
        return
    x = args[0]
    if isinstance(x, dict):
        for k, v in x.items():
            print(f"{k}:", v, **kwargs)
    elif isinstance(x, Iterable) and not isinstance(x, str):
        for i in x:
            print(i, **kwargs)
    else:
        print(x, **kwargs)


def find_names(tree: ast.AST) -> Tuple[Set[str], Set[str]]:
    """Returns a tuple of defined and undefined names in the given AST.

    A defined name is any name that is stored to (or is a function argument).
    An undefined name is any name that is loaded before it is defined.

    Note that we ignore deletes and scopes. Our notion of definition is very simplistic; once
    something is defined, it's never undefined. This is an okay approximation for our use case.
    Note used builtins will appear in undefined names.

    """
    undefined = set()
    defined = set()

    class _Finder(ast.NodeVisitor):
        def generic_visit(self, node: ast.AST) -> None:
            # Adapted from ast.NodeVisitor.generic_visit, but re-orders traversal a little
            def order(f_v: Tuple[str, Any]) -> int:
                return {"generators": -2, "iter": -2, "value": -1}.get(f_v[0], 0)

            for _field, value in sorted(ast.iter_fields(node), key=order):
                if isinstance(value, list):
                    for item in value:
                        if isinstance(item, ast.AST):
                            self.visit(item)
                elif isinstance(value, ast.AST):
                    self.visit(value)

        def visit_Name(self, node: ast.Name) -> None:
            if isinstance(node.ctx, ast.Load):
                if node.id not in defined:
                    undefined.add(node.id)
            elif isinstance(node.ctx, ast.Store):
                defined.add(node.id)
            # Ignore deletes, see docstring
            self.generic_visit(node)

        def visit_AugAssign(self, node: ast.AugAssign) -> None:
            if isinstance(node.target, ast.Name):
                if node.target.id not in defined:
                    undefined.add(node.target.id)
            self.generic_visit(node)

        def visit_arg(self, node: ast.arg) -> None:
            # Mark arguments as defined, see docstring
            defined.add(node.arg)
            self.generic_visit(node)

        def visit_alias(self, node: ast.alias) -> None:
            # Mark imports as defined
            defined.add(node.asname if node.asname is not None else node.name)
            self.generic_visit(node)

    _Finder().visit(tree)
    return defined, undefined


class PypError(Exception):
    pass


class PypTransform:
    """PypTransform is responsible for transforming all input code.

    A lot of pyp's magic comes from it making decisions based on defined and undefined names in the
    input. This class helps keep track of that state as things change based on transformations. In
    general, the logic in here is very sensitive to reordering; there are various implicit
    assumptions about what transformations have happened and what names have been defined. But
    the code is pretty small and the tests are good, so you should be okay!

    """

    def __init__(
        self, before: List[str], code: List[str], after: List[str], define_pypprint: bool
    ) -> None:
        self.before_tree = ast.parse("\n".join(before))
        self.tree = ast.parse("\n".join(code))
        self.after_tree = ast.parse("\n".join(after))

        self.defined: Set[str] = set()
        self.undefined: Set[str] = set()
        for t in (self.before_tree, self.tree, self.after_tree):
            _def, _undef = find_names(t)
            self.undefined |= _undef - self.defined
            self.defined |= _def

        self.define_pypprint = define_pypprint

        # The print statement ``build_output`` will add, if it determines it needs to.
        self.implicit_print: Optional[ast.Call] = None

    def define(self, name: str) -> None:
        """Defines a name."""
        self.defined.add(name)
        self.undefined.discard(name)

    def get_valid_name(self, name: str) -> str:
        """Return a name related to ``name`` that does not conflict with existing definitions."""
        while name in self.defined:
            name += "_"
        return name

    def build_output(self) -> None:
        """Ensures that the AST prints something.

        This is done by either a) checking whether we load a thing that prints, or b) if the last
        thing in the tree is an expression, modifying the tree to print it.

        """
        if self.undefined & {"print", "pprint", "pypprint"}:  # has an explicit print
            return

        def inner(tree: ast.Module, use_pypprint: bool = False) -> bool:
            if not tree.body:
                return False
            if not isinstance(tree.body[-1], ast.Expr):
                if not isinstance(tree.body[-1], ast.Pass):
                    return False
                del tree.body[-1]
                return True

            if isinstance(tree.body[-1].value, ast.Name):
                output = tree.body[-1].value.id
                tree.body.pop()
            else:
                output = self.get_valid_name("output")
                self.define(output)
                tree.body[-1] = ast.Assign(
                    targets=[ast.Name(id=output, ctx=ast.Store())], value=tree.body[-1].value
                )

            print_fn = "print"
            if use_pypprint:
                print_fn = "pypprint"
                self.undefined.add("pypprint")

            if_print = ast.parse(f"if {output} is not None: {print_fn}({output})").body[0]
            tree.body.append(if_print)

            self.implicit_print = if_print.body[0].value  # type: ignore
            return True

        # First attempt to add a print to self.after_tree, then to self.tree
        # We use pypprint in self.after_tree and print in self.tree, although the latter is
        # subject to change later on if we call ``use_pypprint_for_implicit_print``. This logic
        # could be a little simpler if we refactored so that we know what transformations we will
        # do before we do them.
        success = inner(self.after_tree, True) or inner(self.tree)
        if not success:
            raise PypError(
                "Code doesn't generate any output; either explicitly print something, end with "
                "an expression that pyp can print, or explicitly end with `pass`."
            )

    def use_pypprint_for_implicit_print(self) -> None:
        """If we implicitly print, use pypprint instead of print."""
        if self.implicit_print is not None:
            self.implicit_print.func.id = "pypprint"  # type: ignore
            # Make sure we import it later
            self.undefined.add("pypprint")

    def build_input(self) -> None:
        """Modifies the AST to use input from stdin.

        How we do this depends on which magic variables are used.

        """
        # We'll use sys here no matter what; add it to undefined so we import it later
        self.undefined.add("sys")

        MAGIC_VARS = {
            "index": {"i", "idx", "index"},
            "loop": {"line", "x", "l", "s"},
            "input": {"lines", "stdin"},
        }
        possible_vars = {typ: names & self.undefined for typ, names in MAGIC_VARS.items()}

        if not any(possible_vars.values()):
            no_pipe_assertion = ast.parse(
                "assert sys.stdin.isatty() or not sys.stdin.read(), 'No candidates found for loop "
                "variable or input variable, so assert nothing is piped in'"
            )
            self.tree.body = no_pipe_assertion.body + self.tree.body
            self.use_pypprint_for_implicit_print()
            return

        if (possible_vars["loop"] or possible_vars["index"]) and possible_vars["input"]:
            raise PypError("Candidates found for both loop variable and input variable")

        for typ, names in possible_vars.items():
            if len(names) > 1:
                raise PypError(f"Multiple candidates for {typ} variable")

        if possible_vars["loop"] or possible_vars["index"]:
            # We'll loop over stdin and define loop / index variables
            idx_var = possible_vars["index"].pop() if possible_vars["index"] else None
            loop_var = possible_vars["loop"].pop() if possible_vars["loop"] else None

            if loop_var:
                self.define(loop_var)
            if idx_var:
                self.define(idx_var)
            if loop_var is None:
                loop_var = "_"

            if idx_var:
                for_loop = f"for {idx_var}, {loop_var} in enumerate(sys.stdin): "
            else:
                for_loop = f"for {loop_var} in sys.stdin: "
            for_loop += f"{loop_var} = {loop_var}.rstrip('\\n')"

            loop: ast.For = ast.parse(for_loop).body[0]  # type: ignore
            loop.body.extend(self.tree.body)
            self.tree.body = [loop]
        else:
            # We'll read from stdin and define the necessary input variable
            input_var = possible_vars["input"].pop()
            self.define(input_var)

            if input_var == "stdin":
                input_assign = ast.parse(f"{input_var} = sys.stdin")
            else:
                input_assign = ast.parse(f"{input_var} = [x.rstrip('\\n') for x in sys.stdin]")

            self.tree.body = input_assign.body + self.tree.body
            self.use_pypprint_for_implicit_print()

    def build_missing_imports(self) -> None:
        """Modifies the AST to import undefined names."""
        missing_names = self.undefined - set(dir(__import__("builtins")))

        if self.define_pypprint and "pypprint" in missing_names:
            # Add the definition of pypprint to the AST
            self.before_tree.body = (
                ast.parse(inspect.getsource(pypprint)).body + self.before_tree.body
            )
            missing_names.remove("pypprint")

        if not missing_names:
            return

        subimports = {
            name: module
            for module in ("itertools", "math", "collections")
            for name in dir(__import__(module))
        }
        subimports["Path"] = "pathlib"
        subimports["pp"] = "pprint"
        subimports["pypprint"] = "pyp"

        def get_import_for_name(name: str) -> ast.stmt:
            if name in subimports:
                return ast.parse(f"from {subimports[name]} import {name}").body[0]
            return ast.parse(f"import {name}").body[0]

        self.before_tree.body = [
            get_import_for_name(name) for name in sorted(missing_names)
        ] + self.before_tree.body

    def build(self) -> ast.Module:
        """Returns a transformed AST."""
        self.build_output()
        self.build_input()
        self.build_missing_imports()

        ret = ast.parse("")
        ret.body = self.before_tree.body + self.tree.body + self.after_tree.body
        return ast.fix_missing_locations(ret)


def unparse(tree: ast.Module) -> str:
    """Returns a Python script equivalent to executing ``tree``."""
    if sys.version_info >= (3, 9):
        return ast.unparse(tree)
    try:
        import astunparse  # type: ignore

        return astunparse.unparse(tree)  # type: ignore
    except ImportError:
        pass

    return f"""
from ast import *
tree = fix_missing_locations({ast.dump(tree)})
# To see this in human readable form, run `pyp` with Python 3.9
# Alternatively, install a third party ast unparser: `python3 -m pip install astunparse`
# Once you've done that, simply re-run.
# In the meantime, this script is fully functional, if not easily readable or modifiable...
exec(compile(tree, filename="<ast>", mode="exec"), {{}})
"""


def run_pyp(args: argparse.Namespace) -> None:
    try:
        tree = PypTransform(args.before, args.code, args.after, args.define_pypprint).build()
    except PypError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)

    if args.explain:
        print(unparse(tree))
    else:
        exec(compile(tree, filename="<ast>", mode="exec"), {})


def parse_options(args: List[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="pyp",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=(
            "Easily run Python at the shell!\n\n"
            "For help and examples, see https://github.com/hauntsaninja/pyp\n\n"
            "Cheatsheet:\n"
            "- Use `line`, `x`, `l`, or `s` for a line in the input. Use `i`, `idx` or `index` "
            "for the index\n"
            "- Use `lines` to get a list of rstripped lines\n"
            "- Use `stdin` to get sys.stdin\n"
            "- Use print explicitly if you don't like when or how or what pyp's printing\n"
            "- If the magic is ever too mysterious, use --explain"
        ),
    )
    parser.add_argument("code", nargs="+", help="Python you want to run")
    parser.add_argument(
        "--explain",
        "--script",
        action="store_true",
        help="Prints the Python that would get run, instead of running it",
    )
    parser.add_argument(
        "-b",
        "--before",
        action="append",
        default=[],
        metavar="CODE",
        help="Python to run before processing input",
    )
    parser.add_argument(
        "-a",
        "--after",
        action="append",
        default=[],
        metavar="CODE",
        help="Python to run after processing input",
    )
    parser.add_argument(
        "--define-pypprint",
        action="store_true",
        help="Defines pypprint, if used, instead of importing it from pyp.",
    )
    return parser.parse_args(args)


def main() -> None:
    run_pyp(parse_options(sys.argv[1:]))


if __name__ == "__main__":
    main()
