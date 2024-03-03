#!/usr/bin/env python3
import argparse
import ast
import importlib
import inspect
import itertools
import os
import sys
import textwrap
import traceback
from collections import defaultdict
from typing import Any, Dict, Iterator, List, Optional, Set, Tuple, cast

__all__ = ["pypprint"]
__version__ = "1.2.0"


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


class NameFinder(ast.NodeVisitor):
    """Finds undefined names, top-level defined names and wildcard imports in the given AST.

    A top-level defined name is any name that is stored to in the top-level scopes of ``trees``.
    An undefined name is any name that is loaded before it is defined (in any scope).

    Notes: a) we ignore deletes, b) used builtins will appear in undefined names, c) this logic
    doesn't fully support nonlocal / global / late-binding scopes.

    """

    def __init__(self, *trees: ast.AST) -> None:
        self._scopes: List[Set[str]] = [set()]
        self._comprehension_scopes: List[int] = []

        self.undefined: Set[str] = set()
        self.wildcard_imports: List[str] = []
        for tree in trees:
            self.visit(tree)
        assert len(self._scopes) == 1

    @property
    def top_level_defined(self) -> Set[str]:
        return self._scopes[0]

    def flexible_visit(self, value: Any) -> None:
        if isinstance(value, list):
            for item in value:
                if isinstance(item, ast.AST):
                    self.visit(item)
        elif isinstance(value, ast.AST):
            self.visit(value)

    def generic_visit(self, node: ast.AST) -> None:
        def order(f_v: Tuple[str, Any]) -> int:
            # This ordering fixes comprehensions, dict comps, loops, assignments
            return {"generators": -3, "iter": -3, "key": -2, "value": -1}.get(f_v[0], 0)

        # Adapted from ast.NodeVisitor.generic_visit, but re-orders traversal a little
        for _, value in sorted(ast.iter_fields(node), key=order):
            self.flexible_visit(value)

    def visit_Name(self, node: ast.Name) -> None:
        if isinstance(node.ctx, ast.Load):
            if all(node.id not in d for d in self._scopes):
                self.undefined.add(node.id)
        elif isinstance(node.ctx, ast.Store):
            self._scopes[-1].add(node.id)
        # Ignore deletes, see docstring
        self.generic_visit(node)

    def visit_Global(self, node: ast.Global) -> None:
        self._scopes[-1] |= self._scopes[0] & set(node.names)

    def visit_Nonlocal(self, node: ast.Nonlocal) -> None:
        if len(self._scopes) >= 2:
            self._scopes[-1] |= self._scopes[-2] & set(node.names)

    def visit_AugAssign(self, node: ast.AugAssign) -> None:
        if isinstance(node.target, ast.Name):
            # TODO: think about global, nonlocal
            if node.target.id not in self._scopes[-1]:
                self.undefined.add(node.target.id)
        self.generic_visit(node)

    def visit_NamedExpr(self, node: Any) -> None:
        self.visit(node.value)
        # PEP 572 has weird scoping rules
        assert isinstance(node.target, ast.Name)
        assert isinstance(node.target.ctx, ast.Store)
        scope_index = len(self._scopes) - 1
        comp_index = len(self._comprehension_scopes) - 1
        while comp_index >= 0 and scope_index == self._comprehension_scopes[comp_index]:
            scope_index -= 1
            comp_index -= 1
        self._scopes[scope_index].add(node.target.id)

    def visit_alias(self, node: ast.alias) -> None:
        if node.name != "*":
            self._scopes[-1].add(
                node.asname if node.asname is not None else node.name.split(".")[0]
            )

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        if node.module is not None and "*" in (a.name for a in node.names):
            self.wildcard_imports.append(node.module)
        self.generic_visit(node)

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self.flexible_visit(node.decorator_list)
        self.flexible_visit(node.bases)
        self.flexible_visit(node.keywords)

        self._scopes.append(set())
        self.flexible_visit(node.body)
        self._scopes.pop()
        # Classes are not okay with self-reference, so define ``name`` afterwards
        self._scopes[-1].add(node.name)

    def visit_function_helper(self, node: Any, name: Optional[str] = None) -> None:
        # Functions are okay with recursion, but not self-reference while defining default values
        self.flexible_visit(node.args)
        if name is not None:
            self._scopes[-1].add(name)

        self._scopes.append(set())
        for arg_node in ast.iter_child_nodes(node.args):
            if isinstance(arg_node, ast.arg):
                self._scopes[-1].add(arg_node.arg)
        self.flexible_visit(node.body)
        self._scopes.pop()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self.flexible_visit(node.decorator_list)
        self.visit_function_helper(node, node.name)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self.flexible_visit(node.decorator_list)
        self.visit_function_helper(node, node.name)

    def visit_Lambda(self, node: ast.Lambda) -> None:
        self.visit_function_helper(node)

    def visit_ExceptHandler(self, node: ast.ExceptHandler) -> None:
        # ExceptHandler's name is scoped to the handler. If name exists and the name is not already
        # defined, we'll define then undefine it to mimic the scope.
        if not node.name or node.name in self._scopes[-1]:
            self.generic_visit(node)
            return

        self.flexible_visit(node.type)
        assert node.name is not None
        self._scopes[-1].add(node.name)
        self.flexible_visit(node.body)
        self._scopes[-1].remove(node.name)

    def visit_comprehension_helper(self, node: Any) -> None:
        self._comprehension_scopes.append(len(self._scopes))
        self._scopes.append(set())
        self.generic_visit(node)
        self._scopes.pop()
        self._comprehension_scopes.pop()

    visit_ListComp = visit_comprehension_helper
    visit_SetComp = visit_comprehension_helper
    visit_GeneratorExp = visit_comprehension_helper
    visit_DictComp = visit_comprehension_helper


def dfs_walk(node: ast.AST) -> Iterator[ast.AST]:
    """Helper to iterate over an AST depth-first."""
    stack = [node]
    while stack:
        node = stack.pop()
        stack.extend(reversed(list(ast.iter_child_nodes(node))))
        yield node


MAGIC_VARS = {
    "index": {"i", "idx", "index"},
    "loop": {"line", "x", "l"},
    "input": {"lines", "stdin"},
}


def is_magic_var(name: str) -> bool:
    return any(name in vars for vars in MAGIC_VARS.values())


class PypError(Exception):
    pass


def get_config_contents() -> str:
    """Returns the empty string if no config file is specified."""
    config_file = os.environ.get("PYP_CONFIG_PATH")
    if config_file is None:
        return ""
    try:
        with open(config_file, "r") as f:
            return f.read()
    except FileNotFoundError as e:
        raise PypError(f"Config file not found at PYP_CONFIG_PATH={config_file}") from e


class PypConfig:
    """PypConfig is responsible for handling user configuration.

    We allow users to configure pyp with a config file that is very Python-like. Rather than
    executing the config file as Python unconditionally, we treat it as a source of definitions. We
    keep track of what each top-level stmt in the AST of the config file defines, and if we need
    that definition in our program, use it. A wrinkle here is that definitions in the config file
    may depend on other definitions within the config file; this is handled by build_missing_config.
    Another wrinkle is wildcard imports; these are kept track of and added to the list of special
    cased wildcard imports in build_missing_imports.

    """

    def __init__(self) -> None:
        config_contents = get_config_contents()
        try:
            config_ast = ast.parse(config_contents)
        except SyntaxError as e:
            error = f": {e.text!r}" if e.text else ""
            raise PypError(f"Config has invalid syntax{error}") from e

        # List of config parts
        self.parts: List[ast.stmt] = config_ast.body
        # Maps from a name to index of config part that defines it
        self.name_to_def: Dict[str, int] = {}
        self.def_to_names: Dict[int, List[str]] = defaultdict(list)
        # Maps from index of config part to undefined names it needs
        self.requires: Dict[int, Set[str]] = defaultdict(set)
        # Modules from which automatic imports work without qualification, ordered by AST encounter
        self.wildcard_imports: List[str] = []

        self.shebang: str = "#!/usr/bin/env python3"
        if config_contents.startswith("#!"):
            self.shebang = "\n".join(
                itertools.takewhile(lambda line: line.startswith("#"), config_contents.splitlines())
            )

        top_level: Tuple[Any, ...] = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
        top_level += (ast.Import, ast.ImportFrom, ast.Assign, ast.AnnAssign, ast.If, ast.Try)
        for index, part in enumerate(self.parts):
            if not isinstance(part, top_level):
                node_type = type(
                    part.value if isinstance(part, ast.Expr) else part
                ).__name__.lower()
                raise PypError(
                    "Config only supports a subset of Python at top level; "
                    f"unsupported construct ({node_type}) on line {part.lineno}"
                )
            f = NameFinder(part)
            for name in f.top_level_defined:
                if self.name_to_def.get(name, index) != index:
                    raise PypError(f"Config has multiple definitions of {repr(name)}")
                if is_magic_var(name):
                    raise PypError(f"Config cannot redefine built-in magic variable {repr(name)}")
                self.name_to_def[name] = index
                self.def_to_names[index].append(name)
            self.requires[index] = f.undefined
            self.wildcard_imports.extend(f.wildcard_imports)


class PypTransform:
    """PypTransform is responsible for transforming all input code.

    A lot of pyp's magic comes from it making decisions based on defined and undefined names in the
    input. This class helps keep track of that state as things change based on transformations. In
    general, the logic in here is very sensitive to reordering; there are various implicit
    assumptions about what transformations have happened and what names have been defined. But
    the code is pretty small and the tests are good, so you should be okay!

    """

    def __init__(
        self,
        before: List[str],
        code: List[str],
        after: List[str],
        define_pypprint: bool,
        config: PypConfig,
    ) -> None:
        def parse_input(code: List[str]) -> ast.Module:
            try:
                return ast.parse(textwrap.dedent("\n".join(code).strip()))
            except SyntaxError as e:
                message = traceback.format_exception_only(type(e), e)
                message[0] = "Invalid input\n\n"
                raise PypError("".join(message).strip()) from e

        self.before_tree = parse_input(before)
        self.tree = parse_input(code)
        self.after_tree = parse_input(after)

        f = NameFinder(self.before_tree, self.tree, self.after_tree)
        self.defined: Set[str] = f.top_level_defined
        self.undefined: Set[str] = f.undefined
        self.wildcard_imports: List[str] = f.wildcard_imports
        # We'll always use sys in ``build_input``, so add it to undefined.
        # This lets config define it or lets us automatically import it later
        # (If before defines it, we'll just let it override the import...)
        self.undefined.add("sys")

        self.define_pypprint = define_pypprint
        self.config = config

        # The print statement ``build_output`` will add, if it determines it needs to.
        self.implicit_print: Optional[ast.Call] = None

    def build_missing_config(self) -> None:
        """Modifies the AST to define undefined names defined in config."""
        config_definitions: Set[str] = set()
        attempt_to_define = set(self.undefined)
        while attempt_to_define:
            can_define = attempt_to_define & set(self.config.name_to_def)
            # The things we can define might in turn require some definitions, so update the things
            # we need to attempt to define and loop
            attempt_to_define = set()
            for name in can_define:
                config_definitions.update(self.config.def_to_names[self.config.name_to_def[name]])
                attempt_to_define.update(self.config.requires[self.config.name_to_def[name]])
            # We don't need to attempt to define things we've already decided we need to define
            attempt_to_define -= config_definitions

        config_indices = {self.config.name_to_def[name] for name in config_definitions}

        # Run basically the same thing in reverse to see which dependencies stem from magic vars
        before_config_indices = set(config_indices)
        derived_magic_indices = {
            i for i in config_indices if any(map(is_magic_var, self.config.requires[i]))
        }
        derived_magic_names = set()

        while derived_magic_indices:
            before_config_indices -= derived_magic_indices
            derived_magic_names |= {
                name for i in derived_magic_indices for name in self.config.def_to_names[i]
            }
            derived_magic_indices = {
                i for i in before_config_indices if self.config.requires[i] & derived_magic_names
            }
        magic_config_indices = config_indices - before_config_indices

        before_config_defs = [self.config.parts[i] for i in sorted(before_config_indices)]
        magic_config_defs = [self.config.parts[i] for i in sorted(magic_config_indices)]

        self.before_tree.body = before_config_defs + self.before_tree.body
        self.tree.body = magic_config_defs + self.tree.body

        for i in config_indices:
            self.undefined.update(self.config.requires[i])
        self.defined |= config_definitions
        self.undefined -= config_definitions

    def define(self, name: str) -> None:
        """Defines a name."""
        self.defined.add(name)
        self.undefined.discard(name)

    def get_valid_name_in_top_scope(self, name: str) -> str:
        """Return a name related to ``name`` that does not conflict with existing definitions."""
        while name in self.defined or name in self.undefined:
            name += "_"
        return name

    def build_output(self) -> None:
        """Ensures that the AST prints something.

        This is done by either a) checking whether we load a thing that prints, or b) if the last
        thing in the tree is an expression, modifying the tree to print it.

        """
        if self.undefined & {"print", "pprint", "pp", "pypprint"}:  # has an explicit print
            return

        def inner(body: List[ast.stmt], use_pypprint: bool = False) -> bool:
            if not body:
                return False
            if isinstance(body[-1], ast.Pass):
                del body[-1]
                return True
            if not isinstance(body[-1], ast.Expr):
                if (
                    # If the last thing in the tree is a statement that has a body
                    hasattr(body[-1], "body")
                    # and doesn't have an orelse, since users could expect the print in that branch
                    and not getattr(body[-1], "orelse", [])
                    # and doesn't enter a new scope
                    and not isinstance(
                        body[-1], (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)
                    )
                ):
                    # ...then recursively look for a standalone expression
                    return inner(body[-1].body, use_pypprint)
                return False

            if isinstance(body[-1].value, ast.Name):
                output = body[-1].value.id
                body.pop()
            else:
                output = self.get_valid_name_in_top_scope("output")
                self.define(output)
                body[-1] = ast.Assign(
                    targets=[ast.Name(id=output, ctx=ast.Store())], value=body[-1].value
                )

            print_fn = "print"
            if use_pypprint:
                print_fn = "pypprint"
                self.undefined.add("pypprint")

            if_print = ast.parse(f"if {output} is not None: {print_fn}({output})").body[0]
            body.append(if_print)

            self.implicit_print = if_print.body[0].value  # type: ignore
            return True

        # First attempt to add a print to self.after_tree, then to self.tree
        # We use pypprint in self.after_tree and print in self.tree, although the latter is
        # subject to change later on if we call ``use_pypprint_for_implicit_print``. This logic
        # could be a little simpler if we refactored so that we know what transformations we will
        # do before we do them.
        success = inner(self.after_tree.body, True) or inner(self.tree.body)
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
        possible_vars = {typ: names & self.undefined for typ, names in MAGIC_VARS.items()}

        if (possible_vars["loop"] or possible_vars["index"]) and possible_vars["input"]:
            loop_names = ", ".join(possible_vars["loop"] or possible_vars["index"])
            input_names = ", ".join(possible_vars["input"])
            raise PypError(
                f"Candidates found for both loop variable ({loop_names}) and "
                f"input variable ({input_names})"
            )

        for typ, names in possible_vars.items():
            if len(names) > 1:
                names_str = ", ".join(names)
                raise PypError(f"Multiple candidates for {typ} variable: {names_str}")

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
        elif possible_vars["input"]:
            # We'll read from stdin and define the necessary input variable
            input_var = possible_vars["input"].pop()
            self.define(input_var)

            if input_var == "stdin":
                input_assign = ast.parse(f"{input_var} = sys.stdin")
            else:
                input_assign = ast.parse(f"{input_var} = [x.rstrip('\\n') for x in sys.stdin]")

            self.tree.body = input_assign.body + self.tree.body
            self.use_pypprint_for_implicit_print()
        else:
            no_pipe_assertion = ast.parse(
                "assert sys.stdin.isatty() or not sys.stdin.read(), "
                """"The command doesn't process input, but input is present. """
                '''Maybe you meant to use a magic variable like `stdin` or `x`?"'''
            )
            self.tree.body = no_pipe_assertion.body + self.tree.body
            self.use_pypprint_for_implicit_print()

    def build_missing_imports(self) -> None:
        """Modifies the AST to import undefined names."""
        self.undefined -= set(dir(__import__("builtins")))

        # Optimisation: we will almost always define sys and pypprint. However, in order for us to
        # get to `import sys`, we'll need to examine our wildcard imports, which in the presence
        # of config, could be slow.
        if "pypprint" in self.undefined:
            pypprint_def = (
                inspect.getsource(pypprint) if self.define_pypprint else "from pyp import pypprint"
            )
            self.before_tree.body = ast.parse(pypprint_def).body + self.before_tree.body
            self.undefined.remove("pypprint")
        if "sys" in self.undefined:
            self.before_tree.body = ast.parse("import sys").body + self.before_tree.body
            self.undefined.remove("sys")
        # Now short circuit if we can
        if not self.undefined:
            return

        def get_names_in_module(module: str) -> Any:
            try:
                mod = importlib.import_module(module)
            except ImportError as e:
                raise PypError(
                    f"Config contains wildcard import from {module}, but {module} failed to import"
                ) from e
            return getattr(mod, "__all__", (n for n in dir(mod) if not n.startswith("_")))

        subimports = {"Path": "pathlib", "pp": "pprint"}
        wildcard_imports = (
            ["itertools", "math", "collections"]
            + self.config.wildcard_imports
            + self.wildcard_imports
        )
        subimports.update(
            {name: module for module in wildcard_imports for name in get_names_in_module(module)}
        )

        def get_import_for_name(name: str) -> str:
            if name in subimports:
                return f"from {subimports[name]} import {name}"
            return f"import {name}"

        self.before_tree.body = [
            ast.parse(stmt).body[0] for stmt in sorted(map(get_import_for_name, self.undefined))
        ] + self.before_tree.body

    def build(self) -> ast.Module:
        """Returns a transformed AST."""
        self.build_missing_config()
        self.build_output()
        self.build_input()
        self.build_missing_imports()

        ret = ast.parse("")
        ret.body = self.before_tree.body + self.tree.body + self.after_tree.body
        # Add fake line numbers to the nodes, so we can generate a traceback on error
        i = 0
        for node in dfs_walk(ret):
            if isinstance(node, ast.stmt):
                i += 1
            node.lineno = i
            node.end_lineno = i

        return ast.fix_missing_locations(ret)


def unparse(tree: ast.AST, short_fallback: bool = False) -> str:
    """Returns Python code equivalent to executing ``tree``."""
    if sys.version_info >= (3, 9):
        return ast.unparse(tree)
    try:
        import astunparse  # type: ignore

        return cast(str, astunparse.unparse(tree))
    except ImportError:
        pass
    return (
        fallback_unparse(tree)
        if not short_fallback
        else f"# {ast.dump(tree)}  # --explain has instructions to make this readable"
    )


def fallback_unparse(tree: ast.AST) -> str:
    return f"""
from ast import *
tree = fix_missing_locations({ast.dump(tree)})

# To see this in human readable form, run pyp with Python 3.9
# Alternatively, install a third party ast unparser: `python3 -m pip install astunparse`
# Once you've done that, simply re-run.
# In the meantime, this script is fully functional, if not easily readable or modifiable...

exec(compile(tree, filename="<ast>", mode="exec"), {{}})
"""


def run_pyp(args: argparse.Namespace) -> None:
    config = PypConfig()
    tree = PypTransform(args.before, args.code, args.after, args.define_pypprint, config).build()
    if args.explain:
        print(config.shebang)
        print(unparse(tree))
        return
    try:
        exec(compile(tree, filename="<pyp>", mode="exec"), {})
    except Exception as e:
        # On error, reconstruct a traceback into the generated code
        # Also add some diagnostics for ModuleNotFoundError and NameError
        try:
            line_to_node: Dict[int, ast.AST] = {}
            for node in dfs_walk(tree):
                line_to_node.setdefault(getattr(node, "lineno", -1), node)

            def code_for_line(lineno: int) -> str:
                node = line_to_node[lineno]
                # Don't unparse nested child statements. Note this destroys the tree.
                for _, value in ast.iter_fields(node):
                    if isinstance(value, list) and value and isinstance(value[0], ast.stmt):
                        value.clear()
                return unparse(node, short_fallback=True).strip()

            # Time to commit several sins against CPython implementation details
            tb_except = traceback.TracebackException(
                type(e), e, e.__traceback__.tb_next  # type: ignore
            )
            for fs in tb_except.stack:
                if fs.filename == "<pyp>":
                    if fs.lineno is None:
                        raise AssertionError("When would this happen?")
                    fs._line = code_for_line(fs.lineno)  # type: ignore[attr-defined]
                    fs.lineno = "PYP_REDACTED"  # type: ignore[assignment]

            tb_format = tb_except.format()
            assert "Traceback (most recent call last)" in next(tb_format)

            message = "Possible reconstructed traceback (most recent call last):\n"
            message += "".join(tb_format).strip("\n")
            message = message.replace(", line PYP_REDACTED", "")
        except Exception:
            message = "".join(traceback.format_exception_only(type(e), e)).strip()
        if isinstance(e, ModuleNotFoundError):
            message += (
                "\n\nNote pyp treats undefined names as modules to automatically import. "
                "Perhaps you forgot to define something or PYP_CONFIG_PATH is set incorrectly?"
            )
        if args.before and isinstance(e, NameError):
            var = str(e)
            var = var[var.find("'") + 1 : var.rfind("'")]
            if var in ("lines", "stdin"):
                message += (
                    "\n\nNote code in `--before` runs before any magic variables are defined "
                    "and should not process input. Your command should work by simply removing "
                    "`--before`, so instead passing in multiple statements in the main section "
                    "of your code."
                )
        raise PypError(
            "Code raised the following exception, consider using --explain to investigate:\n\n"
            f"{message}"
        ) from e


def parse_options(args: List[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="pyp",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=(
            "Easily run Python at the shell!\n\n"
            "For help and examples, see https://github.com/hauntsaninja/pyp\n\n"
            "Cheatsheet:\n"
            "- Use `x`, `l` or `line` for a line in the input. Use `i`, `idx` or `index` "
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
    parser.add_argument("--version", action="version", version=f"pyp {__version__}")
    return parser.parse_args(args)


def main() -> None:
    try:
        run_pyp(parse_options(sys.argv[1:]))
    except PypError as e:
        print(f"error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
