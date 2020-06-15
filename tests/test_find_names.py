import ast
import re
import sys
from typing import List, Optional, Set

import pytest
from pyp import NameFinder


def check_find_names(
    code: str,
    defined: Set[str],
    undefined: Set[str],
    wildcard_imports: Optional[List[str]] = None,
    confirm: bool = True,
) -> None:
    names = NameFinder(ast.parse(code))
    assert defined == names.top_level_defined
    assert undefined == names.undefined
    assert (wildcard_imports or []) == names.wildcard_imports

    if not confirm:
        return

    exec_locals = {}
    actually_undefined = undefined - set(dir(__import__("builtins")))
    if actually_undefined:
        # If something is actually undefined, we should raise a NameError when we execute
        # (if we hit another exception first, we fix the test!)
        with pytest.raises(NameError) as e:
            exec(code, exec_locals)
        assert re.search(r"name '(\w+)' is not", e.value.args[0]).group(1) in actually_undefined
    else:
        try:
            exec(code, exec_locals)
        except Exception as e:
            # Unlike above, allow this code to fail, but if it fails, it shouldn't be a NameError!
            assert not isinstance(e, NameError)

    exec_locals = set(exec_locals)
    exec_locals -= {"__builtins__", "__annotations__"}
    # In general, we over define things, because we don't deal with scopes and such. So just check
    # a subset relationship holds, we could tighten this check in the future.
    assert exec_locals <= defined


def test_basic():
    check_find_names("x[:3]", set(), {"x"})
    check_find_names("x = 1", {"x"}, set())
    check_find_names("x = 1; y = x + 1", {"x", "y"}, set())


def test_builtins():
    check_find_names("print(5)", set(), {"print"})
    check_find_names("print = 5; print(5)", {"print"}, set())


def test_loops():
    check_find_names("for x in y: print(x)", {"x"}, {"y", "print"})
    check_find_names("while x: pass", set(), {"x"})
    check_find_names("for x in x: pass", {"x"}, {"x"})
    check_find_names("for x in xx: y = 1\nelse: y", {"x", "y"}, {"xx"})
    check_find_names("for x in xx: pass\nelse: y", {"x"}, {"xx", "y"})


def test_weird_assignments():
    check_find_names("x += 1", {"x"}, {"x"})
    check_find_names("for x in x: pass", {"x"}, {"x"})
    check_find_names("x, y = x, y", {"x", "y"}, {"x", "y"})
    check_find_names("x: int = 1", {"x"}, {"int"})
    check_find_names("x: int = x", {"x"}, {"int", "x"})
    check_find_names("x.y = 5", set(), {"x"})


def test_more_control_flow():
    check_find_names("try: ...\nexcept: ...", set(), set())
    check_find_names("try: raise\nexcept Exception as e: print(e)", set(), {"Exception", "print"})
    check_find_names("try: raise\nexcept e as e: print(e)", set(), {"e", "print"})
    check_find_names("try: x = 1\nexcept: pass", {"x"}, set())

    # with and without confirm for readability, since we need to raise to hit branches
    check_find_names("try: x = 1\nexcept: x", {"x"}, set(), confirm=False)
    check_find_names("try:\n x = 1\n raise\nexcept: x", {"x"}, set())
    check_find_names("try: x = 1\nexcept: y", {"x"}, {"y"}, confirm=False)
    check_find_names("try:\n x = 1\n raise\nexcept: y", {"x"}, {"y"})
    check_find_names("try: x = 1\nexcept: y\nfinally: x", {"x"}, {"y"}, confirm=False)
    check_find_names("try:\n x = 1\n raise\nexcept: y\nfinally: x", {"x"}, {"y"})

    check_find_names("try: x = 1\nexcept: y\nfinally: z", {"x"}, {"y", "z"})
    check_find_names("try: x\nexcept ValueError as e: z = e", {"z"}, {"ValueError", "x"})
    check_find_names(
        "try: x\nexcept Exception as e: z = e\nfinally: y", {"z"}, {"Exception", "x", "y"}
    )

    check_find_names("try: raise\nexcept Exception as e: ...\ne", set(), {"Exception", "e"})
    check_find_names("try: raise\nexcept Exception as e: f = e\nf", {"f"}, {"Exception"})

    check_find_names("with a as x: x", {"x"}, {"a"})
    check_find_names("with a as x, b as y: x", {"x", "y"}, {"a", "b"})
    check_find_names("with a as b, b as c: c", {"b", "c"}, {"a"})
    check_find_names("with a as c, b as c: c", {"c"}, {"a", "b"})
    check_find_names("with a as a: a", {"a"}, {"a"})

    check_find_names("if False: x = 5\nelse: y = 5", {"x", "y"}, set())
    # the following isn't ideal; we don't look at branches in parallel
    check_find_names("if False: x = 5\nelse: y = x", {"x", "y"}, set(), confirm=False)


def test_import():
    check_find_names("import x", {"x"}, set())
    check_find_names("import y as x", {"x"}, set())
    check_find_names("from y import x", {"x"}, set())
    check_find_names("from y import z as x", {"x"}, set())
    check_find_names("from x import *", set(), set(), wildcard_imports=["x"])


@pytest.mark.skipif(sys.version_info < (3, 8), reason="requires python 3.8 or later")
def test_walrus():
    check_find_names("(x := 1)", {"x"}, set())
    check_find_names("x = (x := 1)", {"x"}, set())
    check_find_names("(x := x)", {"x"}, {"x"})
    check_find_names("x += (x := 1)", {"x"}, {"x"})
    check_find_names("f((f := lambda x: x))", {"f"}, {"f"})
    check_find_names("f((f := lambda x: (x, y)))", {"f"}, {"f", "y"})
    check_find_names("if (x := 1): print(x)", {"x"}, {"print"})
    check_find_names("(y for x in xx if (y := x) == 'foo')", {"x", "y"}, {"xx"})
    check_find_names("x: (x := 1) = 2", {"x"}, set())
    check_find_names("f'{(x := 1)} {x}'", {"x"}, set())
    check_find_names("class A((A := object)): ...", {"A"}, {"object"})
    if sys.version_info >= (3, 9):
        check_find_names(
            "d1 = lambda i: i\n@(d2 := d1)\n@(d3 := d2)\ndef f(): ...",
            {"d1", "d2", "d3", "f"},
            set(),
        )
        check_find_names(
            "d1 = id\n@(d3 := d2)\n@(d2 := d1)\ndef f(): ...", {"d1", "d2", "d3", "f"}, {"d2", "id"}
        )
        check_find_names(
            "d1 = lambda i: i\n@(d2 := d1)\ndef f(x=d2): (x, d2)", {"d1", "d2", "f"}, set()
        )


def test_comprehensions():
    check_find_names("(x for x in y)", {"x"}, {"y"})
    check_find_names("(x for x in x)", {"x"}, {"x"})
    check_find_names("(x for xx in xxx for x in xx)", {"x", "xx"}, {"xxx"})
    check_find_names("(x for x in xx for xx in xxx)", {"x", "xx"}, {"xx", "xxx"})
    check_find_names("(x for x in xx if x > 0)", {"x"}, {"xx"})
    check_find_names("[x for x in xx if x > 0]", {"x"}, {"xx"})
    check_find_names("{x for x in xx if x > 0}", {"x"}, {"xx"})
    check_find_names("{x: x for x in xx if x > 0}", {"x"}, {"xx"})


def test_args():
    check_find_names("f = lambda: x\nf()", {"f"}, {"x"})
    check_find_names("f = lambda x: x\nf()", {"f"}, set())
    check_find_names("f = lambda x: y\nf(1)", {"f"}, {"y"})
    check_find_names(
        "def f(x, y = 0, *z, a, b = 0, **c): (x, y, z, a, b, c, D)\nf(1, a=2)", {"f"}, {"D"}
    )
    check_find_names(
        "async def f(x, y = 0, *z, a, b = 0, **c): (x, y, z, a, b, c, D)",
        {"f"},
        {"D"},
        confirm=False,
    )

    check_find_names("f = lambda x=(lambda y: y): x\nf(1)", {"f"}, set())
    check_find_names("f = lambda x=(lambda y: y): y\nf(1)", {"f"}, {"y"})
    check_find_names("def f(x=(lambda y: y)): x\nf(1)", {"f"}, set())
    check_find_names("def f(x=(lambda y: y)): y\nf(1)", {"f"}, {"y"})

    check_find_names("f = lambda x: x; x", {"f"}, {"x"})
    check_find_names(
        "def f(x, y = 0, *z, a, b = 0, **c): ...\n(x, y, z, a, b, c, D)",
        {"f"},
        {"x", "y", "z", "a", "b", "c", "D"},
    )


def test_definitions():
    check_find_names("def f(): ...", {"f"}, set())
    check_find_names("def f(): x\nf()", {"f"}, {"x"})
    check_find_names("async def f(): ...", {"f"}, set())
    check_find_names("def f(): f()", {"f"}, set())
    check_find_names("def f(): g()\nf()", {"f"}, {"g"})
    check_find_names("def f(x=g): x", {"f"}, {"g"})
    check_find_names("def f(x=g): y", {"f"}, {"g", "y"})
    check_find_names("def f(x=f): x", {"f"}, {"f"})
    check_find_names("@f\ndef f(): ...", {"f"}, {"f"})
    check_find_names("@f\ndef g(): f = 1", {"g"}, {"f"})
    check_find_names("class A: ...", {"A"}, set())
    check_find_names("class A(B): ...", {"A"}, {"B"})
    check_find_names("class A(A): ...", {"A"}, {"A"})
    check_find_names("class A(A): A = 1", {"A"}, {"A"})
    check_find_names("class A: A", {"A"}, {"A"})
    check_find_names("@A\nclass A: ...", {"A"}, {"A"})
    check_find_names("@A\nclass B: A = 1", {"B"}, {"A"})


@pytest.mark.xfail(reason="do not currently support deletes")
def test_del():
    check_find_names("x = 3; del x; x", {"x"}, {"x"})
