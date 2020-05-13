import ast
import sys
from typing import Set

import pytest
from pyp import find_names


def check_find_names(code: str, defined: Set[str], undefined: Set[str]) -> None:
    assert (defined, undefined) == find_names(ast.parse(code))


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


def test_more_control_flow():
    check_find_names("try: x = 1\nexcept: pass", {"x"}, set())
    check_find_names("try: x = 1\nexcept: x", {"x"}, set())
    check_find_names("try: x = 1\nexcept: y", {"x"}, {"y"})
    check_find_names("try: x = 1\nexcept: y\nfinally: x", {"x"}, {"y"})
    check_find_names("try: x = 1\nexcept: y\nfinally: z", {"x"}, {"y", "z"})
    check_find_names(
        "try: x\nexcept Exception as e: z = e\nfinally: y", {"e", "z"}, {"Exception", "x", "y"}
    )
    check_find_names("try: ...\nexcept e as e: ...", {"e"}, {"e"})
    check_find_names("with a as x: x", {"x"}, {"a"})
    check_find_names("with a as x, b as y: x", {"x", "y"}, {"a", "b"})
    check_find_names("with a as b, b as c: c", {"b", "c"}, {"a"})
    check_find_names("with a as c, b as c: c", {"c"}, {"a", "b"})
    check_find_names("with a as a: a", {"a"}, {"a"})


def test_import():
    check_find_names("import x", {"x"}, set())
    check_find_names("import y as x", {"x"}, set())
    check_find_names("from y import x", {"x"}, set())
    check_find_names("from y import z as x", {"x"}, set())
    check_find_names("from x import *", {"*"}, set())


@pytest.mark.skipif(sys.version_info < (3, 8), reason="requires python 3.8 or later")
def test_walrus():
    check_find_names("(x := x)", {"x"}, {"x"})
    check_find_names("f((f := lambda x: x))", {"f", "x"}, {"f"})
    check_find_names("if (x := 1): print(x)", {"x"}, {"print"})
    check_find_names("(y for x in xx if (y := x) == 'foo')", {"x", "y"}, {"xx"})
    check_find_names("x: (x := 1) = 2", {"x"}, set())
    check_find_names("f'{(x := 1)} {x}'", {"x"}, set())
    check_find_names("class A((A := object)): ...", {"A"}, {"object"})


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
    check_find_names("f = lambda: x", {"f"}, {"x"})
    check_find_names("f = lambda x: x", {"f", "x"}, set())
    check_find_names("f = lambda x: y", {"f", "x"}, {"y"})
    check_find_names(
        "def f(x, y = 0, *z, a, b = 0, **c): ...", {"f", "a", "b", "c", "x", "y", "z"}, set()
    )
    check_find_names(
        "async def f(x, y = 0, *z, a, b = 0, **c): ...", {"f", "a", "b", "c", "x", "y", "z"}, set()
    )


def test_definitions():
    check_find_names("def f(): ...", {"f"}, set())
    check_find_names("def f(): x", {"f"}, {"x"})
    check_find_names("async def f(): ...", {"f"}, set())
    check_find_names("def f(): f()", {"f"}, set())
    check_find_names("def f(): g()", {"f"}, {"g"})
    check_find_names("def f(x=g): ...", {"f", "x"}, {"g"})
    check_find_names("def f(x=f): ...", {"f", "x"}, set())
    check_find_names("@f\ndef f(): ...", {"f"}, {"f"})
    check_find_names("@f\ndef g(): f = 1", {"f", "g"}, {"f"})
    check_find_names("class A: ...", {"A"}, set())
    check_find_names("class A(B): ...", {"A"}, {"B"})
    check_find_names("class A(A): ...", {"A"}, {"A"})
    check_find_names("class A(A): A = 1", {"A"}, {"A"})
    check_find_names("class A: A", {"A"}, {"A"})
    check_find_names("@A\nclass A: ...", {"A"}, {"A"})
    check_find_names("@A\nclass B: A = 1", {"A", "B"}, {"A"})


@pytest.mark.xfail(reason="do not currently support scopes")
def test_args_bad():
    check_find_names("f = lambda x: x; x", {"f", "x"}, {"x"})


@pytest.mark.xfail(reason="do not currently support deletes")
def test_del():
    check_find_names("x = 3; del x; x", {"x"}, {"x"})
