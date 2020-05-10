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


def test_weird_assignments():
    check_find_names("x += 1", {"x"}, {"x"})
    check_find_names("for x in x: pass", {"x"}, {"x"})
    check_find_names("x, y = x, y", {"x", "y"}, {"x", "y"})


@pytest.mark.skipif(sys.version_info < (3, 8), reason="requires python 3.8 or later")
def test_walrus():
    check_find_names("(x := x)", {"x"}, {"x"})
    check_find_names("f((f := lambda x: x))", {"f", "x"}, {"f"})
    check_find_names("if (x := 1): print(x)", {"x"}, {"print"})
    check_find_names("(y for x in xx if (y := x) == 'foo')", {"x", "y"}, {"xx"})
    check_find_names("x: (x := 1) = 2", {"x"}, set())


def test_comprehensions():
    check_find_names("(x for x in y)", {"x"}, {"y"})
    check_find_names("(x for x in x)", {"x"}, {"x"})
    check_find_names("(x for xx in xxx for x in xx)", {"x", "xx"}, {"xxx"})
    check_find_names("(x for x in xx for xx in xxx)", {"x", "xx"}, {"xx", "xxx"})
    check_find_names("(x for x in xx if x == 'foo')", {"x"}, {"xx"})


def test_args():
    check_find_names("f = lambda: x", {"f"}, {"x"})
    check_find_names("f = lambda x: x", {"f", "x"}, set())
    check_find_names("f = lambda x: y", {"f", "x"}, {"y"})
    check_find_names(
        "def f(x, y = 0, *z, a, b = 0, **c): ...", {"a", "b", "c", "x", "y", "z"}, set()
    )


@pytest.mark.xfail(reason="do not currently support scopes")
def test_args_bad():
    check_find_names("f = lambda x: x; x", {"f", "x"}, {"x"})


@pytest.mark.xfail(reason="do not currently support deletes")
def test_del():
    check_find_names("x = 3; del x; x", {"x"}, {"x"})
