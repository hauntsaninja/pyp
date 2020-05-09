import ast
import sys

import pytest
from pyp import find_names


def test_basic():
    assert (set(), set("x")) == find_names(ast.parse("x[:3]"))
    assert ({"x"}, set()) == find_names(ast.parse("x = 1"))
    assert ({"x", "y"}, set()) == find_names(ast.parse("x = 1; y = x + 1"))


def test_builtins():
    assert (set(), {"print"}) == find_names(ast.parse("print(5)"))
    assert ({"print"}, set()) == find_names(ast.parse("print = 5; print(5)"))


def test_loops():
    assert ({"x"}, {"y", "print"}) == find_names(ast.parse("for x in y: print(x)"))
    assert (set(), {"x"}) == find_names(ast.parse("while x: pass"))


def test_weird_assignments():
    assert ({"x"}, {"x"}) == find_names(ast.parse("x += 1"))
    assert ({"x"}, {"x"}) == find_names(ast.parse("for x in x: pass"))
    assert ({"x", "y"}, {"x", "y"}) == find_names(ast.parse("x, y = x, y"))
    if sys.version_info >= (3, 8):
        assert ({"x"}, {"x"}) == find_names(ast.parse("(x := x)"))


def test_comprehensions():
    assert ({"x"}, {"y"}) == find_names(ast.parse("(x for x in y)"))
    assert ({"x"}, {"x"}) == find_names(ast.parse("(x for x in x)"))
    assert ({"x", "xx"}, {"xxx"}) == find_names(ast.parse("(x for xx in xxx for x in xx)"))
    assert ({"x", "xx"}, {"xx", "xxx"}) == find_names(ast.parse("(x for x in xx for xx in xxx)"))
    assert ({"x"}, {"xx"}) == find_names(ast.parse("(x for x in xx if x == 'foo')"))


def test_args():
    assert ({"f"}, {"x"}) == find_names(ast.parse("f = lambda: x"))
    assert ({"f", "x"}, set()) == find_names(ast.parse("f = lambda x: x"))
    assert ({"f", "x"}, {"y"}) == find_names(ast.parse("f = lambda x: y"))
    assert ({"a", "b", "c", "x", "y", "z"}, set()) == find_names(
        ast.parse("def f(x, y = 0, *z, a, b = 0, **c): ...")
    )


@pytest.mark.xfail(reason="do not currently support scopes")
def test_args_bad():
    assert ({"f", "x"}, {"x"}) == find_names(ast.parse("f = lambda x: x; x"))


@pytest.mark.xfail(reason="do not currently support deletes")
def test_del():
    assert ({"x"}, {"x"}) == find_names(ast.parse("x = 3; del x; x"))
