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


@pytest.mark.xfail(reason="do not currently support deletes")
def test_del():
    assert ({"x"}, {"x"}) == find_names(ast.parse("x = 3; del x; x"))
