import ast
import contextlib
import io
import shlex
import subprocess
import sys
from typing import List, Optional, Union

import pyp
import pytest


def run_cmd(cmd: str, input: Optional[str] = None) -> bytes:
    if isinstance(input, str):
        input = input.encode("utf-8")
    return subprocess.check_output(cmd, shell=True, input=input)


def compare_command(
    example_cmd: str, pyp_cmd: str, input: Optional[str] = None, allow_example_fail: bool = False
) -> None:
    try:
        example_output = run_cmd(example_cmd, input)
    except subprocess.CalledProcessError:
        if allow_example_fail:
            return
        raise RuntimeError("Example command failed!")

    assert example_output == run_cmd(pyp_cmd, input)


def test_examples():
    """Test approximately the examples in the README."""

    example = "python\nsnake\nss\nmagpy\npiethon\nreadme.md\n"

    compare_command(example_cmd="cut -c1-4", pyp_cmd="pyp 'x[:4]'", input=example)
    compare_command(
        example_cmd="wc -c | tr -d ' '", pyp_cmd="pyp 'len(stdin.read())'", input=example
    )
    compare_command(
        example_cmd="awk '{s+=$1} END {print s}' ",
        pyp_cmd="pyp 'sum(map(int, lines))'",
        input="1\n2\n3\n4\n5\n",
    )
    compare_command(
        example_cmd="sh",
        pyp_cmd="pyp 'subprocess.run(lines[0], shell=True); pass'",
        input="echo echo",
    )
    compare_command(
        example_cmd="""jq -r '.[1]["lol"]'""",
        pyp_cmd="""pyp 'json.load(stdin)[1]["lol"]'""",
        input='[0, {"lol": "hmm"}, 0]',
    )
    compare_command(
        example_cmd="grep -E '(py|md)'",
        pyp_cmd="""pyp 'x if re.search("(py|md)", x) else None'""",
        input=example,
    )
    compare_command(example_cmd="echo 'sqrt(9.0)' | bc", pyp_cmd="pyp 'sqrt(9)'")
    compare_command(
        example_cmd="x=README.md; echo ${x##*.}",
        pyp_cmd='''x=README.md; pyp "Path('$x').suffix[1:]"''',
    )
    compare_command(
        example_cmd="echo '  1 a\n  2 b'", pyp_cmd="""pyp 'f"{idx+1: >3} {x}"'""", input="a\nb"
    )
    compare_command(
        example_cmd="grep py", pyp_cmd="""pyp 'x if "py" in x else None'""", input=example
    )
    compare_command(example_cmd="tail -n 3", pyp_cmd="pyp 'lines[-3:]'", input=example)
    compare_command(example_cmd="sort", pyp_cmd="pyp 'sorted(lines)'", input=example)
    compare_command(
        example_cmd="echo 'Sorting 2 lines\n1\n2'",
        pyp_cmd="""pyp 'print(f"Sorting {len(lines)} lines"); pypprint(sorted(lines))'""",
        input="2\n1\n",
    )
    compare_command(
        example_cmd="sort | uniq", pyp_cmd="pyp 'sorted(set(lines))'", input="2\n1\n1\n1\n3"
    )
    compare_command(
        example_cmd='''echo "a: ['1', '3']\nb: ['2']"''',
        pyp_cmd="pyp -b 'd = defaultdict(list)' 'k, v = x.split(); d[k].append(v)' -a 'd'",
        input="a 1\nb 2\na 3",
    )


def run_pyp(cmd: Union[str, List[str]], input: str = "") -> str:
    """Run pyp in process. It's quicker and allows us to mock and so on."""
    if isinstance(cmd, str):
        cmd = shlex.split(cmd)
    if cmd[0] == "pyp":
        del cmd[0]

    output = io.StringIO()
    with contextlib.redirect_stdout(output):
        try:
            old_stdin = sys.stdin
            sys.stdin = io.StringIO(input)
            pyp.run_pyp(pyp.parse_options(cmd))
        finally:
            sys.stdin = old_stdin
    return output.getvalue()


def test_explain():
    command = [
        "pyp",
        "--explain",
        "-b",
        "d = defaultdict(list)",
        "user, pid, *_ = x.split()",
        "d[user].append(pid)",
        "-a",
        'del d["root"]',
        "-a",
        "d",
    ]
    explain_output = run_pyp(command)
    script = r"""
#!/usr/bin/env python3
from collections import defaultdict
from pyp import pypprint
import sys
d = defaultdict(list)
for x in sys.stdin:
    x = x.rstrip('\n')
    (user, pid, *_) = x.split()
    d[user].append(pid)
del d['root']
if d is not None:
    pypprint(d)
"""
    script = script.lstrip("\n")
    if sys.version_info < (3, 9):
        # astunparse seems to parenthesise things slightly differently, so filter through ast to
        # hackily ensure that the scripts are the same.
        assert ast.dump(ast.parse(explain_output)) == ast.dump(ast.parse(script))
    else:
        assert explain_output == script


def test_failures():
    with pytest.raises(pyp.PypError):
        # No possible output
        run_pyp("pyp 'x = 1'")
    with pytest.raises(pyp.PypError):
        # Unclear which transformation
        run_pyp("pyp 'print(x); print(len(lines))'")
    with pytest.raises(pyp.PypError):
        # Multiple candidates for loop variable
        run_pyp("pyp 'print(x); print(s)'")


def test_edge_cases():
    """Tests that hit various edge cases and/or improve coverage."""
    assert run_pyp("pyp pass") == ""
    assert run_pyp("pyp '1; pass'") == ""
    assert run_pyp("pyp 'print(1)'") == "1\n"
    assert run_pyp("pyp 'output = 0; 1'") == "1\n"
    assert run_pyp("pyp 'pypprint(1); pypprint(1, 2)'") == "1\n1 2\n"
    assert run_pyp("pyp i", input="a\nb") == "0\n1\n"
    assert run_pyp("pyp --define-pypprint lines", input="a\nb") == "a\nb\n"

    assert run_pyp("pyp 'if int(x) > 2: x'", input="1\n4\n2\n3") == "4\n3\n"
    assert run_pyp("pyp 'if int(x) > 2:' ' if int(x) < 4: x'", input="1\n4\n2\n3") == "3\n"
    assert run_pyp("pyp 'with contextlib.suppress(): x'", input="a\nb") == "a\nb\n"
    with pytest.raises(pyp.PypError):
        run_pyp("pyp 'if int(x) > 2: int(x)' 'else: int(x) + 1'")
