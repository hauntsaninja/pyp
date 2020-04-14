import subprocess
from typing import Optional


def run_cmd(cmd: str, input: Optional[str] = None) -> None:
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
    """Based on examples in the README."""

    INPUT = "python\nsnake\nss\nmagpy\npiethon\nreadme.md\n"

    compare_command("cut -c1-4", "pyp 'x[:4]'", input=INPUT)

    compare_command("wc -c | tr -d ' '", "pyp 'len(stdin.read())'", input=INPUT)

    compare_command(
        "awk '{s+=$1} END {print s}' ", "pyp 'sum(map(int, lines))'", input="1\n2\n3\n4\n5\n"
    )

    compare_command("sh", "pyp 'subprocess.run(lines[0], shell=True); pass'", input="echo echo")

    compare_command(
        'jq .[1]["lol"]',
        """pyp 'json.load(stdin)[1]["lol"]'""",
        input='[0, {"lol": "hmm"}, 0]',
        allow_example_fail=True,
    )

    compare_command(
        "grep -E '(py|md)'", """pyp 'x if re.search("(py|md)", x) else None'""", input=INPUT
    )

    compare_command("echo 'sqrt(9.0)' | bc", "pyp 'sqrt(9)'")

    compare_command("x=README.md; echo ${x##*.}", '''x=README.md; pyp "Path('$x').suffix[1:]"''')

    compare_command("echo '  1 a\n  2 b'", """pyp 'f"{idx+1: >3} {x}"'""", input="a\nb")

    compare_command("grep py", """pyp 'x if "py" in x else None'""", input=INPUT)

    compare_command("tail -n 3", "pyp 'lines[-3:]'", input=INPUT)

    compare_command("sort", "pyp 'sorted(lines)'", input=INPUT)
    compare_command(
        "echo 'Sorting 2 lines\n1\n2'",
        """pyp 'print(f"Sorting {len(lines)} lines"); pypprint(sorted(lines))'""",
        input="2\n1\n",
    )
    compare_command("sort | uniq", "pyp 'sorted(set(lines))'", input="2\n1\n1\n1\n3")


def test_explain():
    command = (
        "pyp --explain "
        "-b 'd = defaultdict(list)' "
        "'user, pid, *_ = x.split()' "
        "'d[user].append(pid)' "
        """-a 'del d["root"]' -a 'd'"""
    )
    script = r"""
from collections import defaultdict
from pyp import pypprint
import sys
d = defaultdict(list)
for x in sys.stdin:
    x = x.rstrip('\n')
    (user, pid, *_) = x.split()
    d[user].append(pid)
del d['root']
output = d
if (output is not None):
    pypprint(output)

"""
    assert run_cmd(command).decode("utf-8") == script
