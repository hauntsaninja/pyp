# Installation

TODO: put this on PyPI

`pip install git+https://github.com/hauntsaninja/pyp@master`

# Examples

`pyp` can easily be used to apply Python code to each line in the input. Just use one of the magic
variables `x`, `l`, `s` or `line` to refer to the current line.

```
# pyp like cut
ls | pyp 'x[:3]'
ps x | pyp 'line.split()[4]'
```

`pyp` can be used to easily apply Python code to the entire input as well. Use the magic variable
`lines` for a list of stripped lines or `stdin` for `sys.stdin`.

```
# pyp like wc -c
cat /usr/share/dict/words | pyp 'len(stdin.read())'

# pyp like awk
seq 1 5 | pyp 'sum(map(int, lines))'
```

`pyp` will automatically import modules you use.

```
# pyp like sh
echo echo echo | pyp 'subprocess.run(lines[0], shell=True); pass'

# pyp like jq
curl -s 'https://api.github.com/repos/hauntsaninja/pyp/commits?per_page=1' | pyp 'json.load(stdin)[0]["commit"]["author"]'

# pyp like egrep
cat /usr/share/dict/words | pyp 'x if re.search("(p|m)yth", x) else None'
```

For `collections`, `math`, `itertools`, `pathlib.Path`, `pprint.pp`, `pyp` will figure it out even
if you don't use the qualified name.
```
# pyp like bc
pyp 'sqrt(5)'

# pyp like ${x##*.}
ls | pyp 'Path(x).suffix'
```

`pyp` can give you access to loop indices using the magic variables `i`, `idx` or `index`.

```
# pyp like line numbers
cat setup.py | pyp 'f"{idx+1: >3} {x}"'
```

Note so far you haven't had to call `print`! By default, `pyp` will print the last expression in
your code -- except if it evaluates to `None`. And you can always explicitly call `print` yourself,
in which case `pyp` will stay out of your way.

```
# pyp like grep
cat /usr/share/dict/words | pyp 'x if "python" in x else None'
cat /usr/share/dict/words | pyp 'if "python" in x: print(x); "this won't print"'
```

`pyp` will intelligently attempt to print dicts and iterables in a way that makes it easier to
compose with shell tools. Again, explicit printing will stop this magic, but `pyp` makes the
function `pypprint` available if you do want to explicitly opt back in.

```
# pyp like tail
ls | pyp 'lines[-10:]'

# pyp like sort
ls | pyp 'sorted(lines)'
ls | pyp 'print(f"Sorting {len(lines)} lines"); pypprint(sorted(lines))'

# pyp like sort | uniq
ls | pyp 'sorted(set(lines))'
```

`pyp` lets you run snippets of Python before and after. Note if you run into trouble with
semicolons and want a new line, you can just pass another string to `pyp`.
You can also always pipe `pyp` to `pyp`!

```
# pyp like anything!
ps aux | pyp -b 'd = defaultdict(list)' 'user, pid, *_ = x.split()' 'd[user].append(pid)' -a 'del d["root"]' -a 'd'
```

`pyp` can be magical, but it doesn't have to be mysterious! Use `--explain` or `--script` to get a
script equivalent to what `pyp` will run. This can also be a useful starting point for more complex
scripts.
```
pyp --explain -b 'd = defaultdict(list)' 'user, pid, *_ = x.split()' 'd[user].append(pid)' -a 'del d["root"]' -a 'd'

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
```

# Related Projects

### [Pyed Piper](https://code.google.com/archive/p/pyp/) aka  Python Power at the Prompt

`pyp` takes inspiration (and the command name!) from here.
However, Pyed Piper appears to be unmaintained, Python 2 only, and further away from Python syntax
than `pyp` aims to be.

### [Pyped](https://github.com/ksamuel/Pyped)

I discovered Pyped while making this project! It's actually very similar, probably similar enough
that I wouldn't have written this had I known. However, Pyped doesn't do the AST introspection
and manipulation that we do. This means:
- It's less magical! It relies on you to pass in flags to tell it what to do.
- It doesn't provide smart printing of iterables and dicts.
- It hardcodes a list of imports and installs some libraries on your system. This project's
automatic import will work for any library you use.
- It doesn't have anything like `--script`.

However,
- It has some conveniences, like regex splitting of input, that you'd have to do for yourself here.
- It supports Python 2 (if that's still something you need).
- It's been around for much longer.

### [xonsh](https://xon.sh/)

`xonsh` is a shell whose language is a superset of Python; this is more ambitious and pretty
different from `pyp`. `pyp` is easier to use for the one-liner piping use case, but if you need
more Python in your shell, check out `xonsh`.

### [awk](https://www.gnu.org/software/gawk/manual/gawk.html)

If `awk` works for you, how did you end up here?
