# pyp

[![Build Status](https://travis-ci.org/hauntsaninja/pyp.svg?branch=master)](https://travis-ci.org/hauntsaninja/pyp)
[![Coverage Status](https://coveralls.io/repos/github/hauntsaninja/pyp/badge.svg?branch=master)](https://coveralls.io/github/hauntsaninja/pyp?branch=master)
[![Checked with mypy](http://www.mypy-lang.org/static/mypy_badge.svg)](http://mypy-lang.org/)

Easily run Python at the shell! Magical, but never mysterious.

## Installation

Run `pip install pypyp` <sup>(note the extra "yp"!)</sup>

pyp requires Python 3.6 or above.

## How it works

pyp will statically analyse the input code to detect undefined variables. Based on what it finds,
it will proceed to transform the AST of the input code as needed. We then compile and execute the
result, or if using `--explain`, unparse the AST back to source code.

## Examples

This section will walk you through the details of using pyp, and hopefully replace your needs
for many common shell utilities. For a cheatsheet / tldr, run `pyp --help`.

#### pyp can easily be used to apply Python code to each line in the input.
Just use one of the magic variables `x`, `l`, `s` or `line` to refer to the current line.

```
# pyp like cut
ls | pyp 'x[:3]'
ps x | pyp 'line.split()[4]'
```

#### pyp can be used to easily apply Python code to the entire input as well.
Use the magic variable `lines` for a list of rstripped lines or `stdin` for `sys.stdin`.

```
# pyp like wc -c
cat /usr/share/dict/words | pyp 'len(stdin.read())'

# pyp like awk
seq 1 5 | pyp 'sum(map(int, lines))'
```

#### pyp will automatically import modules you use.

```
# pyp like sh
echo echo echo | pyp 'subprocess.run(lines[0], shell=True); pass'

# pyp like jq
curl -s 'https://api.github.com/repos/hauntsaninja/pyp/commits?per_page=1' | pyp 'json.load(stdin)[0]["commit"]["author"]'

# pyp like egrep
cat /usr/share/dict/words | pyp 'x if re.search("(p|m)yth", x) else None'
```

For `collections`, `math`, `itertools`, `pathlib.Path`, `pprint.pp`, pyp will figure it out even
if you don't use the qualified name.
```
# pyp like bc
pyp 'sqrt(5)'

# pyp like ${x##*.}
ls | pyp 'Path(x).suffix'
```

#### pyp can give you access to loop indices using the magic variables `i`, `idx` or `index`.

```
# pyp like line numbers
cat setup.py | pyp 'f"{idx+1: >3} {x}"'
```

#### Note so far you haven't had to call `print`!
By default, pyp will print the last expression in your code — except if it evaluates to `None`.
And you can always explicitly call `print` yourself, in which case pyp will stay out of your way.

```
# pyp like grep
cat /usr/share/dict/words | pyp 'x if "python" in x else None'
cat /usr/share/dict/words | pyp 'if "python" in x: print(x); "this won't print"'
```

#### pyp will attempt to intelligently print dicts and iterables.
This makes the output of pyp easier to compose with shell tools.
Again, explicit printing will stop this magic, but pyp makes the function `pypprint` available if
you do want to explicitly opt back in.

```
# pyp like tail
ls | pyp 'lines[-10:]'

# pyp like sort
ls | pyp 'sorted(lines)'
ls | pyp 'print(f"Sorting {len(lines)} lines"); pypprint(sorted(lines))'

# pyp like sort | uniq
ls | pyp 'sorted(set(lines))'
```

#### pyp lets you run snippets of Python before and after processing input.
Note if you run into trouble with semicolons and want a new line, you can just pass another string
to pyp. You can also always pipe pyp to pyp!

```
# pyp like anything!
ps aux | pyp -b 'd = defaultdict(list)' 'user, pid, *_ = x.split()' 'd[user].append(pid)' -a 'del d["root"]' -a 'd'
```

#### pyp can be magical, but it doesn't have to be mysterious!
Use `--explain` or `--script` to get a script equivalent to what pyp will run. This can also be a
useful starting point for more complex scripts.
```
pyp --explain -b 'd = defaultdict(list)' 'user, pid, *_ = x.split()' 'd[user].append(pid)' -a 'del d["root"]' -a 'd'

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
```

#### pyp is configurable.

Point the environment variable `PYP_CONFIG_PATH` to a file containing, for example:
```
import numpy as np
import tensorflow as tf
from pipetools import *

def useful_function(): ...
class UsefulClass: ...
```

When attempting to define undefined names, pyp will statically* analyse this file as a source of
possible definitions. This means that if you don't use `tf`, we won't import `tensorflow`! And of
course, `--explain` will show you exactly what gets run:

```
pyp --explain 'np.array([0]); pass'

#!/usr/bin/env python3
import sys
import numpy as np
assert sys.stdin.isatty() or not sys.stdin.read(), "The command doesn't process input, but input is present"
np.array([0])
```

Note, importing things from libraries like [pipetools](https://0101.github.io/pipetools/doc/index.html)
in your configuration can allow you to achieve high levels of syntax sugar:
```
seq 1 110 | pyp 'lines > foreach(int) | where(X > 100) | group_by(X % 3) | sort_by(X[0])'
```

<sub>\*If you use wildcard imports, we will need to import those modules if there remain undefined
names, though we skip this in the happy path. If this matters to you, definitely don't
`from tensorflow import *`! </sub>

## Related projects

### [Pyed Piper](https://code.google.com/archive/p/pyp/) aka  Python Power at the Prompt

pyp takes inspiration (and the command name!) from here.
However, Pyed Piper appears to be unmaintained, Python 2 only, and further away from Python syntax
than pyp aims to be. Github mirror [here](https://github.com/yuvadm/pyp/).

### [Pyped](https://github.com/ksamuel/Pyped)

I discovered Pyped while making this project! It's actually very similar, probably similar enough
that I wouldn't have written this had I known. However, Pyped doesn't do the AST introspection
and manipulation that we do. This means:
- It's less magical! It relies on you to pass in flags to tell it what to do, when intention can
be inferred from the input.
- It doesn't provide easy automatic printing, or smart printing of iterables and dicts.
- It hardcodes a list of imports and installs some libraries on your system. This project's
automatic import will work for any library you use.
- It doesn't have anything like `--explain`/`--script`.

However,
- It has some conveniences, like regex splitting of input, that you'd have to do for yourself here.
- It supports Python 2 and early versions of Python 3.
- It's been around for much longer.

### [piep](http://gfxmonk.net/dist/doc/piep/) / [spy](https://github.com/edk0/spy) / [pyfil](https://github.com/ninjaaron/pyfil) / [pythonpy](https://github.com/fish2000/pythonpy-fork) / [oneliner](https://python-oneliner.readthedocs.io/en/latest/)

Since writing pyp, it turns out there are more alternatives out there than I thought :-) Some quick
notes:
- Most of them rely on the user passing in flags, like Pyped.
- Most of them have limitations around automatic printing, like only being able to automatically
print single expressions or not handling iterables and dicts well.
- Some of them have custom syntax for in-process command chaining, which can be convenient.
- Some of them have specialised support for things like JSON input or running shell commands.
- Some of them expose the input in interesting ways with custom line / file / stream objects.
- Some of them have more advanced options for error handling.
- None of them have anything like `--explain`.

For whatever it's worth, I've listed these projects in approximate order of my personal preference.

### [mario](https://github.com/python-mario/mario)

`mario` is a featureful take on shell processing with Python. It doesn't use undefined name detection,
instead relying on a pluggable subcommand system. While the subcommands can mean a little more
verbosity than pyp, `mario` makes up some ground by automatic application of functions and a custom
command chaining syntax. The result can feel a little DSL-like, while pyp tries to feel very close
to writing Python.

Consider using `mario` if:
- You find yourself stringing together long sequences of pyp commands and want to be able to
command chain within a single process.
- You find yourself often needing to reuse complex pyp commands or doing a lot of domain specific
shell processing that you wish you could reuse with a single command.
- You want more builtin support for things like processing CSV or TOML.
- You want to easily be able to use async functions to process your input concurrently.

Consider pyp if:
- You want to minimise keystrokes for things that should be quick and easy.
- You want something minimal and lightweight that feels very close to Python. You don't want to have
to remember commands.
- You're happy using Python modules to do domain specific heavy lifting. You don't mind (or want to
be able to) fall back to a script via `--script` to deal with complexity.

### [xonsh](https://xon.sh/)

`xonsh` is a shell whose language is a superset of Python; this is more ambitious and pretty
different from pyp. pyp is easier to use for the one-liner piping use case, but if you need
more Python in your shell, check out `xonsh`.

### [awk](https://www.gnu.org/software/gawk/manual/gawk.html)

If `awk` works for you, how did you end up here?
