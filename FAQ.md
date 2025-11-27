# FAQ

### Contents

- [I'm running into issues with newlines / complicated statements](#im-running-into-issues-with-newlines--complicated-statements)
- [What's in your config?](#whats-in-your-config)
- [What are pyp's dependencies?](#what-are-pyps-dependencies)
- [Can I customise the shebang on the output of `--script`?](#can-i-customise-the-shebang-on-the-output-of---script)
- [The output of `--explain` is a little weirdly formatted](#the-output-of---explain-is-a-little-weirdly-formatted)
- [What is start up performance like?](#what-is-start-up-performance-like)
- [What is overall performance like?](#what-is-overall-performance-like)
- [Can I use pyp with PyPy?](#can-i-use-pyp-with-pypy)

---

#### I'm running into issues with newlines / complicated statements

You have a couple options here! As mentioned in the [README](https://github.com/hauntsaninja/pyp#pyp-lets-you-run-snippets-of-python-before-and-after-processing-input),
you can pass in another string to pyp to get a newline. You can also take advantage of features of
your shell, such as multiline strings or ANSI C-quoted strings. Let's look at examples!

Passing multiple strings to pyp:
```
$ pyp --explain 'if int(x) >= 100: x += " is big"; print(x)' 'else: print(x + " is small")'
#!/usr/bin/env python3
import sys
for x in sys.stdin:
    x = x.rstrip('\n')
    if int(x) >= 100:
        x += ' is big'
        print(x)
    else:
        print(x + ' is small')
```

Using a multiline string:
```
$ pyp --explain '
> if int(x) >= 100:
>   x += " is big"
>   print(x)
> else:
>   print(x + " is small")
> '
```

Using an ANSI C-quoted string:
```
$ pyp --explain $'if int(x) >= 100:\n  x += " is big"\n  print(x)\nelse:\n  print(x + " is small")'
```

#### What's in your config?

My config is pretty simple:
```py
import numpy as np

n = int(x)
j = json.loads(stdin)
f = x.split()
# like f, but returns None if index is of bounds
ff = defaultdict(lambda: None, dict(enumerate(x.split())))

d = defaultdict(list)
```

#### Can I customise the shebang on the output of `--script`?

Yes! Just add the shebang you want to your [config file](https://github.com/hauntsaninja/pyp#pyp-is-configurable).
`pyp` will use the shebang of the file pointed to by `PYP_CONFIG_PATH` as the shebang for the output
of `--explain`/`--script`.

#### The output of `--explain` is a little weirdly formatted

You can pipe the output of `pyp --explain` to an autoformatter like [black](https://github.com/psf/black):

```
$ pyp --explain x | black --quiet -
```

#### What is start up performance like?

It's not perceptible in my use of pyp. And remember that if you're piping input into `pyp`,
processes start in parallel, so this is zero extra wall time if your piped input has any latency.

Here's a benchmark that should basically just be measuring the fixed costs of start up and AST
transformation (run on my laptop):
```
hyperfine -w 10 -m 100 'pyp x'
Benchmark #1: pyp x
  Time (mean ± σ):      56.3 ms ±   1.0 ms    [User: 41.2 ms, System: 11.4 ms]
  Range (min … max):    53.9 ms …  60.3 ms    100 runs
```

One note here, as mentioned in the README, is that if you use wildcard imports (`from x import *`)
in your [config file](https://github.com/hauntsaninja/pyp#pyp-is-configurable), pyp might need to
perform those imports to resolve undefined names (although pyp avoids this if possible). If this is
causing you latency, just import what you need instead (i.e., change your config to read
`from x import y, z`).

#### What is overall performance like?

Better than awk! ;-)
```
$ hyperfine -w 3 -m 10 "seq 1 999999 | pyp 'sum(map(int, stdin))'"
Benchmark #1: seq 1 999999 | pyp 'sum(map(int, stdin))'
  Time (mean ± σ):     258.2 ms ±   5.6 ms    [User: 422.3 ms, System: 17.0 ms]
  Range (min … max):   252.1 ms … 270.7 ms    11 runs

$ hyperfine -w 3 -m 10 'seq 1 999999 | awk "{s += $0} END {print s}"'
Benchmark #1: seq 1 999999 | awk "{s += $0} END {print s}"
  Time (mean ± σ):     405.3 ms ±   3.4 ms    [User: 599.6 ms, System: 5.5 ms]
  Range (min … max):   399.4 ms … 410.9 ms    10 runs
```

More seriously, random micro benchmark aside, pyp should be fast enough that you shouldn't worry
about performance until you know you need to, as with Python itself.

One note here is that the the magic variable `lines` is currently always a list. Hence, if you find
yourself processing really large input, it might be better to use the magic variable `stdin`
(this is just `sys.stdin`). For the above example:

```
$ hyperfine -w 3 -m 10 "seq 1 999999 | pyp 'sum(map(int, lines))'"
Benchmark #1: seq 1 999999 | pyp 'sum(map(int, lines))'
  Time (mean ± σ):     378.9 ms ±   3.2 ms    [User: 530.2 ms, System: 38.5 ms]
  Range (min … max):   375.4 ms … 384.6 ms    10 runs
```

#### Can I use pyp with PyPy?

You should be able to use PyPy! Tests currently pass against PyPy (run `tox -e pypy3`).
You can install pyp using PyPy with something like `pypy3 -m pip install pypyp`.
Running `pyp sys.version` will tell you whether it worked. Note that pyp requires at least
Python 3.9; this constraint naturally applies when using PyPy as well.
