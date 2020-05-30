# FAQ

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


#### What are pyp's dependencies?

If run on Python 3.9 or later, pyp has no dependencies.

On earlier Python versions, pyp is dependent on [astunparse](https://github.com/simonpercivall/astunparse)
for `--explain`/`--script` to work as promised. Note you can use `pyp.py` as a Python script without
installing `astunparse` and everything else will work. (In fact `--script` will still output a
script that does what you want it to, but it won't be particularly readable...)


#### The output of `--explain` is a little weirdly formatted

Try running with Python 3.9 or later! The AST unparser that the standard library includes in 3.9
and later fixes a lot of formatting nits that are present in third party AST unparsers.

You can also pipe the output of `pyp --explain` to an autoformatter like [black](https://github.com/psf/black):

```
$ pyp --explain x | black --quiet -
```

#### Can I customise the shebang on the output of `--script`?

Yes! Just add the shebang you want to your [config file](https://github.com/hauntsaninja/pyp#pyp-is-configurable).
`pyp` will use the shebang of the file pointed to by `PYP_CONFIG_PATH` as the shebang for the output
of `--explain`/`--script`.

#### What is start up performance like?

It's not perceptible in my use of pyp. And remember if you're piping input into `pyp` that processes
start in parallel, so this is zero extra wall time if your piped input has any latency.

Here's a benchmark that should basically just be measuring the fixed costs of start up and AST
transformation (run on my old, not powerful laptop):
```
$ hyperfine -w 10 -m 100 'pyp x'
Benchmark #1: pyp x
  Time (mean ± σ):      81.5 ms ±   1.4 ms    [User: 60.3 ms, System: 15.9 ms]
  Range (min … max):    78.6 ms …  84.8 ms    100 runs
```

#### Can I use pyp with PyPy?

You should be able to use PyPy! Tests currently pass against PyPy (run `tox -e pypy3`).
You can install pyp using PyPy with something like `pypy3 -m pip install pypyp`.
Running `pyp sys.version` will tell you whether it worked. Note that pyp requires at least
Python 3.6; this constraint naturally applies when using PyPy as well.
