# Changelog

## [Unreleased]

- Reduce reconstructed traceback's reliance on CPython implementation details
- Fix automatic print behaviour in edge case interaction with scopes

## [v0.3.3]

- Wildcard imports in code passed to pyp are now supported
- Name finding now does basic scope analysis. What does this mean for you? Lambdas are more
convenient to use, you can use `global` in your config, unused name detection for exception handlers
is slightly more accurate, etc
- Config definition finding better reuses name finding code. This means you can do horrible things
like using walrus expressions in default values, and things should just work
- Top-level conditionals in config now work better. We also now forbid top-level loops, since
they're not very useful and I want to reserve the right to give them magical semantics
- Tracebacks without AST unparsing are improved
- Use of `pp` is now properly recognised as disabling automatic printing

## [v0.3.2]

- Improved reconstructed tracebacks when errors occur in statements with nested child statements
- Added a helpful note for errors resulting from using magic variables in `--before` code
- Added an FAQ
- Added `--version`

Thanks to @piec!

## [v0.3.1]

- Improved error messages resulting from failed automatic imports
- Improved undefined name detection for some pathological cases

## [v0.3]

- pyp is now configurable! See README.md for the details
- Added CHANGELOG.md
- We now attempt to show a reconstructed traceback for errors raised in generated code
- Improved various error messages. We should never show you tracebacks into pyp code unless it's a
genuine pyp crash
- The output of `--script` now has a shebang and orders imports better
- Improved undefined name detection for imports, exception handlers, class and function definitions
- Automatic printing now tries a little harder to find standalone expressions to print
- Input code is now dedent-ed
- Updated Related Projects in README.md
- Several improvements to tests

Extra thanks to @rmcgibbo!
Thanks to @nickodell, @yuvadm, @dbieber, @alendit!

## [v0.2.1]

- Fixed bugs in undefined name detection that affected loops and filtering comprehensions
- Added a shebang
- Improved tests

Thanks to @nickodell and @stuartlangridge!

## [v0.2]

- Added `--define-pypprint`, in case you don't want the output of `--script` to import `pypprint` from `pyp`
- Basic support for lambdas and functions in undefined name detection, despite a lack of scopes.
- Improved documentation and tests

## [v0.1.1]

- Added installation instructions from PyPI to README.md

## [v0.1]

- Initial release
