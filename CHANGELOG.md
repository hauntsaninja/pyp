# Changelog

## [Unreleased]

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
