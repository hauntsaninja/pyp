# Changelog

## Unreleased

- Added CHANGELOG.md
- Automatic printing now tries a little harder to find standalone expressions to print
- The output of --script now has a shebang
- Mention variable names in error messages
- Fixed bugs in undefined name detection that affected imports
- Updated Related Projects in README.md

Thanks to @rmcgibbo and @yuvadm!

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
