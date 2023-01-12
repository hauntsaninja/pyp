Hello, and thank you for you interest in pyp!

Contributions in the form of bug reports, questions and feature requests are
very welcome and very appreciated.

Contributions in the form of PRs are also very appreciated, however, be warned
that for this project I will likely be picky. For example, I may err on the
side of caution in choosing to accept new features, or refactor perfectly fine
contributions to better suit my personal taste. This isn't a reflection on your
contribution, rather just how I'm choosing to manage this particular project.
I want to ensure that pyp remains small, standalone, and most importantly,
"magical, but never mysterious". In particular, the definition of "mysterious"
can be subjective.  Thank you for understanding and for wanting to improve pyp!


## Making a release

- Update the changelog
- Update the version in `CHANGELOG.md`
- Update the version in `__version__`
- Update the version in `pyproject.toml`
- `rm -rf dist`
- `python -m build .`
- `twine upload dist/*`
- Tag the release on Github
