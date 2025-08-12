# Contributor Guide

Thank you for your interest in contributing to this project!
This project is open-source under the [MIT license] and
welcomes contributions in the form of bug reports, feature requests,
and pull requests.

Here is a list of important resources for contributors:

- [Source Code]
- [Issue Tracker]
- [Code of Conduct]
- [Postmodern Python]

[mit license]: LICENSE
[source code]: https://github.com/bilgelm/neurovoxel
[issue tracker]: https://github.com/bilgelm/neurovoxel/issues
[code of conduct]: CODE_OF_CONDUCT.md
[postmodern python]: https://rdrn.me/postmodern-python/

## How to report a bug

Report bugs on the [Issue Tracker].

When filing an issue, make sure to answer these questions:

- Which operating system and Python version are you using?
- Which version of this project are you using?
- What did you do?
- What did you expect to see?
- What did you see instead?

The best way to get your bug fixed is to provide a test case,
and/or steps to reproduce the issue.

## How to request a feature

Request features on the [Issue Tracker].

## How to set up your development environment

This project follows [Postmodern Python] guidelines, with [`uv`] for dependency
management.

You can install [`uv`] using:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Then, you can install Python and dependencies with:

```bash
uv sync
```

You can now run the basic command-line interface:

```bash
uv run neurovoxel --help
```

or fire up the server app:

```bash
uv run neurovoxel-app
```

[`uv`]: https://docs.astral.sh/uv/

Before making commits with your modified code, make sure to first run format,
lint (python and markdown), type, and test checks.
These can be run individually with

```bash
uv run poe format
           lint
           typecheck
           test
           mdlint
```

or you can use `uv run poe all` to run the above tests sequentially.

## 🦺 CI/CD

This has Github Actions setup for Pull Requests.
The [pr.yml](.github/workflows/pr.yml) workflow will run on any new Pull Request.
Change some code, open a PR and wait for the green tick!

## 🐳 Docker

It also has a Dockerfile that you can try out as follows:

1. Build it

```bash
docker build --tag neurovoxel .
```

2. Run it

```bash
docker run -p 8501:8501 --rm neurovoxel
```
