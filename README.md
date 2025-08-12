# NeuroVoxel

_NeuroVoxel_ is a Python-based application for quantifying associations of
neuroimaging measures with non-imaging variables.

## Requirements

_NeuroVoxel_ requires Python 3.13+ and the following modules:
`nilearn` for neuroimage I/O, visualization, and statistical analysis,
`nanslice` for dual-coded statistical map visualization,
`streamlit` for the web app, and
`click` for the CLI.

## 💾 Installation

You can install _NeuroVoxel_ via pip after cloning the repository:

```bash
git clone https://github.com/bilgelm/neurovoxel.git
pip install -e neurovoxel
```

## 🚀 Usage

From the command line:

```bash
python -m neurovoxel-app

# or

neurovoxel-app
```

## Contributing

Contributions are very welcome.
To learn more, see the [Contributor Guide].

## License

Distributed under the terms of the [MIT license][license],
_NeuroVoxel_ is free and open source software.

## Issues

If you encounter any problems,
please [file an issue] along with a detailed description.

## Credits

This project was generated from [@carderne]'s [Postmodern Python template].

[@carderne]: https://github.com/carderne
[Postmodern Python template]: https://github.com/carderne/postmodern-python
[file an issue]: https://github.com/bilgelm/neurovoxel/issues

<!-- github-only -->

[license]: LICENSE
[contributor guide]: CONTRIBUTING.md
