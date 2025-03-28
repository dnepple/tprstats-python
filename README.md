# tprstats

Statistical methods and applications for students at the Tepper School of Business.

## Installation

### Using pip
`pip install git+https://github.com/dnepple/tprstats-python`

### Using conda
1. Activate your conda environment.
2. `conda install git pip`
3. `pip install git+https://github.com/dnepple/tprstats-python`

## License

`tprstats` was created by Stephen Epple and Dennis Epple. It is licensed under the terms of the MIT license.

## Development
The package is built with the [Pixi](https://pixi.sh/latest/) package management tool.

`pixi build` - build the package for conda.

If you encounter the following error building inside the devcontainer, simply move the file manually. 
```bash
Error:   × failed to move /workspaces/tprstats-python/.pixi/tprstats-python-RojFWwUxk8k/noarch/tprstats-0.1.0-pyhbf21a9e_0.conda to ./tprstats-0.1.0-pyhbf21a9e_0.conda
```

`pixi run docs` - update and rebuild the documentation website. The build directory is docs/_build/html. 

## Citations 
**statsmodels**  
Skipper, S., & Josef, P. (2010). statsmodels: Econometric and statistical modeling with python. 9th Python in Science Conference

**numpy**  
Harris, C.R., Millman, K.J., van der Walt, S.J. et al. Array programming with NumPy. Nature 585, 357–362 (2020). DOI: 10.1038/s41586-020-2649-2.

**scipy**  
Pauli Virtanen, Ralf Gommers, Travis E. Oliphant et al. SciPy 1.0 Contributors. (2020) SciPy 1.0: Fundamental Algorithms for Scientific Computing in Python. Nature Methods, 17(3), 261-272. DOI: 10.1038/s41592-019-0686-2.  

## Credits
tprstats was created with [`cookiecutter`](https://cookiecutter.readthedocs.io/en/latest/) and the [py-pkgs-cookiecutter template](https://github.com/py-pkgs/py-pkgs-cookiecutter).
