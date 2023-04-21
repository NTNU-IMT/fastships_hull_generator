# FASTSHIPS_open
Open Source part of the FASTSHIPS software package for design and 
analysis of high-speed vessels. More information can be found on 
the [FASTSHIPS web site](https://www.ntnu.edu/imt/software/fastships).
The open parts of the software include the hull_generator and mesh_tools
modules. The former allows the generation of slender catamaran hulls, 
either from detailed data on longitudinal distributions of sectional parameters 
or from main parameters and the ratios thereof. The latter is for modification 
of meshes, creation of 3D meshes from 2D sectional data and for analysis of 
volumetric and surface properties of meshes.

## Getting Started

These instructions will give you a copy of the project up and running on
your local machine for development and testing purposes.

### Prerequisites

Requirements for the software and other tools to build, test and push 
- numpy
- scipy
- matplotlib
- setuptools

### Installing

Download the package directory to your computer or the server computer on 
which it is to run. Enter the root folder of the package and run the
following commands:

    $ sudo python3 setup.py install


## Running the examples

Example scripts for running the package are included in the examples 
directory, in the form of jupyter notebook documents. These can be opened
in most editors, e.g. VS Code, or by installing and running Jupyter 
Notebook from the command line. See instructions on 
https://jupyter.org/install


## Built With

  - [Contributor Covenant](https://www.contributor-covenant.org/) - Used
    for the Code of Conduct

## Contributing

Please read [CONTRIBUTING.md](CONTRIBUTING.md) for details on our code
of conduct, and the process for submitting pull requests to us.

## Versioning

We use [Semantic Versioning](http://semver.org/) for versioning. For the versions
available, see the [tags on this
repository](https://github.com/NTNU-IMT/FASTSHIPS_open/tags).

## Authors

  - **John Martin Kleven Godø** - *Initial code development* -
    [johnmartingodo](https://github.com/johnmartingodo)

See also the list of
[contributors](https://github.com/NTNU-IMT/FASTSHIPS_open/contributors)
who participated in this project.

## License

This project is licensed under the [GNU General Public License v3.0](LICENSE.md) - see 
the [LICENSE.md](LICENSE.md) file for details

## Acknowledgments

The development of this code was done at the NTNU Department of Marine Technology. 
It was funded by the enabling Zero Emission passenger 
Vessel Services (ZEVS) research project (NFR grant No. 320659) and 
SFI Smart Maritime (NFR grant No. 237917). Work was supervised by professors
Sverre Steen and Odd Faltinsen. 
