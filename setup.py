from setuptools import setup, find_packages


with open('LICENSE.md') as f:
    license = f.read()

setup(name="fastships_hull_generator",
	  version="0.0.0",
	  description="Tools for generating demihulls representative of those found on slender-hull catamaran fast-ferries.",
	  author="John Martin Godoe",
	  author_email="john.martin.godo@ntnu.no",
	  license=license,
	  packages=find_packages(exclude=('examples')),
	  install_requires=["numpy", "scipy", "matplotlib"],
	  include_package_data=True,
)
