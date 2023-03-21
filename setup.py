#!/usr/bin/env python

# This is a shim to hopefully allow Github to detect the package, build is done with poetry

import setuptools

with open('requirements.txt') as f:
    requirements = f.read().splitlines()

if __name__ == "__main__":
    setuptools.setup(name="cross_species_prediction",
                     packages=['cross_species_prediction', 'constraint_pancreas_example'],
                     install_requires=requirements)
