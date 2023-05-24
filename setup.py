#!/usr/bin/env python

# This is a shim to hopefully allow Github to detect the package, build is done with poetry

import setuptools

with open('requirements.txt') as f:
    requirements = f.read().splitlines()

if __name__ == "__main__":
    setuptools.setup(name="cross_system_integration",
                     packages=['cross_system_integration_old', 'cross_system_integration'],
                     install_requires=requirements)
