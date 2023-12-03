from setuptools import setup, find_packages

VERSION = "0.0.1"
DESCRIPTION = "Module for performing regression on SPARQL queries"
LONG_DESCRIPTION = "Module for performing regression on SPARQL queries. Prerequisite for this model is a training PlanRCGN model"

# Setting up
setup(
    # the name must match the folder name 'verysimplemodule'
    name="PlanRegr",
    version=VERSION,
    author="Abiram Mohanaraj",
    author_email="<abiramm@cs.aau.dk>",
    description=DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    packages=find_packages(),
    install_requires=[
        "configparser",
        "numpy",
        "pandas",
        "matplotlib",
        "torch",
        "dgl"
    ],  # add any additional packages that
    keywords=["python", "SPARQL"],
    classifiers=[],
)