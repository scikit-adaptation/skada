from setuptools import find_packages, setup

setup(
    name='skada',
    version='0',
    description='A domain adaptation toolbox to reduce shift between domains.',

    # The project's main homepage.
    url='https://github.com/scikit-adaptation/skada',

    # Author details
    author='Théo Gnassounou',
    author_email='theo.gnassounou@inria.fr',

    # Choose your license
    license='BSD 3-Clause',
    # What does your project relate to?
    keywords='domain adaptation, machine learning, deep-learning',

    packages=find_packages(),
)
