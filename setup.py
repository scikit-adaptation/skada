from setuptools import setup, find_packages

setup(
    name='da_toolbox',
    version='0',
    description='A domain adaptation toolbox to reduce shift between domains.',

    # The project's main homepage.
    url='https://github.com/tgnassou/da-toolbox',

    # Author details
    author='Théo Gnassounou',
    author_email='theo.gnassounou@inria.fr',

    # Choose your license
    license='BSD 3-Clause',
    # What does your project relate to?
    keywords='da deep-learning',

    packages=find_packages(),
)
