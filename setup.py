import os
import setuptools
import versioneer

# Constants
THIS_DIR = os.path.dirname(__file__)

# Package description
with open("README.md", "r") as fh:
    long_description = fh.read()


# Package requirements helper
def read_requirements_from_file(filepath):
    '''Read a list of requirements from the given file and split into a
    list of strings. It is assumed that the file is a flat
    list with one requirement per line.
    :param filepath: Path to the file to read
    :return: A list of strings containing the requirements
    '''
    with open(filepath, 'rU') as req_file:
        return req_file.readlines()


setup_args = dict(
    install_requires=read_requirements_from_file(
        os.path.join(
            THIS_DIR,
            'requirements.txt')),
    tests_require=read_requirements_from_file(
        os.path.join(
            THIS_DIR,
            'requirements-dev.txt')))

authors = [
    'Marshall McDonnell (marshallmcdonnell)',
    'Mathieu Doucet (mdoucet)',
    'Ross Whitfield (rosswhitfield)',
    'Pete Peterson (peterfpeterson)',
    'Yuanpeng Zhang (Kvieta1990)',
]

setuptools.setup(
    name="pystog",
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    author=",".join(authors),
    author_email="mcdonnellmt@ornl.gov",
    url="https://github.com/neutrons/pystog",
    description="Manipulate total scattering functions",
    long_description_content_type="text/markdown",
    license="GPL License (version 3)",
    packages=setuptools.find_packages(exclude=["fortran"]),
    package_data={'': ['*.dat', '*.gr']},
    setup_requires=[],
    install_requires=setup_args['install_requires'],
    tests_require=setup_args['install_requires'] + setup_args['tests_require'],
    test_suite='tests',
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: OS Independent",
    ],
    entry_points={
        'console_scripts': [
            "pystog_cli = pystog.cli:pystog_cli",
        ]
    }
)
