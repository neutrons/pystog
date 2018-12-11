import setuptools
import versioneer

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="pystog",
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    author="Marshall McDonnell",
    author_email="mcdonnellmt@ornl.gov",
    description="Transforms reciprocal and real space function",
    long_description_content_type="text/markdown",
    url="https://github.com/marshallmcdonnell/pystog",
    packages=setuptools.find_packages(exclude=["fortran"]),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: OS Independent",
    ],
    setup_requires=[
        "matplotlib",
        "numpy",
        "pandas"
    ],
    install_requires=[
        "matplotlib",
        "numpy",
        "pandas"
    ],
    scripts=['bin/pystog_cli']
)
