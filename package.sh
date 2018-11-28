#!/usr/bin/env bash

python3 setup.py sdist bdist_wheel
twine upload --repository pypi dist/*
rm -r build/ dist/ pystog.egg-info/
