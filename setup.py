import os

from setuptools import find_packages, setup

_PATH_ROOT = os.path.dirname(__file__)

with open(os.path.join(_PATH_ROOT, "README.md"), encoding="utf-8") as fo:
    readme = fo.read()

setup(
    name="indic_llm",
    version="0.1.0",
    description="Open source large language model implementation",
    author="Adithya S Kolavi",
    url="https://github.com/adithya-s-k/Indic-llm",
    install_requires=[
        "torch>=2.2.0",
    ],
    packages=find_packages(),
    long_description=readme,
    long_description_content_type="text/markdown",
)