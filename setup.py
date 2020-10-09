from setuptools import setup, find_packages

setup(
    name="actionable-recourse",
    version="1.0.1",
    author="Berk Ustun, Alexander Spangher",
    author_email="squall14414@gmail.com, alexander.spangher@gmail.com",
    description="actionable-recourse is a python library for recourse verification and reporting.",
    long_description=open('README.md').read(),
    long_description_content_type="text/markdown",
    url="https://github.com/ustunb/actionable-recourse",
    packages=find_packages(),
    install_requires=open('requirements.txt').read().split('\n'),
    python_requires='>=3.6',
)