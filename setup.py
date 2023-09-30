from setuptools import setup, find_packages

with open('README.md', encoding='utf-8') as f:
    long_description = f.read()

with open('requirements.txt', encoding='utf-8') as f:
    install_requires = f.read()

setup(
    name='Slice-based latent diffusion for the simultaneous generation of medical images and their segmentation masks in a controled way',
    author="",
    version="1.0",
    description="", 
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(exclude=['contrib', 'docs', 'tests']),
    install_requires=install_requires,
)