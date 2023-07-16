from setuptools import setup, find_packages
from typing import List

def get_requirements(file_path):
    with open(file_path) as obj:
        requirements= obj.readlines()
        requirements=[req.replace("\n","") for req in requirements]
        requirements.remove("-e .")
        return requirements

setup(name='Boston Model',
      version='0.0.1',
      author='suhasini gupta',
      packages=find_packages(),
      install_requires=get_requirements('requirements.txt'))