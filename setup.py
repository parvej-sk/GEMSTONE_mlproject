from setuptools import setup, find_packages
from typing import List


Hypen_e = '-e .'

def get_req(file_path:str) ->List[str]:

    requirements = []

    with open(file_path) as file_obj:
        requirements = file_obj.readline()
        requirements = [req.replace('\n', "") for req in requirements]

        if Hypen_e in requirements:
            requirements.remove(Hypen_e)

    return requirements



setup(
    name='src',
    version='0.0.1',
    author='parvej',
    license='MIT',
    packages=find_packages(),
    install_requires = get_req('requirements.txt'))