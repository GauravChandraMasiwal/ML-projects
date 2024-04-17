from setuptools import find_packages,setup
from typing import List

def get_requirement(path_file:str)-> List[str]:
    require = []
    with open(path_file,'r') as f:
        require= f.readlines()
        require = [req.replace('\n','') for req in require]

        if '-e .' in require:
            require.remove('-e .')
    return require


setup(
    name = 'ml_project',
    version = '0.0.1',
    author = 'gaurav',
    author_email = 'chandra.gaurav2018@gmail.com',
    packages = find_packages(),
    install_requires = get_requirement('requirements.txt')

)