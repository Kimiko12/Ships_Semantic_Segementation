from setuptools import find_packages, setup
from typing import List
import os

HYPEN_E_DOT = '-e .'

def get_requirements(file_path: str) -> List[str]:
    '''
    This function will return the list of requirements
    '''
    requirements = []
    if os.path.exists(file_path):
        with open(file_path) as file_obj:
            requirements = file_obj.readlines()
            requirements = [req.strip() for req in requirements if req.strip()]
            if HYPEN_E_DOT in requirements:
                requirements.remove(HYPEN_E_DOT)
    else:
        print(f"Error: File {file_path} not found.")
    return requirements


setup(
    name="Ship_Semantic_Segementation",
    version="0.0.1",
    author="Nechay",
    author_email="kolak3021@gmail.com",
    packages=find_packages(),
    install_requires=get_requirements('requirements.txt')
)