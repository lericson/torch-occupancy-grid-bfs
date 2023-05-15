from setuptools import setup, find_packages

setup(
    name='torch-occupancy-grid-bfs',
    version='0.1.0',
    description='BFS implementation in an occupancy grid using PyTorch',
    author='Ludvig Ericson',
    author_email='ludvig@lericson.se',
    #packages=find_packages(),
    py_modules=['occupancy_grid_bfs'],
    install_requires=[
        'torch',
    ],
)
