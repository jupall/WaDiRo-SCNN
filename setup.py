from setuptools import setup, find_packages

setup(
    name='wadiro_ml',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'numpy>=1.26.2',
        'torch>=2.3.0',
        'cvxpy>=1.4.1',
        'pandas>=2.1.4',
        'scikit-learn>=1.3.2',
        'scipy>=1.11.4'],
    python_requires=">=3.8",
        )