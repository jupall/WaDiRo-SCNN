from setuptools import setup, find_packages

setup(
    name='wadiro_ml',
    version='0.1',
    package_dir = {"": "src"},
    packages = find_packages(where="src"),
    install_requires=[
        'numpy>=1.26',
        'torch>=2.*',
        'cvxpy>=1.3',
        'pandas>=2.1',
        'scikit-learn>=1.3',
        'scipy>=1.11'],
    python_requires=">=3.8",
    url = "https://github.com/jupall/WaDiRo-SCNN",
    classifiers = [
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
        )