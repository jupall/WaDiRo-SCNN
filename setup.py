from setuptools import setup, find_packages

setup(
    name='wadiroscnn',
    version='0.6',
    package_dir = {"": "src"},
    packages = find_packages(where="src"),
    description='WaDiRo-SCNN: A Python implementation of the Wasserstein Distributionally Robust Shallow Convex Neural Networks from the work of Julien Pallage and Antoine Lesage-Landry.',
    long_description="For more information, please visit the GitHub repository: jupall/WaDiRo-SCNN",
    install_requires=[
        'numpy>=1.26',
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

