from setuptools import setup, find_packages

setup(
    name='rules-to-cnf',
    version='1.0.0',
    description='A converter for transforming DNF rulesets to CNF format.',
    author='Bartosz Piguła',
    author_email='your.email@example.com',
    url='https://github.com/ruleminer/dnf-rules-to-cnf',
    packages=find_packages(),
    install_requires=[
        'decision-rules>=1.4.1',
        'nnf>=0.4.1',
        'sympy>=1.13.3',
        'pandas>=2.2.3',
        'joblib>=1.4.2',
        'rulekit>=2.1.24.0',
    ],
    python_requires='>=3.10',
)
