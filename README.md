# DNF to CNF ruleset converter

This repository contains converters that transform a Disjunctive Normal Form (DNF) ruleset into a Conjunctive Normal Form (CNF) ruleset. It is designed to work with objects from [decision-rules](https://github.com/ruleminer/decision-rules).

## Installation

Clone repository and use pip to install the package:

```bash
git clone https://github.com/ruleminer/dnf-rules-to-cnf.git dnf_rules_to_cnf
cd dnf_rules_to_cnf
pip install .
```

## Features

There are 2 converters available:

1. **SympyCNFConverter**
    - Utilizes the [SymPy](https://www.sympy.org/en/index.html) library for the strict CNF conversion
    - Ensures a pure CNF form but can be computationally expensive for complex datasets and large rulesets.
2. **NnfCNFConverter**
    - Uses [NNF](https://python-nnf.readthedocs.io/en/stable/) library and applies the [Tseytin transformation](https://en.wikipedia.org/wiki/Tseytin_transformation)
    - It is faster than the SymPy converter but doesn't produce a strictly pure CNF form - the reason is that the Tseytin transformation introduces auxiliary variables that are not present in the original ruleset, requiring conversion back to the original variables.

## Usage

For detailed usage see the [example notebook](https://github.com/ruleminer/dnf-rules-to-cnf/blob/master/examples/basic_usage.ipynb). Here is a quick example:

```python
from dnf_rules_to_cnf import SympyCNFConverter

# Load a ruleset
decision_rules_ruleset = ...

# Convert a ruleset to CNF
converter = SympyCNFConverter()
cnf_ruleset = converter.convert_to_cnf(decision_rules_ruleset)
```