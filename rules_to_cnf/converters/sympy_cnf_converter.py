"""This module contains the SympyCNFConverter class, which is a subclass of the CNFConverter class. 
It is used to convert a DNF-form ruleset to CNF-form ruleset using the sympy library."""

from copy import deepcopy
from typing import Tuple, Union, List

from decision_rules.core.condition import AbstractCondition
from decision_rules.conditions import LogicOperators, CompoundCondition
from decision_rules.classification.ruleset import ClassificationRuleSet
from sympy.core.symbol import Symbol
from sympy.logic.boolalg import (
    Or,
    And,
    Not,
    is_dnf,
    is_cnf,
    to_cnf,
    BooleanFunction,
)

from .cnf_converter import CNFConverter


class SympyCNFConverter(CNFConverter):
    """Sympy CNF converter class."""

    def convert_to_cnf(
        self, dnf_ruleset: ClassificationRuleSet
    ) -> ClassificationRuleSet:
        """Convert a DNF-form ruleset to CNF-form ruleset.

        Args:
            dnf_ruleset (ClassificationRuleSet): Decision rules ruleset in DNF form.

        Returns:
            ClassificationRuleSet: Decision ruleset in CNF form.
        """
        attributes = dnf_ruleset.column_names
        rules_by_conclusion = self.split_rules_by_conclusion(dnf_ruleset.rules)
        cnf_rules = []
        for conclusion, rules in rules_by_conclusion.items():
            print(
                f"Converting rules for conclusion {conclusion} to sympy...", flush=True
            )
            sympy_premises, unique_conditions = self.rules_to_sympy(rules)
            sympy_dnf = Or(
                *sympy_premises[: self.max_num_rules if self.max_num_rules else None]
            )

            assert is_dnf(sympy_dnf) is True, "Sympy expression is not in DNF form"
            print("Converting DNF to CNF...", flush=True)
            sympy_cnf = to_cnf(sympy_dnf, simplify=True, force=True)
            assert is_cnf(sympy_cnf) is True, "Sympy expression is not in CNF form"

            # convert to decision rules and treat as a single rule for a given conclusion
            print("Converting CNF to decision rules...", flush=True)
            cnf_operator, cnf_subexpressions = self.split_sympy_expr(sympy_cnf)
            cnf_premises = self.sympy_to_rules(cnf_subexpressions, unique_conditions)
            cnf_merged_premise = self.merge_premises(cnf_premises, cnf_operator)
            cnf_rules += self.create_rules_for_premises(
                [cnf_merged_premise], conclusion, attributes
            )
            print(
                "Conversion finished.\n",
                flush=True,
            )

        print(f"Creating ruleset out of {len(cnf_rules)} rules...", flush=True)
        return ClassificationRuleSet(cnf_rules)

    @staticmethod
    def get_decision_rules_operator(operator: BooleanFunction) -> LogicOperators:
        """Get the logic operator for decision rules corresponding to the given sympy operator.

        Args:
            operator (BooleanFunction): Operator to convert

        Returns:
            LogicOperators: Logic operator
        """
        operators = {
            And: LogicOperators.CONJUNCTION,
            Or: LogicOperators.ALTERNATIVE,
        }
        if operator not in operators:
            raise ValueError(f"Operator {operator} not supported")
        return operators[operator]

    @staticmethod
    def get_operator_for_decision_rules(operator: LogicOperators) -> BooleanFunction:
        """Get the sympy operator corresponding to the given logic operator for decision rules.

        Args:
            operator (LogicOperators): Logic operator

        Returns:
            BooleanFunction: Operator
        """
        operators = {
            LogicOperators.CONJUNCTION: And,
            LogicOperators.ALTERNATIVE: Or,
        }
        return operators[operator]

    def rules_to_sympy(self, rules: list) -> Tuple[BooleanFunction, dict]:
        """Convert decision rules to sympy statements. Returns a list with rules converted
        to sympy statements and a dictionary with unique conditions. The unique conditions
        dictionary contains condition strings as keys and dictionaries with keys 'sympy'
        and 'obj' as values. The 'sympy' key contains a sympy symbol representing the condition
        and the 'obj' key contains the condition object from decision rules library.

        Args:
            rules (list): List of decision rules.

        Returns:
            Tuple[BooleanFunction, dict]: Tuple with a list of sympy statements and a dictionary
            with unique conditions."""
        unique_conditions = {}
        sympy_rules = []
        for rule in rules:
            sympy_rule = self._process_rules_premise(
                rule.premise, rule.column_names, unique_conditions, Symbol
            )
            sympy_rules.append(sympy_rule)

        return sympy_rules, unique_conditions

    def sympy_to_rules(
        self, rules: list, unique_conditions: dict
    ) -> list[AbstractCondition]:
        """Convert sympy statements to decision rules. Assumes, that given rules are just
        premises - and so, returns also a list of premises. Conclusions should be added
        separately.

        Args:
            rules (list): List of sympy statements.
            unique_conditions (dict): Dictionary with unique conditions.

        Returns:
            list: List of premises."""
        premises = []
        for rule in rules:
            if isinstance(rule, Not):
                # TODO: handle negated premises
                raise NotImplementedError("Negated premises are not supported... yet")

            premises.append(self._process_sympy_premise(rule, unique_conditions))

        return premises

    def _process_sympy_premise(
        self, premise: Union[Symbol, LogicOperators], unique_conditions: dict
    ) -> AbstractCondition:
        """Recursively process sympy premises and return a condition object.

        Args:
            premise (Union[Symbol, LogicOperators]): Sympy premise.
            unique_conditions (dict): Dictionary with unique conditions.

        Returns:
            AbstractCondition: Condition object."""
        if isinstance(premise, Symbol):
            if self._is_in_symbols(premise, unique_conditions):
                return self._get_condition_by_symbol(premise, unique_conditions)
            elif self._is_in_symbols(~premise, unique_conditions):
                condition = deepcopy(
                    self._get_condition_by_symbol(~premise, unique_conditions)
                )
                condition.negated = not condition.negated
                return condition

            raise ValueError(f"Premise {premise} not found in unique_conditions")

        operator, subexpressions = self.split_sympy_expr(premise)
        subconditions = [
            self._process_sympy_premise(condition, unique_conditions)
            for condition in subexpressions
        ]

        if len(subconditions) == 1:
            return subconditions[0]

        return CompoundCondition(subconditions, operator)

    def split_sympy_expr(
        self, expression: BooleanFunction
    ) -> Tuple[LogicOperators, list]:
        """Split a sympy expression into a list of subexpressions and the main operator.

        Args:
            expression (BooleanFunction): Sympy expression.

        Returns:
            Tuple[LogicOperators, list]: Tuple with the main operator and a list of subexpressions.
        """
        if isinstance(expression, Symbol):
            # in case of a single condition
            return LogicOperators.CONJUNCTION, [expression]

        if not is_dnf(expression) and not is_cnf(expression):
            raise ValueError("Expression is not in DNF or CNF form")

        operator = self.get_decision_rules_operator(expression.func)
        subexpressions = list(expression.args)
        return operator, subexpressions

    @staticmethod
    def negate_if_needed(
        negate: bool,
        operator: LogicOperators,
        subconditions: List[Union[Symbol, LogicOperators]],
    ) -> LogicOperators:
        """Returns a boolean operator with subconditions, negated if negate is True.

        Args:
            negate (bool): Whether to negate the operator.
            operator (LogicOperators): Operator to negate.
            subconditions (List[Union[Symbol, LogicOperators]]): List of subconditions.

        Returns:
            LogicOperators: Operator with subconditions. For example And(A, B, C) or ~And(A, B, C).
        """
        return ~(operator(*subconditions)) if negate else operator(*subconditions)
