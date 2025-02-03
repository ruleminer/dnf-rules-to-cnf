"""This module contains the abstract class CNFConverter, which is used to convert a 
DNF-form ruleset to CNF-form ruleset."""

from abc import ABC, abstractmethod
from typing import Any, List

from decision_rules.conditions import LogicOperators
from decision_rules.classification.rule import (
    ClassificationRule,
    ClassificationConclusion,
)
from decision_rules.classification.ruleset import ClassificationRuleSet
from decision_rules.core.condition import AbstractCondition
from decision_rules.conditions import (
    CompoundCondition,
)


class CNFConverter(ABC):
    """Abstract class for converting a DNF-form ruleset to CNF-form ruleset."""

    def __init__(self, max_num_rules: int = None):
        """Initialize the CNF converter.

        Args:
            max_num_rules (Optional[int], optional): Maximum number of rules for a single
                conclusion. Defaults to None. If given, only the first max_num_rules will be
                converted."""
        self.max_num_rules = max_num_rules

    @abstractmethod
    def convert_to_cnf(
        self, dnf_ruleset: ClassificationRuleSet
    ) -> ClassificationRuleSet:
        """Convert a DNF-form ruleset to CNF-form ruleset.

        Args:
            dnf_ruleset (ClassificationRuleSet): Decision rules ruleset in DNF form.

        Returns:
            ClassificationRuleSet: Decision ruleset in CNF form.
        """

    @staticmethod
    @abstractmethod
    def get_decision_rules_operator(operator: Any) -> LogicOperators:
        """Get the logic operator for decision rules corresponding to the given operator.

        Args:
            operator: Operator to convert

        Returns:
            LogicOperators: Logic operator
        """

    @staticmethod
    @abstractmethod
    def get_operator_for_decision_rules(operator: LogicOperators) -> Any:
        """Get the operator corresponding to the given logic operator for decision rules.

        Args:
            operator: Logic operator

        Returns:
            Any: Operator
        """

    @staticmethod
    @abstractmethod
    def negate_if_needed(negate: bool, operator: Any, subconditions: List[Any]) -> Any:
        """Returns a boolean operator with subconditions, negated if negate is True.

        Args:
            negate (bool): Whether to negate the operator.
            operator (Any): Operator to negate.
            subconditions (List[Any]): List of subconditions.

        Returns:
            Any: Operator with subconditions. For example And(A, B, C) or ~And(A, B, C).
        """

    @staticmethod
    def split_rules_by_conclusion(rules: list[ClassificationRule]) -> dict:
        """Split decision rules by their conclusions. Returns a dictionary with conclusions as keys
        and lists of rules as values.

        Args:
            rules (list): List of decision rules.

        Returns:
            dict: Dictionary with conclusions as keys and lists of rules as values."""
        rules_by_conclusion = {}
        for rule in rules:
            if rule.conclusion not in rules_by_conclusion:
                rules_by_conclusion[rule.conclusion] = []
            rules_by_conclusion[rule.conclusion].append(rule)

        return rules_by_conclusion

    @staticmethod
    def merge_premises(
        premises: list[AbstractCondition], operator: LogicOperators
    ) -> AbstractCondition:
        """Merge premises into a single compound condition.

        Args:
            premises (list): List of premises.
            operator (LogicOperators): Operator for merging premises.

        Returns:
            AbstractCondition: Merged premises."""
        if len(premises) == 1:
            return premises[0]

        return CompoundCondition(premises, operator)

    @staticmethod
    def create_rules_for_premises(
        premises: list[AbstractCondition],
        conclusion: ClassificationConclusion,
        attributes: list[str],
    ) -> list[ClassificationRule]:
        """Create decision rules for given premises and conclusion.

        Args:
            premises (list): List of premises.
            conclusion (ClassificationConclusion): Conclusion of the rule.
            attributes (list): List of attributes.

        Returns:
            list: List of decision rules."""
        rules = []
        for premise in premises:
            rules.append(ClassificationRule(premise, conclusion, attributes))
        return rules

    @staticmethod
    def _get_condition_by_symbol(
        symbol: Any, unique_conditions: dict
    ) -> AbstractCondition:
        """Get a decision rules condition object by its symbol representation. It can be
        of any type, depending on the library used for cnf conversion. It is assumed that
        the symbol is available in the unique_conditions dictionary at "symbol" key and the
        condition object is available at "obj" key.

        Args:
            symbol (Any): nnf symbol representing the condition.
            unique_conditions (dict): Dictionary with unique conditions.

        Returns:
            AbstractCondition: Decision rules condition object."""
        for condition in unique_conditions.values():
            if condition["symbol"] == symbol:
                return condition["obj"]

        raise ValueError(
            f"Condition with symbol {symbol} not found in unique_conditions"
        )

    @staticmethod
    def _is_in_symbols(symbol: Any, unique_conditions: dict) -> bool:
        """Check if a symbol representing a decision rules condition is available in the
        unique_conditions dictionary.

        Args:
            symbol (Any): Symbol representing the condition.
            unique_conditions (dict): Dictionary with unique conditions.

        Returns:
            bool: True if the symbol is available in the dictionary, False otherwise."""
        return symbol in [
            condition["symbol"] for condition in unique_conditions.values()
        ]

    @staticmethod
    def _get_symbol_by_string(condition_str: str, unique_conditions: dict) -> Any:
        """Get a symbol representing a decision rules condition by its string representation. It can
        be of any type, depending on the library used for cnf conversion. It is assumed that the
        condition is available in the unique_conditions dictionary at "obj" key and the symbol is
        available at "symbol" key.

        Args:
            condition_str (str): String representation of the condition.
            unique_conditions (dict): Dictionary with unique conditions.

        Returns:
            Any: a symbol representing the condition."""
        return unique_conditions[condition_str]["symbol"]

    def _process_rules_premise(
        self,
        premise: AbstractCondition,
        attributes: List[str],
        unique_conditions: dict,
        base_symbol_cls: Any,
    ) -> Any:
        """Recursively process premise and return a statement expressed with base_symbol_cls and
        it's logic operators.

        Args:
            premise (AbstractCondition): Premise to process.
            attributes (List[str]): List of attributes. Needed for string representation.
            unique_conditions (dict): Dictionary with unique conditions.

        Returns:
            Any: Statement expressed with given base_symbol_cls and related bool operators."""
        if isinstance(premise, CompoundCondition):
            operator = self.get_operator_for_decision_rules(premise.logic_operator)
            subconditions = [
                self._process_rules_premise(
                    subcondition, attributes, unique_conditions, base_symbol_cls
                )
                for subcondition in premise.subconditions
            ]

            return self.negate_if_needed(premise.negated, operator, subconditions)

        premise_str = premise.to_string(attributes)
        # skip negation - treat ~A as a whole
        if premise_str not in unique_conditions:
            subcondition_nnf = base_symbol_cls(premise_str)

            unique_conditions[premise_str] = {
                "symbol": subcondition_nnf,
                "obj": premise,
            }

        return self._get_symbol_by_string(premise_str, unique_conditions)
