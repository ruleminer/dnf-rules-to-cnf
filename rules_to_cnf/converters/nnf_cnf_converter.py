"""This module contains the NnfCNFConverter class, which is a subclass of the CNFConverter class. 
It is used to convert a DNF-form ruleset to CNF-form ruleset using the nnf python library."""

from copy import deepcopy
from typing import Tuple, Type, Union, List

from decision_rules.core.condition import AbstractCondition
from decision_rules.conditions import LogicOperators, CompoundCondition
from decision_rules.classification.ruleset import ClassificationRuleSet
from nnf import Or, And, Var, Internal, NNF
from nnf.util import memoize

from .cnf_converter import CNFConverter


class NnfCNFConverter(CNFConverter):
    """NNF library CNF converter class. Uses Tseitin encoding to convert DNF to CNF."""

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
                f"Converting rules for conclusion {conclusion} to nnf objects...",
                flush=True,
            )
            nnf_premises, unique_conditions = self.rules_to_nnf(rules)
            nnf_dnf = Or(
                {*nnf_premises[: self.max_num_rules if self.max_num_rules else None]}
            )

            print("Converting DNF to CNF...", flush=True)
            # Since tseytin method adds auxiliary variables, we need to replace them
            # in order to use ruleset for classification. Because of that the resulting ruleset
            # will be in a semi-cnf form
            nnf_cnf = self._to_cnf_with_mapping(
                nnf_dnf, simplify=True, replace_aux_vars=True
            )

            # convert to decision rules and treat as a single rule for a given conclusion
            print("Converting CNF to decision rules...", flush=True)
            cnf_operator, cnf_subexpressions = self.split_nnf_expr(nnf_cnf)
            cnf_premises = self.nnf_to_rules(cnf_subexpressions, unique_conditions)
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
    def get_decision_rules_operator(operator: Type[Internal]) -> LogicOperators:
        """Get the logic operator for decision rules corresponding to the given sympy operator.

        Args:
            operator (Type[Internal]): Operator to convert

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
    def get_operator_for_decision_rules(operator: LogicOperators) -> Type[Internal]:
        """Get the sympy operator corresponding to the given logic operator for decision rules.

        Args:
            operator (LogicOperators): Logic operator

        Returns:
            Type[Internal]: Operator
        """
        operators = {
            LogicOperators.CONJUNCTION: And,
            LogicOperators.ALTERNATIVE: Or,
        }
        return operators[operator]

    def rules_to_nnf(
        self, rules: list
    ) -> Tuple[List[Union[Var, Type[Internal]]], dict]:
        """Convert decision rules to nnf statements. Returns a list with rules converted
        to nnf statements and a dictionary with unique conditions. The unique conditions
        dictionary contains condition strings as keys and dictionaries with keys 'nnf'
        and 'obj' as values. The 'nnf' key contains a nnf symbol representing the condition
        and the 'obj' key contains the condition object from decision rules library.

        Args:
            rules (list): List of decision rules.

        Returns:
            Tuple[List[Union[Var, Type[Internal]]], dict]: Tuple with a list of nnf library
                statements and a dictionary with unique conditions."""
        unique_conditions = {}
        nnf_rules = []
        for rule in rules:
            nnf_rule = self._process_rules_premise(
                rule.premise, rule.column_names, unique_conditions, Var
            )
            nnf_rules.append(nnf_rule)

        return nnf_rules, unique_conditions

    def nnf_to_rules(
        self, rules: list, unique_conditions: dict
    ) -> list[AbstractCondition]:
        """Convert nnf library statements to decision rules. Assumes, that given rules are just
        premises - and so, returns also a list of premises. Conclusions should be added
        separately.

        Args:
            rules (list): List of nnf statements.
            unique_conditions (dict): Dictionary with unique conditions.

        Returns:
            list: List of premises."""
        premises = []
        for rule in rules:
            if hasattr(rule, "true") and not rule.true:
                # TODO: handle negated premises
                raise NotImplementedError("Negated premises are not supported... yet")

            premises.append(self._process_nnf_premise(rule, unique_conditions))

        return premises

    def _process_nnf_premise(
        self, premise: Union[Var, Type[Internal]], unique_conditions: dict
    ) -> AbstractCondition:
        """Recursively process premise and return a condition object.

        Args:
            premise (Union[Var, Type[Internal]]): Premise to process.
            unique_conditions (dict): Dictionary with unique conditions.

        Returns:
            AbstractCondition: Condition object."""
        if isinstance(premise, Var):
            if self._is_in_symbols(premise, unique_conditions):
                return self._get_condition_by_symbol(premise, unique_conditions)
            elif self._is_in_symbols(~premise, unique_conditions):
                condition = deepcopy(
                    self._get_condition_by_symbol(~premise, unique_conditions)
                )
                condition.negated = not condition.negated
                return condition

            raise ValueError(f"Premise {premise} not found in unique_conditions")

        operator = self.get_decision_rules_operator(type(premise))
        subconditions = [
            self._process_nnf_premise(subcondition, unique_conditions)
            for subcondition in premise.children
        ]

        if len(subconditions) == 1:
            return subconditions[0]

        return CompoundCondition(subconditions, operator)

    def split_nnf_expr(self, expression: Type[Internal]) -> Tuple[LogicOperators, list]:
        """Split a nnf expression into a list of subexpressions and the main operator.

        Args:
            expression (BooleanFunction): Sympy expression.

        Returns:
            Tuple[LogicOperators, list]: Tuple with the main operator and a list of subexpressions.
        """
        if isinstance(expression, Var):
            # in case of a single condition
            return LogicOperators.CONJUNCTION, [expression]

        operator = self.get_decision_rules_operator(type(expression))
        subexpressions = list(child for child in expression.children)
        return operator, subexpressions

    @staticmethod
    def _to_cnf_with_mapping(
        theory: NNF, simplify: bool = True, replace_aux_vars: bool = True
    ) -> And[Or[Var]]:
        """Convert an NNF into CNF using the Tseitin Encoding. Modified to replace aux variables
        on demand.
        Source: https://python-nnf.readthedocs.io/en/stable/_modules/nnf/tseitin.html#to_CNF

        Args:
            theory (NNF): The NNF to convert.
            simplify (bool): Whether to simplify the resulting CNF.
            replace_aux_vars (bool): Whether to replace auxiliary variables. If False, variables
                added via Tseitin encoding will be returned without change.

        Returns:
            And[Or[Var]]: The CNF representing the theory.
        """
        clauses = []
        aux_to_expr = {}

        @memoize
        def process_node(node: NNF) -> Var:
            nonlocal aux_to_expr

            if isinstance(node, Var):
                return node

            assert isinstance(node, Internal)

            children = {process_node(c) for c in node.children}

            if len(children) == 1:
                [child] = children
                return child

            aux = Var.aux()
            aux_to_expr[aux] = node

            if simplify and any(~var in children for var in children):
                if isinstance(node, And):
                    clauses.append(Or({~aux}))
                else:
                    clauses.append(Or({aux}))

            elif isinstance(node, And):
                clauses.append(Or({~c for c in children} | {aux}))
                for c in children:
                    clauses.append(Or({~aux, c}))

            elif isinstance(node, Or):
                clauses.append(Or(children | {~aux}))
                for c in children:
                    clauses.append(Or({~c, aux}))

            else:
                raise TypeError(node)

            return aux

        @memoize
        def process_required(node: NNF) -> None:
            """For nodes that have to be satisfied.

            This lets us perform some optimizations.
            """
            if isinstance(node, Var):
                clauses.append(Or({node}))
                return

            assert isinstance(node, Internal)

            if len(node.children) == 1:
                [child] = node.children
                process_required(child)

            elif isinstance(node, Or):
                children = {process_node(c) for c in node.children}
                if simplify and any(~v in children for v in children):
                    return
                clauses.append(Or(children))

            elif isinstance(node, And):
                for child in node.children:
                    process_required(child)

            else:
                raise TypeError(node)

        def replace_aux_in_cnf(cnf: And[Or[Var]], aux_to_expr: dict) -> And[Or[Var]]:
            """Replace auxiliary variables in CNF with their original expressions.

            Args:
                cnf (And[Or[Var]]): CNF to replace variables in.
                aux_to_expr (dict): Dictionary mapping auxiliary variables to their
                    original expressions.

            Returns:
                And[Or[Var]]: CNF with replaced variables.
            """

            def replace_clause(clause: Or[Var]) -> Or[Var]:
                def replace_var(var):
                    if not var.true:
                        # replace the negated auxiliary variable
                        original = aux_to_expr.get(~var, ~var)
                        return (
                            ~original
                            if isinstance(original, Var)
                            else original.negate()
                        )
                    return aux_to_expr.get(var, var)

                return Or({replace_var(var) for var in clause})

            return And({replace_clause(clause) for clause in cnf})

        process_required(theory)
        ret = And(clauses)

        NNF._is_CNF_loose.set(ret, True)
        NNF._is_CNF_strict.set(ret, True)

        if replace_aux_vars:
            ret = replace_aux_in_cnf(ret, aux_to_expr)

        return ret

    @staticmethod
    def negate_if_needed(
        negate: bool,
        operator: Type[Internal],
        subconditions: List[Union[Var, Type[Internal]]],
    ) -> Type[Internal]:
        """Returns a boolean operator with subconditions, negated if negate is True.

        Args:
            negate (bool): Whether to negate the operator.
            operator (Type[Internal]): Operator to negate.
            subconditions (List[Union[Var, Type[Internal]]]): List of subconditions.

        Returns:
            Type[Internal]: Operator with subconditions. For example And(A, B, C) or ~And(A, B, C).
        """
        return ~(operator({*subconditions})) if negate else operator({*subconditions})
