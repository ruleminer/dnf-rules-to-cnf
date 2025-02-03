from typing import Optional, List

import pandas as pd

from decision_rules.classification.ruleset import ClassificationRuleSet
from decision_rules.classification.rule import ClassificationRule
from decision_rules.classification.prediction_indicators import (
    calculate_for_classification,
)


def get_metrics_for_ruleset(
    ruleset: ClassificationRuleSet,
    X: pd.DataFrame,
    y: pd.Series,
    metrics_names: Optional[list] = None,
) -> list[str]:
    """Get metrics for each rule in the ruleset. Returns logs with rule, number of
    conditions and metrics chosen in metrics_names. If metrics_names is empty,
    prints all metrics.

    Args:
        ruleset (ClassificationRuleSet): Decision rules
        X (pd.DataFrame): Features
        y (pd.Series): Target
        metrics_names (list, optional): List of metrics to print. Defaults to None.
    """
    logs = []
    metrics_dict = ruleset.calculate_rules_metrics(X, y)
    for rule_id, metrics in metrics_dict.items():
        rule = get_rule_by_id(ruleset, rule_id)
        logs.append(str(rule))
        logs.append(f"\tnumber of conditions: {count_conditions(rule.premise)}")
        for name in metrics:
            if (name in metrics_names) or (metrics_names is None):
                logs.append(f"\t{name}: {metrics[name]}")
        logs.append("")
    return logs


def count_conditions(condition) -> int:
    """Recursively count the leaf subconditions.
    Source: decision_rules.core.ruleset.AbstractRuleset.calculate_ruleset_stats (inner method)

    Args:
        condition: Condition to count the leaf subconditions

    Returns:
        int: Number of leaf subconditions"""
    if not condition.subconditions:
        return 1
    return sum(
        count_conditions(subcondition) for subcondition in condition.subconditions
    )


def get_rule_by_id(ruleset: ClassificationRuleSet, rule_id: int) -> ClassificationRule:
    """Get rule by id from the ruleset.

    Args:
        ruleset (ClassificationRuleSet): Decision rules
        rule_id (int): Rule id

    Returns:
        Rule: Rule with the given id"""
    return next((rule for rule in ruleset.rules if rule._uuid == rule_id), None)


def create_full_report(
    dnf_ruleset: ClassificationRuleSet,
    cnf_ruleset: ClassificationRuleSet,
    X: pd.DataFrame,
    y: pd.Series,
    path: str = "./data/report.txt",
    rules_metrics: List[str] = ["precision", "coverage", "C2"],
    preambule: str = "",
):
    """Create a full report for DNF and CNF rulesets.

    Args:
        dnf_ruleset (ClassificationRuleSet): DNF ruleset
        cnf_ruleset (ClassificationRuleSet): CNF ruleset
        X (pd.DataFrame): Features
        y (pd.Series): Target
        path (str, optional): Path to save the report. Defaults to "./data/report.txt".
        rules_metrics (List[str], optional): List of metrics to print.
            Defaults to ["precision", "coverage", "C2"].
    """
    rules_complexity = pd.DataFrame(
        {
            "DNF_ruleset": dnf_ruleset.calculate_ruleset_stats(),
            "CNF_ruleset": cnf_ruleset.calculate_ruleset_stats(),
        }
    )

    dnf_metrics = get_metrics_for_ruleset(dnf_ruleset, X, y, rules_metrics)
    cnf_metrics = get_metrics_for_ruleset(cnf_ruleset, X, y, rules_metrics)

    y_pred_dnf = dnf_ruleset.predict(X)
    y_pred_cnf = cnf_ruleset.predict(X)
    pred_df = pd.DataFrame(
        {
            "y_true": y,
            "y_pred_dnf": y_pred_dnf,
            "y_pred_cnf": y_pred_cnf,
        }
    )

    dnf_stats = calculate_for_classification(y, y_pred_dnf)
    cnf_stats = calculate_for_classification(y, y_pred_cnf)

    general_stats = pd.DataFrame(
        {
            "DNF": dnf_stats["general"],
            "CNF": cnf_stats["general"],
        }
    ).drop(index="Confusion_matrix", errors='ignore')

    # dnf and cnf should have same keys
    classes_stats = {}
    for key in dnf_stats["for_classes"]:
        classes_stats[f"{key}_DNF"] = dnf_stats["for_classes"][key]
        classes_stats[f"{key}_CNF"] = cnf_stats["for_classes"][key]

    classes_stats = pd.DataFrame(classes_stats).drop(index="Confusion_matrix", errors='ignore')

    with open(path, "w", encoding="utf-8") as f:
        separator = 5 * "-------------------------" + "\n\n"
        f.write(preambule)
        f.write("Rules complexity:\n")
        f.write(rules_complexity.to_string())
        f.write("\n\n")
        f.write(separator)

        f.write("Metrics for DNF ruleset:\n")
        f.write("\n".join(dnf_metrics))
        f.write("\n\n")
        f.write(separator)

        f.write("Metrics for CNF ruleset:\n")
        f.write("\n".join(cnf_metrics))
        f.write("\n\n")
        f.write(separator)

        f.write("Predictions:\n")
        f.write(pred_df.to_string())
        f.write("\n\n")
        f.write(separator)

        f.write("General classification stats:\n")
        f.write(general_stats.to_string())
        f.write("\n\n")
        f.write(separator)

        f.write("Classes classification stats:\n")
        f.write(classes_stats.to_string())
        f.write("\n\n")
        f.write(separator)
