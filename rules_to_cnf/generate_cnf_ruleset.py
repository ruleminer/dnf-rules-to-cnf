from typing import Tuple
import json
import time
import os

from decision_rules.measures import c2
from decision_rules.serialization.utils import JSONSerializer
from decision_rules.classification.ruleset import ClassificationRuleSet
from decision_rules.ruleset_factories._factories.classification.rulekit_factory import (
    get_rulekit_factory_class,
)
from joblib import Parallel, delayed
import pandas as pd
from rulekit import RuleKit
from rulekit.classification import RuleClassifier
from rulekit.params import Measures

from converters.cnf_converter import CNFConverter
from converters.sympy_cnf_converter import SympyCNFConverter
from converters.nnf_cnf_converter import NnfCNFConverter
from utils.ruleset_quality_report import create_full_report


def prepare_dataset(
    path: str = "./data/datasets/iris.csv",
    class_col: str = "class",
    use_parquet: bool = False,
) -> Tuple[pd.DataFrame, pd.Series]:
    """Prepare iris dataset. Return X, y.

    Args:
        path (str, optional): Path to the dataset. Defaults to "./data/iris.csv".
        class_col (str, optional): Name of the target column. Defaults to "class".
        use_parquet (bool, optional): Whether to use parquet format. Defaults to False.

    Returns:
        Tuple[pd.DataFrame, pd.Series]: X, y
    """
    df = pd.read_csv(path) if not use_parquet else pd.read_parquet(path)
    X = df.drop(class_col, axis=1)
    y = df[class_col]
    return X, y


def create_rulekit_ruleset(
    X: pd.DataFrame,
    y: pd.Series,
    max_rule_count: int = 5,
    max_growing: int = 4,
) -> RuleClassifier:
    """Create a RuleKit ruleset from X, y.

    Args:
        X (pd.DataFrame): Features
        y (pd.Series): Target
        max_rule_count (int, optional): Maximum number of rules in the ruleset per class.
            Defaults to 5. 0 means no limit.
        max_growing (int, optional): Maximum number of growing. Specifies max number of
            conditions in rule. Defaults to 4. 0 means no limit.

    Returns:
        RuleClassifier: RuleKit rules"""
    RuleKit.init()
    c2_measure = Measures.C2
    rulekit_model = RuleClassifier(
        induction_measure=c2_measure,
        pruning_measure=c2_measure,
        voting_measure=c2_measure,
        max_rule_count=max_rule_count,
        max_growing=max_growing,
    )
    rulekit_model.fit(X, y)
    return rulekit_model


def convert_rulekit_to_decision_rules(
    ruleset: RuleClassifier, X: pd.DataFrame, y: pd.Series
) -> ClassificationRuleSet:
    """Convert RuleKit ruleset to decision rules.

    Args:
        ruleset (RuleClassifier): RuleKit ruleset
        X (pd.DataFrame): Features
        y (pd.Series): Target

    Returns:
        ClassificationRuleSet: Decision rules
    """
    factory = get_rulekit_factory_class()
    return factory().make(ruleset, X, y)


def save_ruleset(ruleset: ClassificationRuleSet, path: str):
    """Save ruleset to a file.

    Args:
        ruleset (ClassificationRuleSet): Decision rules
        path (str): Path to save the ruleset
    """
    ruleset_json = JSONSerializer.serialize(ruleset)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(ruleset_json, f)


def load_ruleset(path: str) -> ClassificationRuleSet:
    """Load ruleset from a file.

    Args:
        path (str): Path to load the ruleset

    Returns:
        ClassificationRuleSet: Decision rules
    """
    with open(path, "r", encoding="utf-8") as f:
        ruleset_json = json.load(f)
    return JSONSerializer.deserialize(ruleset_json, target_class=ClassificationRuleSet)


def single_cnf_report(
    converter: CNFConverter,
    file: str,
    datasets_dir: str,
    ruleset_dir: str,
    reports_dir: str,
    max_rule_count: int,
    max_growing: int,
    pred_strategy: str = "vote",
):
    """Generate DNF and CNF rulesets for a single dataset file.

    Args:
        converter (CNFConverter): CNF converter subclass
        file (str): Dataset file name
        datasets_dir (str): Directory with dataset files
        ruleset_dir (str): Directory to save ruleset files
        reports_dir (str): Directory to save report files
        max_rule_count (int): Maximum number of rules in the ruleset per class.
            0 means no limit. Used in rulekit.
        max_growing (int): Maximum number of growing. Specifies max number of
            conditions in rule. 0 means no limit. Used in rulekit.
        pred_strategy (str, optional): Prediction strategy to use for both rulesets.
            Defaults to "vote".
    """
    dataset_name = file.split(".")[0]
    full_path = os.path.join(datasets_dir, file)

    start = time.time()

    print(f"Creating DNF ruleset from {file}...", flush=True)
    X, y = prepare_dataset(full_path, use_parquet=file.endswith(".parquet"))
    rulekit_model = create_rulekit_ruleset(
        X=X, y=y, max_rule_count=max_rule_count, max_growing=max_growing
    )
    dnf_ruleset = convert_rulekit_to_decision_rules(rulekit_model, X, y)
    dnf_ruleset.set_prediction_strategy(pred_strategy)
    save_ruleset(dnf_ruleset, os.path.join(ruleset_dir, f"{dataset_name}_dnf.json"))
    print("Done\n", flush=True)

    print(f"Creating CNF ruleset from {file}...", flush=True)
    cnf_ruleset = converter.convert_to_cnf(dnf_ruleset)
    # !!necessary step!!
    cnf_ruleset.update(X, y, c2)
    cnf_ruleset.set_prediction_strategy(pred_strategy)
    save_ruleset(cnf_ruleset, os.path.join(ruleset_dir, f"{dataset_name}_cnf.json"))
    print("Done\n", flush=True)

    print(f"Creating report from {file}...", flush=True)
    create_full_report(
        dnf_ruleset=dnf_ruleset,
        cnf_ruleset=cnf_ruleset,
        X=X,
        y=y,
        path=os.path.join(reports_dir, f"{dataset_name}_report.txt"),
        preambule=(
            f"Dataset: {dataset_name}\nmax_rule_count: {max_rule_count}\n"
            f"max_growing: {max_growing}\n\n"
        ),
    )
    print("Done\n", flush=True)

    end = time.time()
    print(f"Time for {file}: {end - start:.2f} seconds", flush=True)


def generate_cnf_reports(
    converter: CNFConverter, dir_suffix: str, data_dir: str, max_rule_count: int = 5, max_growing: int = 4, n_jobs: int = 1
):
    """Generate DNF and CNF rulesets for all csv and parquet files listed in data_dir/datasets
    directory.

    Args:
        converter (CNFConverter): an object used to convert ruleset
        dir_suffix (str): suffix added to rulesets and reports dir in order to distinguish between converters
        data_dir (str): Directory with dataset and ruleset files
        max_rule_count (int, optional): Maximum number of rules in the ruleset per class.
            Defaults to 5. 0 means no limit. Used in rulekit.
        max_growing (int, optional): Maximum number of growing. Specifies max number of
            conditions in rule. Defaults to 4. 0 means no limit. Used in rulekit.
        n_jobs (int, optional): Number of jobs to run in parallel. Defaults to 1. If -1,
            then the number of jobs is set to the number of CPU cores.
    """
    ruleset_dir = os.path.join(data_dir, f"rulesets_{dir_suffix}")
    reports_dir = os.path.join(data_dir, f"reports_{dir_suffix}")
    os.makedirs(ruleset_dir, exist_ok=True)
    os.makedirs(reports_dir, exist_ok=True)

    datasets_dir = os.path.join(data_dir, "datasets")
    if not os.path.exists(datasets_dir):
        raise FileNotFoundError("No datasets directory found.")

    dataset_files = [
        f
        for f in os.listdir(datasets_dir)
        if f.endswith(".csv") or f.endswith(".parquet")
    ]
    print("Dataset files found:", dataset_files, flush=True)
    if not dataset_files:
        raise FileNotFoundError(
            "No CSV or Parquet files found in the specified directory."
        )

    # prediction strategy to use for both rulesets
    PRED_STRATEGY = "vote"

    total_start = time.time()
    print(
        f"Starting generation of DNF and CNF rulesets for {len(dataset_files)} datasets\n"
        f"max_rule_count: {max_rule_count}\n",
        f"max_growing: {max_growing}\n",
        f"Prediction strategy: {PRED_STRATEGY}\n",
        f"Number of jobs: {n_jobs}\n",
        flush=True,
    )

    Parallel(n_jobs=n_jobs)(
        delayed(single_cnf_report)(
            converter,
            file,
            datasets_dir,
            ruleset_dir,
            reports_dir,
            max_rule_count,
            max_growing,
            PRED_STRATEGY,
        )
        for file in dataset_files
    )

    total_end = time.time()
    print(f"Total time: {total_end - total_start:.2f} seconds", flush=True)


if __name__ == "__main__":
    DATA_DIR = (
        "/home/bpigula/decision-rules/data/"
    )
    MAX_RULE_COUNT = 5
    MAX_GROWING = 4
    N_JOBS = 8

    converter = NnfCNFConverter()
    DIR_SUFFIX = "nnf"

    # converter = SympyCNFConverter()
    # DIR_SUFFIX = "sympy"

    generate_cnf_reports(converter, DIR_SUFFIX, DATA_DIR, MAX_RULE_COUNT, MAX_GROWING, N_JOBS)
