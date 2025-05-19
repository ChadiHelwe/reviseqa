import json
import re

from ordered_set import OrderedSet

from src.data_structure import Fact, Predicate, Rule


def read_file(file_path):
    """
    Read a JSON file and return its content.

    Args:
        file_path (str): The path to the JSON file.

    Returns:
        dict: The content of the JSON file.
    """
    with open(file_path, "r") as file:
        data = json.load(file)
    return data


TRIGGER_WORDS_RULE = {
    "if",
    "then",
    "either",
    "or",
    "both",
    "unless",
    "implies",
    "for all",
    "every",
    "any",
    "and",
    "all",
    "some",
}

RULES_SYMBOLS = {
    "→",
    "⇒",
    "↔",
    "⇔",
    "∧",
    "∨",
    "∀",
    "∃",
    "⊕",
}


# modified
def is_rule(expression, expression_fol):
    # expr = expression.lower()
    # tokens = expr.split()
    if any(symbol in expression_fol for symbol in RULES_SYMBOLS):
        # print(expression_fol)
        return True
    # return any(token in TRIGGER_WORDS_RULE for token in tokens)


def extract_predicates(fol_expression):
    bound_vars = set(re.findall(r"[∀∃]\s*([A-Za-z_]\w*)", fol_expression))
    candidates = re.findall(r"([A-Za-z_]\w*)\s*\(", fol_expression)
    return [p for p in candidates if p not in bound_vars]


def extract_context_facts_rules_conclusion_answer(all_facts, all_rules, example):
    facts = []
    rules = []
    conclusion = None
    answer = None

    for expression, expression_fol in zip(example["context"], example["context_fol"]):
        if is_rule(expression, expression_fol):
            for rule_obj in all_rules:
                if rule_obj.text == expression:
                    rules.append(rule_obj)
                    break
        else:
            for fact_obj in all_facts:
                if fact_obj.text == expression:
                    facts.append(fact_obj)
                    break

    conclusion = example["conclusion"]

    if is_rule(conclusion, example["conclusion_fol"]):
        for rule_obj in all_rules:
            if rule_obj.text == conclusion:
                conclusion = rule_obj
                break
    else:
        for fact_obj in all_facts:
            if fact_obj.text == conclusion:
                conclusion = fact_obj
                break
    answer = example["answer"]

    return facts, rules, conclusion, answer


def extract_reasoning_chains(all_facts, all_rules, example):
    reasoning_chains = []

    for reasoning_chain, reasoning_chain_fol in zip(
        example["reasoning_chains"], example["reasoning_chains_fol"]
    ):
        facts = []
        rules = []
        conclusion = None

        if reasoning_chain["facts"] is not None:
            for fact in reasoning_chain["facts"]:
                for fact_obj in all_facts:
                    if fact_obj.text == fact:
                        facts.append(fact_obj)
                        break

        if reasoning_chain["rules"] is not None:
            rule = reasoning_chain["rules"]
            for rule_obj in all_rules:
                if rule_obj.text == rule:
                    rules.append(rule_obj)
                    break

        if reasoning_chain["conclusion"] is not None:
            conclusion = reasoning_chain["conclusion"]
            if is_rule(conclusion, reasoning_chain_fol["conclusion"]):
                for rule_obj in all_rules:
                    if rule_obj.text == conclusion:
                        conclusion = rule_obj
                        break
            else:
                for fact_obj in all_facts:
                    if fact_obj.text == conclusion:
                        conclusion = fact_obj
                        break
        reasoning_chains.append(
            {
                "facts": facts,
                "rules": rules,
                "conclusion": conclusion,
            }
        )

    return reasoning_chains


def extract_all_predicates(example):
    predicates = OrderedSet()
    pred_cnt = 0

    for fact_fol in example["facts_fol"]:
        extracted_preds = extract_predicates(fact_fol)

        for pred in extracted_preds:
            if Predicate(pred_cnt, pred) not in predicates:
                predicates.add(Predicate(pred_cnt, pred))
                pred_cnt += 1

    for rule_fol in example["rules_fol"]:
        extracted_preds = extract_predicates(rule_fol)

        for pred in extracted_preds:
            if Predicate(pred_cnt, pred) not in predicates:
                predicates.add(Predicate(pred_cnt, pred))
                pred_cnt += 1

    conclusion_fol = example["conclusion_fol"]
    extracted_preds = extract_predicates(conclusion_fol)
    for pred in extracted_preds:
        if Predicate(pred_cnt, pred) not in predicates:
            predicates.add(Predicate(pred_cnt, pred))
            pred_cnt += 1

    for distracting_fact_fol in example["distracting_facts_fol"]:
        extracted_preds = extract_predicates(distracting_fact_fol)

        for pred in extracted_preds:
            if Predicate(pred_cnt, pred) not in predicates:
                predicates.add(Predicate(pred_cnt, pred))
                pred_cnt += 1

    for distracting_rule_fol in example["distracting_rules_fol"]:
        extracted_preds = extract_predicates(distracting_rule_fol)

        for pred in extracted_preds:
            if Predicate(pred_cnt, pred) not in predicates:
                predicates.add(Predicate(pred_cnt, pred))
                pred_cnt += 1

    for reasoning_chain in example["reasoning_chains"]:
        if reasoning_chain["facts"] is not None:
            for fact_fol in reasoning_chain["facts"]:
                extracted_preds = extract_predicates(fact_fol)

                for pred in extracted_preds:
                    if Predicate(pred_cnt, pred) not in predicates:
                        predicates.add(Predicate(pred_cnt, pred))
                        pred_cnt += 1

        if reasoning_chain["rules"] is not None:
            for rule_fol in reasoning_chain["rules"]:
                extracted_preds = extract_predicates(rule_fol)

                for pred in extracted_preds:
                    if Predicate(pred_cnt, pred) not in predicates:
                        predicates.add(Predicate(pred_cnt, pred))
                        pred_cnt += 1

        if reasoning_chain["conclusion"] is not None:
            conclusion_fol = reasoning_chain["conclusion"]
            extracted_preds = extract_predicates(conclusion_fol)
            for pred in extracted_preds:
                if Predicate(pred_cnt, pred) not in predicates:
                    predicates.add(Predicate(pred_cnt, pred))
                    pred_cnt += 1

    return predicates


def __extract_fact(predicate, subject, fact, fact_fol, fact_cnt):
    negation = False
    if "¬" in fact_fol:
        negation = True

    return Fact(
        fact_cnt,
        subject,
        fact,
        fact_fol.replace(predicate.name, f"p_{predicate.id}"),
        fact_fol,
        negation,
    )


def extract_all_facts(predicates, example):
    facts = OrderedSet()
    fact_cnt = 0

    subject = example["name"]

    for fact, fact_fol in zip(example["facts"], example["facts_fol"]):
        for predicate in predicates:
            if predicate.name in fact_fol:
                extracted_fact = __extract_fact(
                    predicate, subject, fact, fact_fol, fact_cnt
                )
                if extracted_fact not in facts:
                    facts.add(extracted_fact)
                    fact_cnt += 1
                break

    for distracting_fact, distracting_fact_fol in zip(
        example["distracting_facts"], example["distracting_facts_fol"]
    ):
        for predicate in predicates:
            if predicate.name in distracting_fact_fol:
                extracted_fact = __extract_fact(
                    predicate, subject, distracting_fact, distracting_fact_fol, fact_cnt
                )
                if extracted_fact not in facts:
                    facts.add(extracted_fact)
                    fact_cnt += 1
                break

    for reasoning_chain, reasoning_chain_fol in zip(
        example["reasoning_chains"], example["reasoning_chains_fol"]
    ):
        if reasoning_chain["facts"] is not None:
            for fact, fact_fol in zip(
                reasoning_chain["facts"], reasoning_chain_fol["facts"]
            ):
                for predicate in predicates:
                    if predicate.name in fact_fol:
                        extracted_fact = __extract_fact(
                            predicate, subject, fact, fact_fol, fact_cnt
                        )
                        if extracted_fact not in facts:
                            facts.add(extracted_fact)
                            fact_cnt += 1
                        break

        if reasoning_chain["conclusion"] is not None:
            conclusion_fol = reasoning_chain_fol["conclusion"]
            for predicate in predicates:
                if predicate.name in conclusion_fol:
                    extracted_fact = __extract_fact(
                        predicate,
                        subject,
                        reasoning_chain["conclusion"],
                        conclusion_fol,
                        fact_cnt,
                    )
                    if extracted_fact not in facts:
                        facts.add(extracted_fact)
                        fact_cnt += 1
                    break

    conclusion = example["conclusion"]
    conlusion_fol = example["conclusion_fol"]

    if not is_rule(conclusion, conlusion_fol):
        for predicate in predicates:
            if predicate.name in conlusion_fol:
                extracted_fact = __extract_fact(
                    predicate, subject, conclusion, conlusion_fol, fact_cnt
                )
                if extracted_fact not in facts:
                    facts.add(extracted_fact)
                    fact_cnt += 1
                break

    return facts


def extract_all_rules(predicates, example):
    rules = OrderedSet()
    rule_cnt = 0

    for rule, rule_fol in zip(example["rules"], example["rules_fol"]):
        tmp_rule_fol = rule_fol
        for predicate in predicates:
            if predicate.name in rule_fol:
                tmp_rule_fol = tmp_rule_fol.replace(predicate.name, f"p_{predicate.id}")

        if is_rule(rule, rule_fol):
            extracted_rule = Rule(rule_cnt, rule, tmp_rule_fol, rule_fol)
            if extracted_rule not in rules:
                rules.add(extracted_rule)
                rule_cnt += 1

    for distracting_rule, distracting_rule_fol in zip(
        example["distracting_rules"], example["distracting_rules_fol"]
    ):
        tmp_rule_fol = distracting_rule_fol
        for predicate in predicates:
            if predicate.name in distracting_rule_fol:
                tmp_rule_fol = tmp_rule_fol.replace(predicate.name, f"p_{predicate.id}")

        if is_rule(distracting_rule, distracting_rule_fol):
            extracted_rule = Rule(
                rule_cnt, distracting_rule, tmp_rule_fol, distracting_rule_fol
            )
            if extracted_rule not in rules:
                rules.add(extracted_rule)
                rule_cnt += 1

    for reasoning_chain, reasoning_chain_fol in zip(
        example["reasoning_chains"], example["reasoning_chains_fol"]
    ):
        if reasoning_chain["rules"] is not None:
            rule, rule_fol = reasoning_chain["rules"], reasoning_chain_fol["rules"]

            tmp_rule_fol = rule_fol
            for predicate in predicates:
                if predicate.name in rule_fol:
                    tmp_rule_fol = tmp_rule_fol.replace(
                        predicate.name, f"p_{predicate.id}"
                    )

            if is_rule(rule, rule_fol):
                extracted_rule = Rule(rule_cnt, rule, tmp_rule_fol, rule_fol)
                if extracted_rule not in rules:
                    rules.add(extracted_rule)
                    rule_cnt += 1

    conclusion = example["conclusion"]
    conlusion_fol = example["conclusion_fol"]

    if is_rule(conclusion, conlusion_fol):
        tmp_rule_fol = conlusion_fol
        for predicate in predicates:
            if predicate.name in conlusion_fol:
                tmp_rule_fol = tmp_rule_fol.replace(predicate.name, f"p_{predicate.id}")

        extracted_rule = Rule(rule_cnt, conclusion, tmp_rule_fol, conlusion_fol)
        if extracted_rule not in rules:
            rules.add(extracted_rule)
            rule_cnt += 1

    return rules


def prover9_input_to_str(data_dict):
    """
    Generates a Prover9 input file from a dictionary containing formulas.

    Parameters:
    - data_dict (dict): Dictionary with keys like 'formulas(assumptions)' and 'formulas(goals)',
                        each mapping to a string of formulas separated by newlines.
    - output_file_path (str): Path to the output .in file for Prover9.
    """
    output_str = ""
    for section in ["formulas(assumptions)", "formulas(goals)"]:
        if section in data_dict:
            output_str += f"{section}.\n"
            if isinstance(data_dict[section], str):
                formulas = data_dict[section].strip().split("\n")
            else:
                formulas = data_dict[section]
            for formula in formulas:
                formula = formula.strip()
                if formula and not formula.endswith("."):
                    formula += "."
                formula = formula.replace("¬", "-")
                output_str += f"  {formula}\n"
            output_str += "end_of_list.\n\n"

    return output_str
