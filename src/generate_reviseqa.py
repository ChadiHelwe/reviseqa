import json
import os
import random
import re
import subprocess
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy
from dataclasses import asdict

import instructor
from dotenv import load_dotenv
from instructor import Mode
from nltk.inference import Prover9Command
from nltk.sem.logic import Expression
from openai import OpenAI

from src.data_structure import KB
from src.prompt_engine import (
    assumptions_modification_prompting,
    backward_reasoning_prompting,
    edits_made_prompting,
    forward_reasoning_prompting,
    generate_prover9_input_prompting,
    invariant_modification_prompting,
    uncertain_modification_prompting,
)
from src.prover import FOL2Prover9Converter, prover9_prove
from src.utils import (
    extract_all_facts,
    extract_all_predicates,
    extract_all_rules,
    extract_context_facts_rules_conclusion_answer,
    extract_reasoning_chains,
    prover9_input_to_str,
    read_file,
)

load_dotenv()


def make_kb_reasoning_chain(example):
    predicates = extract_all_predicates(example)
    all_facts = extract_all_facts(predicates, example)
    all_rules = extract_all_rules(predicates, example)

    (
        context_facts,
        context_rules,
        context_conclusion,
        initial_answer,
    ) = extract_context_facts_rules_conclusion_answer(all_facts, all_rules, example)
    reasoning_chains = extract_reasoning_chains(all_facts, all_rules, example)

    example_kb = KB(
        predicates=predicates,
        all_facts=all_facts,
        all_rules=all_rules,
        context_facts=context_facts,
        context_rules=context_rules,
        context=context_facts + context_rules,
        conclusion=context_conclusion,
        background_story=example["background_story"],
        subject_name=example["name"],
        subject_category=example["subject_category"],
        keyword=example["keyword"],
    )

    return example_kb, reasoning_chains, initial_answer


def match_checking_initial_answer_with_prover(kb, answer, prover):
    try:
        context_fol = []
        context_str_fol = []

        for item in kb.context:
            context_fol.append(item.fol)
            context_str_fol.append(item.str_fol)

        conclusion_fol = kb.conclusion.fol
        conclusion_str_fol = kb.conclusion.str_fol

        p9_context_fol, p9_conclusion_fol = prover.convert_fol_instance(
            context_fol, conclusion_fol
        )
        p9_context_str_fol, p9_conclusion_str_fol = prover.convert_fol_instance(
            context_str_fol, conclusion_str_fol
        )
        prover_answer_fol = prover9_prove(p9_context_fol, p9_conclusion_fol)
        prover_answer_str_fol = prover9_prove(p9_context_str_fol, p9_conclusion_str_fol)

        if (
            answer.upper() == prover_answer_fol.upper()
            and answer.upper() == prover_answer_str_fol.upper()
        ):
            return True

    except Exception as e:
        print(e)

    return False


def __extract_assumptions(proof_text: str):
    """
    Parses Prover9 proof output and returns a list of formulas marked as assumptions.
    """
    pattern = re.compile(r"^\s*\d+\s+(.+?)\.\s+\[assumption\]", re.MULTILINE)
    raw = pattern.findall(proof_text)
    cleaned = []
    for f in raw:
        f = f.strip()
        # Remove surrounding parentheses if present
        if f.startswith("(") and f.endswith(")"):
            f = f[1:-1].strip()
        f = f.replace("(", "")
        f = f.replace(")", "")
        cleaned.append(f)
    return cleaned


def __separate_facts_rules(assumptions):
    """
    Splits assumptions into 'facts' (atomic) and 'rules' (contain quantifiers or connectives).
    """
    facts, rules = [], []
    for formula in assumptions:
        # Identify rules by the presence of quantifiers or logical connectives
        if re.search(r"\b(all|exists)\b|->|\||&|<->", formula):
            rules.append(formula)
        else:
            facts.append(formula)

    context = facts + rules
    return context, facts, rules


def __extract_supported_facts_rules(kb, support_strs, converter):
    support_items = [
        item
        for item in kb.context
        if converter.convert_expression(item.fol)
        .rstrip(".")
        .replace(".", "")
        .replace("not ", "-")
        .replace("(", "")
        .replace(")", "")
        in support_strs[0]
    ]

    non_used_items = [
        item
        for item in kb.context
        if converter.convert_expression(item.fol)
        .rstrip(".")
        .replace(".", "")
        .replace("not ", "-")
        .replace("(", "")
        .replace(")", "")
        not in support_strs[0]
    ]

    return support_items, non_used_items


def __extract_clean_prover9_proof(output):
    start_marker = (
        "============================== PROOF ==============================="
    )
    end_marker = (
        "============================== end of proof =========================="
    )

    start_idx = output.find(start_marker)
    end_idx = output.find(end_marker)

    if start_idx == -1 or end_idx == -1:
        return "Proof section not found."

    proof_block = output[start_idx + len(start_marker) : end_idx].strip()
    clean_lines = [
        line for line in proof_block.splitlines() if not line.strip().startswith("%")
    ]

    return "\n".join(clean_lines).strip()


def __extract_proof_trace(raw):
    proof_lines = []
    in_trace = False
    for line in raw.splitlines():
        # skip comments
        if line.lstrip().startswith("%"):
            continue

        # detect the first proof‐step (a line beginning with a number)
        if not in_trace:
            if re.match(r"^\s*\d+\s", line):
                in_trace = True
                proof_lines.append(line)
        else:
            # stop if we hit the end‐of‐proof marker
            if re.match(r"^=+ end of proof", line, re.IGNORECASE):
                break
            # otherwise keep collecting any line that looks like a step
            if re.match(r"^\s*\d+\s", line):
                proof_lines.append(line)

    return "\n".join(proof_lines)


def prove_with_reasoning_facts_rules(kb):
    converter = FOL2Prover9Converter()
    p9_assumps = [
        Expression.fromstring(converter.convert_expression(item.fol).rstrip("."))
        for item in kb.context
    ]

    goal_pos = Expression.fromstring(
        converter.convert_expression(kb.conclusion.fol).rstrip(".")
    )
    cmd_pos = Prover9Command(goal_pos, assumptions=p9_assumps)
    pos_ok = cmd_pos.prove()
    proof_pos = cmd_pos.proof(True)

    assumptions_proof_pos = __extract_assumptions(proof_pos)
    support_strs = __separate_facts_rules(assumptions_proof_pos)

    support_items, non_support_items = __extract_supported_facts_rules(
        kb, support_strs, converter
    )

    goal_neg = Expression.fromstring(
        f'-({converter.convert_expression(kb.conclusion.fol).rstrip(".")})'
    )
    cmd_neg = Prover9Command(goal_neg, assumptions=p9_assumps)
    neg_ok = cmd_neg.prove()
    proof_neg = cmd_neg.proof(True)

    assumptions_proof_neg = __extract_assumptions(proof_neg)
    refute_strs = __separate_facts_rules(assumptions_proof_neg)
    refute_items, non_refute_items = __extract_supported_facts_rules(
        kb, refute_strs, converter
    )

    non_used_items = non_support_items + non_refute_items
    # print("Support Items:", support_items)
    # print("Refute Items:", refute_items)

    assumptions = "\n".join(str(i) for i in cmd_pos.assumptions())
    if pos_ok == neg_ok:
        return (
            "Uncertain",
            support_items,
            refute_items,
            non_used_items,
            None,
            assumptions,
        )
    elif pos_ok:
        return (
            "True",
            support_items,
            refute_items,
            non_used_items,
            __extract_proof_trace(proof_pos),
            assumptions,
        )
    elif neg_ok:
        return (
            "False",
            support_items,
            refute_items,
            non_used_items,
            __extract_proof_trace(proof_neg),
            assumptions,
        )


def is_theorem_proved(output: str) -> bool:
    return bool(re.search(r"^\s*THEOREM PROVED\s*$", output, re.MULTILINE))


def run_prover9_cmd(prover9_input):
    try:
        with tempfile.NamedTemporaryFile(delete=False) as temp_input_file:
            temp_input_file.write(prover9_input.encode())
            temp_file_path = temp_input_file.name

        # Run Prover9 command
        prover9_cmd = ["prover9", "-f", temp_file_path]
        prover9_process = subprocess.run(prover9_cmd, capture_output=True, text=True)

        # if prover9_process.returncode != 0:
        #     raise Exception(f"Prover9 command failed: {prover9_process.stderr}")

        prover9_output = prover9_process.stdout
        return prover9_output

    except Exception as e:
        if "Prover9 command failed" in str(e):
            print(str(e))
            print("Prover9 command failed.")
            return None
        else:
            print(f"An error occurred: {str(e)}")
            return None

    finally:
        try:
            os.remove(temp_file_path)
        except OSError as e:
            print(f"Error deleting temporary file {temp_file_path}: {e}")


def make_dataset(data):
    prover = FOL2Prover9Converter()
    client = OpenAI(
        api_key=os.environ["OPENROUTER_API_KEY"],
        base_url="https://openrouter.ai/api/v1",  # OpenRouter endpoint
    )
    instructor_client = instructor.from_openai(
        client, mode=Mode.OPENROUTER_STRUCTURED_OUTPUTS
    )

    data = read_file(data)
    for example_nbr, example in enumerate(data):
        example_kb, initial_reasoning_chain, initial_answer = make_kb_reasoning_chain(
            example
        )

        matched = match_checking_initial_answer_with_prover(
            example_kb, initial_answer, prover
        )

        if matched:
            (
                prover_answer,
                _,
                _,
                _,
                proof_trace,
                assumptions,
            ) = prove_with_reasoning_facts_rules(example_kb)

            assumptions = [fact.fol for fact in example_kb.context_facts]
            assumptions += [rule.fol for rule in example_kb.context_rules]
            assumptions = "\n".join(assumptions)

            initial_goal = example_kb.conclusion.fol
            neg_goal = f"¬({example_kb.conclusion.fol})"

            print("#" * 20)
            print("Assumptions:", assumptions)
            print("Initial Goal:", initial_goal)
            print("Negated Goal:", neg_goal)
            print("Initial Answer", initial_answer)
            print("#" * 20)

            reasoning_chains_asdict = []

            for item in initial_reasoning_chain:
                print(item)
                if item["conclusion"] is None:
                    conclusion = None
                else:
                    conclusion = asdict(item["conclusion"])
                item_dict = {
                    "facts": [asdict(fact) for fact in item["facts"]],
                    "rules": [asdict(rule) for rule in item["rules"]],
                    "conclusion": conclusion,
                }
                reasoning_chains_asdict.append(item_dict)

            kb_example = {
                "background_story": example["background_story"],
                "predicates": [asdict(item) for item in example_kb.predicates],
                "subject_name": example["name"],
                "subject_category": example["subject_category"],
                "keyword": example["keyword"],
                "all_facts": [asdict(item) for item in example_kb.all_facts],
                "context_facts": [asdict(item) for item in example_kb.context_facts],
                "context_rules": [asdict(item) for item in example_kb.context_rules],
                "context": [asdict(item) for item in example_kb.context],
                "context_fol": assumptions,
                "conclusion": asdict(example_kb.conclusion),
                "initial_answer": initial_answer,
                "initial_goal": initial_goal,
                "reasoning_chain": reasoning_chains_asdict,
                "proof_trace": proof_trace,
                "edits_made": [],
            }
            all_edits_example = []
            for i in range(1, 8):

                if initial_answer == "Uncertain":
                    modification_type = "UNCERTAIN"
                    edited_prove9_input, edited_fol_input, edits_made = (
                        assumptions_modifier(
                            assumptions,
                            initial_goal,
                            neg_goal,
                            client,
                            instructor_client,
                            modification_type="UNCERTAIN",
                        )
                    )
                    initial_answer = "True"
                else:
                    modification_type = random.choice(["FLIP", "INVARIANT"])
                    print(modification_type)
                    edited_prove9_input, edited_fol_input, edits_made = (
                        assumptions_modifier(
                            assumptions,
                            initial_goal,
                            neg_goal,
                            client,
                            instructor_client,
                            modification_type=modification_type,
                        )
                    )

                    if edits_made is not None:
                        assumptions = "\n".join(
                            edited_fol_input["formulas(assumptions)"]
                        )

                        if modification_type == "FLIP":
                            neg_goal, initial_goal = initial_goal, neg_goal
                            if initial_answer == "True":
                                initial_answer = "False"
                            else:
                                initial_answer = "True"

                if edits_made is not None:
                    print("Edit#:", i)
                    print("Edited Assumptions:", assumptions)
                    print("Initial Goal:", initial_goal)
                    print("Negated Goal:", neg_goal)
                    print("Initial Answer:", initial_answer)
                    print("Edits Made", edits_made)

                    all_edits_example.append(
                        {
                            "Edit#": i,
                            "Modification Type": modification_type,
                            "Edited Assumptions": assumptions,
                            "Initial Goal": initial_goal,
                            "Negated Goal": neg_goal,
                            "Initial Answer": initial_answer,
                            "Edited Prover9 Input": edited_prove9_input,
                            "Edits Made": edits_made.model_dump(),
                        }
                    )

            kb_example["edits_made"] = all_edits_example

            with open(f"reviseqa_data/fol/ex_{example_nbr}.json", "w") as f:
                json.dump(kb_example, f, indent=4)


def assumptions_modifier(
    assumptions,
    initial_goal,
    neg_goal,
    client,
    instructor_client,
    retries=5,
    modification_type="FLIP",
):
    edits_made = None
    edited_fol_input = None
    edited_prover9_input = None

    for _ in range(retries):
        history = []
        initial_history = []
        initial_assumptions = assumptions
        try:
            if modification_type == "FLIP":
                response, history = assumptions_modification_prompting(
                    assumptions, initial_goal, neg_goal, client, history
                )
                initial_history = deepcopy(history)
            elif modification_type == "UNCERTAIN":
                response, history = uncertain_modification_prompting(
                    assumptions, initial_goal, neg_goal, client, history
                )
                initial_history = deepcopy(history)
            else:
                response, history = invariant_modification_prompting(
                    assumptions, initial_goal, neg_goal, client, history
                )
                initial_history = deepcopy(history)

            for _ in range(retries):
                try:
                    if modification_type == "FLIP":
                        response, history = generate_prover9_input_prompting(
                            initial_goal, neg_goal, instructor_client, history
                        )
                    else:
                        response, history = generate_prover9_input_prompting(
                            neg_goal, initial_goal, instructor_client, history
                        )

                    if response is not None:
                        edited_fol_input = response.fol_input
                        edited_prover9_input = response.prover9_input

                        updated_assumptions = "\n".join(
                            edited_fol_input["formulas(assumptions)"]
                        )
                        out = prover9_input_to_str(response.prover9_input)

                        neg_prover9_input = response.prover9_input.copy()
                        neg_prover9_input["formulas(goals)"] = [
                            f"-({item})"
                            for item in neg_prover9_input["formulas(goals)"]
                        ]
                        neg_out = prover9_input_to_str(neg_prover9_input)

                        # print(out)
                        # print(neg_out)

                        prover9_output = run_prover9_cmd(out)
                        neg_prover9_output = run_prover9_cmd(neg_out)

                        if prover9_output is None or neg_prover9_output is None:
                            break

                        theorem_proved = is_theorem_proved(prover9_output)
                        neg_theorem_proved = is_theorem_proved(neg_prover9_output)

                        if theorem_proved and neg_theorem_proved:
                            proof = __extract_clean_prover9_proof(neg_prover9_output)
                            print("Both theorems proved.")

                            if modification_type == "FLIP":
                                _, history = backward_reasoning_prompting(
                                    proof,
                                    initial_goal,
                                    neg_goal,
                                    client,
                                    initial_history,
                                )
                            else:
                                _, history = backward_reasoning_prompting(
                                    proof,
                                    neg_goal,
                                    initial_goal,
                                    client,
                                    initial_history,
                                )

                            initial_history = deepcopy(history)
                            # print(initial_history)
                            continue
                        elif not theorem_proved:
                            print("Theorem not proved.")
                            # print(prover9_output)
                            if modification_type == "FLIP":
                                _, history = forward_reasoning_prompting(
                                    initial_goal, neg_goal, client, initial_history
                                )
                            else:
                                _, history = forward_reasoning_prompting(
                                    neg_goal, initial_goal, client, initial_history
                                )

                            initial_history = deepcopy(history)
                            # print(initial_history)
                            continue
                        else:
                            print("Theorem proved")

                            for _ in range(retries):
                                try:
                                    print("--" * 20)
                                    print("Initial Assumptions")
                                    print(initial_assumptions)
                                    print("#" * 20)
                                    print("Updated Assumptions")
                                    print(updated_assumptions)
                                    print("--" * 20)

                                    edits_made = edits_made_prompting(
                                        initial_assumptions,
                                        updated_assumptions,
                                        instructor_client,
                                        history,
                                    )
                                    break
                                except Exception as e:
                                    print(f"Error: {e}")
                                    print("Retrying...")
                                    history.append(
                                        {
                                            "role": "user",
                                            "content": "You generated an invalid JSON. Please retry and output a valid JSON only, with no extra explanation or text.",
                                        }
                                    )
                                    edits_made = None
                                    continue

                        # exiting the inner loop
                        if edits_made is not None:
                            break

                except Exception as e:
                    print(f"Error: {e}")
                    print("Retrying...")
                    history.append(
                        {
                            "role": "user",
                            "content": "You generated an invalid JSON. Please retry and output a valid JSON only, with no extra explanation or text.",
                        }
                    )
                    continue

            # exiting the outer loop
            if edits_made is not None:
                break
        except Exception as e:
            print(f"Error: {e}")
            print("Retrying...")

    return edited_prover9_input, edited_fol_input, edits_made


def process_example(example_nbr, example, output_dir="reviseqa_data/fol"):
    print("Example #", example_nbr)
    prover = FOL2Prover9Converter()
    client = OpenAI(
        api_key=os.environ["OPENROUTER_API_KEY"],
        base_url="https://openrouter.ai/api/v1",  # OpenRouter endpoint
    )
    instructor_client = instructor.from_openai(
        client, mode=Mode.OPENROUTER_STRUCTURED_OUTPUTS
    )

    example_kb, initial_reasoning_chain, initial_answer = make_kb_reasoning_chain(
        example
    )

    matched = match_checking_initial_answer_with_prover(
        example_kb, initial_answer, prover
    )

    if not matched:
        return
    else:
        (
            prover_answer,
            _,
            _,
            _,
            proof_trace,
            assumptions,
        ) = prove_with_reasoning_facts_rules(example_kb)

        assumptions = [fact.fol for fact in example_kb.context_facts]
        assumptions += [rule.fol for rule in example_kb.context_rules]
        assumptions = "\n".join(assumptions)

        initial_goal = example_kb.conclusion.fol
        neg_goal = f"¬({example_kb.conclusion.fol})"

        the_goal = example_kb.conclusion.fol
        print("#" * 20)
        print("Assumptions:", assumptions)
        print("Initial Goal:", the_goal)
        # print("Negated Goal:", neg_goal)
        print("Initial Answer", initial_answer)
        print("#" * 20)

        reasoning_chains_asdict = []

        for item in initial_reasoning_chain:
            if item["conclusion"] is None:
                conclusion = None
            else:
                conclusion = asdict(item["conclusion"])
            item_dict = {
                "facts": [asdict(fact) for fact in item["facts"]],
                "rules": [asdict(rule) for rule in item["rules"]],
                "conclusion": conclusion,
            }
            reasoning_chains_asdict.append(item_dict)

        kb_example = {
            "background_story": example["background_story"],
            "predicates": [asdict(item) for item in example_kb.predicates],
            "subject_name": example["name"],
            "subject_category": example["subject_category"],
            "keyword": example["keyword"],
            "all_facts": [asdict(item) for item in example_kb.all_facts],
            "context_facts": [asdict(item) for item in example_kb.context_facts],
            "context_rules": [asdict(item) for item in example_kb.context_rules],
            "context": [asdict(item) for item in example_kb.context],
            "context_fol": assumptions,
            "conclusion": asdict(example_kb.conclusion),
            "initial_answer": initial_answer,
            "initial_goal": initial_goal,
            "reasoning_chain": reasoning_chains_asdict,
            "proof_trace": proof_trace,
            "edits_made": [],
        }
        all_edits_example = []

        if initial_answer == "False":
            initial_goal, neg_goal = neg_goal, initial_goal

        for i in range(1, 8):

            if initial_answer == "Uncertain":
                modification_type = "UNCERTAIN"
                edited_prove9_input, edited_fol_input, edits_made = (
                    assumptions_modifier(
                        assumptions,
                        initial_goal,
                        neg_goal,
                        client,
                        instructor_client,
                        modification_type="UNCERTAIN",
                    )
                )
                initial_answer = "True"
            else:
                modification_type = random.choice(["FLIP", "INVARIANT"])
                print(modification_type)
                edited_prove9_input, edited_fol_input, edits_made = (
                    assumptions_modifier(
                        assumptions,
                        initial_goal,
                        neg_goal,
                        client,
                        instructor_client,
                        modification_type=modification_type,
                    )
                )

                if edits_made is not None:
                    assumptions = "\n".join(edited_fol_input["formulas(assumptions)"])

                    if modification_type == "FLIP":
                        neg_goal, initial_goal = initial_goal, neg_goal
                        if initial_answer == "True":
                            initial_answer = "False"
                        else:
                            initial_answer = "True"

            if edits_made is not None:
                print("Edit#:", i)
                print("Edited Assumptions:", assumptions)
                print("Goal:", the_goal)
                # print("Negated Goal:", neg_goal)
                print("Answer:", initial_answer)
                print("Edits Made", edits_made)

                all_edits_example.append(
                    {
                        "Edit#": i,
                        "Modification Type": modification_type,
                        "Edited Assumptions": assumptions,
                        "Initial Goal": the_goal,
                        # "Previous Goal": initial_goal,
                        # "Negated Goal": neg_goal,
                        "Answer": initial_answer,
                        "Edited Prover9 Input": edited_prove9_input,
                        "Edits Made": edits_made.model_dump(),
                    }
                )

        kb_example["edits_made"] = all_edits_example

        with open(f"{output_dir}/ex_{example_nbr}.json", "w") as f:
            json.dump(kb_example, f, indent=4)


def parallel_make_dataset(data_path):
    data = read_file(data_path)
    # verified_data = []
    output_dir = "reviseqa_data/fol"
    os.makedirs(output_dir, exist_ok=True)

    # for f in os.listdir("reviseqa_data/nl/xor_verified"):
    #     id_file = int(f.split(".")[0].replace("ex_", ""))
    #     verified_data.append(id_file)

    # Use as many workers as you have CPU cores (or fewer, if you’re IO‐bound)
    with ProcessPoolExecutor() as executor:
        futures = {
            executor.submit(process_example, i, example, output_dir): i
            for i, example in enumerate(data)
        }

        for future in as_completed(futures):
            i = futures[future]
            try:
                future.result()
                print(f"✅ Example {i} done")
            except Exception as e:
                print(f"❌ Example {i} raised {e!r}")
