import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import List

import instructor
from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel

from src.utils import read_file

load_dotenv()


class Element(BaseModel):
    fol: str
    nl: str


class EditsMade(BaseModel):
    removed_facts: List[Element]
    removed_rules: List[Element]
    added_facts: List[Element]
    added_rules: List[Element]


class Edit(BaseModel):
    edit_number: int
    edited_context_fol: List[str]
    edited_natural_language_context: List[str]
    edits_made: EditsMade


class OutputSchema(BaseModel):
    original_context: List[str]
    original_context_fol: List[str]
    conclusion: str
    conclusion_fol: str
    edits: List[Edit]


class ConsistencyError(Exception):
    """Base class for application-specific errors."""

    pass


MODEL = "anthropic/claude-3.7-sonnet"

FOL_TO_NL_PROMPT = """
You are a Formal Logic, FOL-to-Natural-Language Translation Expert and JSON Schema Specialist.

When you respond, follow these instructions precisely:

Given:
1. Subject name (string)
2. Subject category (string)
3. Context with its original FOL formulas
4. Conclusion sentence with its FOL formula
5. A sequence of edits, each defined by:
   - edit_number (integer)
   - edited_context_fol: full list of FOL formulas after the edit (array of strings)
   - edits_made: four lists (removed_facts, removed_rules, added_facts, added_rules), each containing FOL strings

Your job:
- Translate **every** FOL formula in edited_context_fol into a corresponding natural-language (NL) sentence, preserving the original meaning.
- Translate **every** FOL string in each of the four edits_made lists into a natural-language (NL) description.

Output:
A single JSON object matching this schema exactly (no additional keys or text):

json
{
  "original_context": [string, ...]
  "original_context_fol": [string, ...],
  "conclusion": string,
  "conclusion_fol" string,
  "edits": [
    {
      "edit_number": integer,
      "edited_context_fol": [string, ...],
      "edited_natural_language_context": [string, ...],
      "edits_made": {
        "removed_facts":   [{"fol": string, "nl": string}, ...],
        "removed_rules":   [{"fol": string, "nl": string}, ...],
        "added_facts":     [{"fol": string, "nl": string}, ...],
        "added_rules":     [{"fol": string, "nl": string}, ...]
      }
    },
    ...
  ]
}

Once you have generated every `{ "fol": ..., "nl": ... }` pair, perform a review pass:
  1. Confirm each `nl` sentence accurately and completely expresses the original `fol`.
  2. Ensure no formula is omitted or combined with another.
  3. If you spot any mismatch or omission, correct the `nl` so it matches the `fol` exactly.


The output must be a valid JSON matching the schema—no extra keys and no prose outside the JSON.
"""


def __read_input(example_data):

    output_lines = []
    output_lines.append(f"Subject Name: {example_data['subject_name']}")
    output_lines.append(f"Subject Category: {example_data['subject_category']}")
    output_lines.append("")
    output_lines.append("#" * 20)
    output_lines.append("Initial Context:")

    # Context
    for i in example_data["context"]:
        output_lines.append(f"\ntext: {i['text']}")
        output_lines.append(f"fol: {i['fol']}")
        output_lines.append(f"str_fol: {i['str_fol']}")

    output_lines.append("")
    output_lines.append("#" * 20)
    output_lines.append("Conclusion:")
    concl = example_data["conclusion"]
    output_lines.append(f"text: {concl['text']}")
    output_lines.append(f"fol: {concl['fol']}")
    output_lines.append(f"str_fol: {concl['str_fol']}")
    output_lines.append(f"Answer: {example_data['initial_answer']}")
    output_lines.append("")
    output_lines.append("#" * 20)

    # Edits
    for edit in example_data["edits_made"]:
        output_lines.append(f"Edit#: {edit['Edit#']}")
        output_lines.append(f"\nEdited Context FOL: {edit['Edited Assumptions']}")
        output_lines.append("#" * 20)
        output_lines.append("\nEdits Made:")
        # Removed Facts
        output_lines.append("\nRemoved Facts")
        if edit["Edits Made"]["removed_facts"]:
            output_lines.extend(edit["Edits Made"]["removed_facts"])
        else:
            output_lines.append("No facts removed")
            # Removed Rules
        output_lines.append("\nRemoved Rules")
        if edit["Edits Made"]["removed_rules"]:
            output_lines.extend(edit["Edits Made"]["removed_rules"])
        else:
            output_lines.append("No rules removed")
            # Added Facts
        output_lines.append("\nAdded Facts")
        if edit["Edits Made"]["added_facts"]:
            output_lines.extend(edit["Edits Made"]["added_facts"])
        else:
            output_lines.append("No facts added")
            # Added Rules
        output_lines.append("\nAdded Rules")
        if edit["Edits Made"]["added_rules"]:
            output_lines.extend(edit["Edits Made"]["added_rules"])
        else:
            output_lines.append("No rules added")
        output_lines.append("")

    return "\n".join(output_lines)


# def make_dataset_nl(path, instructor_client):

#     for example in os.listdir(path):
#         history = []
#         example_path = os.path.join(path, example)
#         example_data = read_file(example_path)
#         user_input = __read_input(example_data)

#         history.append({"role": "system", "content": FOL_TO_NL_PROMPT})
#         history.append({"role": "user", "content": user_input})

#         try:
#             for _ in range(5):
#                 response = instructor_client.chat.completions.create(
#                     model=MODEL,
#                     temperature=0.7,
#                     messages=history,
#                     max_retries=3,
#                     response_model=OutputSchema,
#                 )
#                 response = response.model_dump()
#                 if len(response["edits"]) != len(example_data["edits_made"]):
#                     raise ConsistencyError(
#                         "The number of edits in the JSON do not match the number of edits in the user input "
#                     )

#                 break

#         except ConsistencyError as e:
#             print(e)
#             print("Retry")
#             history.append({"role": "assistant", "content": response})
#             history.append(
#                 {
#                     "content": "user",
#                     "content": "The number of edits in the generated JSON do not match the number of edits in the user input. Please retry and output a valid JSON only, with no extra explanation or text",
#                 }
#             )
#         except Exception as e:
#             print(e)
#             print("Retry")
#             history.append(
#                 {
#                     "role": "user",
#                     "content": "You generated an invalid JSON. Please retry and output a valid JSON only, with no extra explanation or text.",
#                 }
#             )

#         output_json = {
#             "original_context": response["original_context"],
#             "original_context_fol": response["original_context_fol"],
#             "conclusion": example_data["conclusion"]["text"],
#             "conclusion_fol": example_data["conclusion"]["fol"],
#             "answer": example_data["initial_answer"],
#             "reasoning_chain": example_data["reasoning_chain"],
#             "edits": [],
#         }

#         for json_edit, edit in zip(response["edits"], example_data["edits_made"]):
#             output_edit = {
#                 "edit_number": json_edit["edit_number"],
#                 "modification_type": edit["Modification Type"],
#                 "edited_context_fol": json_edit["edited_context_fol"],
#                 "edited_natural_language_context": json_edit[
#                     "edited_natural_language_context"
#                 ],
#                 "edits_made": {
#                     "removed_facts": [],
#                     "removed_rules": [],
#                     "added_facts": [],
#                     "added_rules": [],
#                 },
#                 "conclusion": example_data["conclusion"]["text"],
#                 "conclusion_fol": example_data["conclusion"]["fol"],
#                 "prove9_input": edit["Edited Prover9 Input"],
#                 "answer": edit["Initial Answer"],
#             }

#             for i in json_edit["edits_made"]["removed_facts"]:
#                 output_edit["edits_made"]["removed_facts"].append(
#                     {"fol": i["fol"], "nl": i["nl"]}
#                 )

#             for i in json_edit["edits_made"]["removed_rules"]:
#                 output_edit["edits_made"]["removed_rules"].append(
#                     {"fol": i["fol"], "nl": i["nl"]}
#                 )

#             for i in json_edit["edits_made"]["added_facts"]:
#                 output_edit["edits_made"]["added_facts"].append(
#                     {"fol": i["fol"], "nl": i["nl"]}
#                 )

#             for i in json_edit["edits_made"]["added_rules"]:
#                 output_edit["edits_made"]["added_rules"].append(
#                     {"fol": i["fol"], "nl": i["nl"]}
#                 )

#             output_json["edits"].append(output_edit)

#         with open(os.path.join("reviseqa_data/nl", example), "w") as f:
#             json.dump(output_json, f, indent=4)


def process_example(path, example):

    client = OpenAI(
        api_key=os.environ["OPENROUTER_API_KEY"],
        base_url="https://openrouter.ai/api/v1",
    )

    instructor_client = instructor.from_openai(
        client, mode=instructor.Mode.OPENROUTER_STRUCTURED_OUTPUTS
    )

    history = []
    example_path = os.path.join(path, example)
    example_data = read_file(example_path)
    user_input = __read_input(example_data)

    history.append({"role": "system", "content": FOL_TO_NL_PROMPT})
    history.append({"role": "user", "content": user_input})

    error = False
    for _ in range(5):
        try:
            response = instructor_client.chat.completions.create(
                model=MODEL,
                temperature=0.7,
                messages=history,
                max_retries=3,
                response_model=OutputSchema,
            )
            response = response.model_dump()
            if len(response["edits"]) != len(example_data["edits_made"]):
                raise ConsistencyError(
                    "The number of edits in the JSON do not match the number of edits in the user input "
                )
            error = False
            break

        except ConsistencyError as e:
            print(e)
            print("Retry")
            history.append({"role": "assistant", "content": response})
            history.append(
                {
                    "content": "user",
                    "content": "The number of edits in the generated JSON do not match the number of edits in the user input. Please retry and output a valid JSON only, with no extra explanation or text",
                }
            )
            error = True
        except Exception as e:
            print(e)
            print("Retry")
            history.append(
                {
                    "role": "user",
                    "content": "You generated an invalid JSON. Please retry and output a valid JSON only, with no extra explanation or text.",
                }
            )
            error = True

    if error:
        raise Exception("LLM wasn't able to generate a correct JSON file")

    output_json = {
        "original_context": response["original_context"],
        "original_context_fol": response["original_context_fol"],
        "conclusion": example_data["conclusion"]["text"],
        "conclusion_fol": example_data["conclusion"]["fol"],
        "answer": example_data["initial_answer"],
        "reasoning_chain": example_data["reasoning_chain"],
        "edits": [],
    }

    # print(response)
    # print(len(response["edits"]))
    # print(len(example_data["edits_made"]))
    for json_edit, edit in zip(response["edits"], example_data["edits_made"]):
        output_edit = {
            "edit_number": json_edit["edit_number"],
            "modification_type": edit["Modification Type"],
            "edited_context_fol": json_edit["edited_context_fol"],
            "edited_natural_language_context": json_edit[
                "edited_natural_language_context"
            ],
            "edits_made": {
                "removed_facts": [],
                "removed_rules": [],
                "added_facts": [],
                "added_rules": [],
            },
            "conclusion": example_data["conclusion"]["text"],
            "conclusion_fol": example_data["conclusion"]["fol"],
            "prover9_input": edit["Edited Prover9 Input"],
            "answer": edit["Answer"],
        }

        for i in json_edit["edits_made"]["removed_facts"]:
            output_edit["edits_made"]["removed_facts"].append(
                {"fol": i["fol"], "nl": i["nl"]}
            )

        for i in json_edit["edits_made"]["removed_rules"]:
            output_edit["edits_made"]["removed_rules"].append(
                {"fol": i["fol"], "nl": i["nl"]}
            )

        for i in json_edit["edits_made"]["added_facts"]:
            output_edit["edits_made"]["added_facts"].append(
                {"fol": i["fol"], "nl": i["nl"]}
            )

        for i in json_edit["edits_made"]["added_rules"]:
            output_edit["edits_made"]["added_rules"].append(
                {"fol": i["fol"], "nl": i["nl"]}
            )

        output_json["edits"].append(output_edit)

    with open(os.path.join("reviseqa_data/nl", example), "w") as f:
        json.dump(output_json, f, indent=4)


def parallel_make_dataset_nl(data_path):

    with ProcessPoolExecutor() as executor:
        futures = {
            executor.submit(
                process_example,
                data_path,
                example,
            ): example
            for example in os.listdir(data_path)
        }

        for future in as_completed(futures):
            i = futures[future]
            try:
                future.result()
                print(f"✅ Example {i} done")
            except Exception as e:
                print(f"❌ Example {i} raised {e!r}")
