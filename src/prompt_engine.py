import openai
from dotenv import load_dotenv
from jinja2 import Template
from pydantic import BaseModel, ConfigDict

load_dotenv()


# MODEL = "openai/o4-mini-high"
MODEL = "deepseek/deepseek-prover-v2"
# MODEL_OUTPUT_GENERATION = "anthropic/claude-3.7-sonnet"
MODEL_OUTPUT_GENERATION = "deepseek/deepseek-prover-v2"


class Prover9InputStructure(BaseModel):
    prover9_input: dict
    fol_input: dict


class UpdateStructure(BaseModel):
    removed_facts: list
    removed_rules: list
    added_facts: list
    added_rules: list


SYSTEM_UNCERTAIN_MODIFICATION_PROMPT = """
"You are an expert in formal logic and Prover9. You will receive:

1. A list of assumptions (facts and rules).  
2. A target goal G, which is currently not provable.  
3. The negation of the goal ¬G, which is currently not provable.

**Your objective** is **to edit the assumptions** so that G becomes provable, but the negation ¬G remains unprovable.

Your task is to produce a **non-simple minimal edit set** that:

• **Adds** only the minimal number of new facts or rules (based strictly on the original assumptions) to derive G.  
• **Does not enable** the derivation of ¬G in any form.  
• **Never adds** G as a standalone fact, but you may add a rule that implies G.
• **You are encouraged to use non-obvious but logically derivable rules** — that is, rules that are logically entailed by the assumptions but not explicitly stated.  
• **Ensures**  that after editing:
    1. Use forward reasoning to derive G.  
    2. Use backward reasoning from ¬G fails (¬G remains not derivable).

**Step-by-step approach**:
• **Analyze** the assumptions and verify that G is not currently provable.
• **Introduce** minimal additions (based on existing assumptions) to make G provable.
• **Avoid** any additions or changes that could imply or derive ¬G.
• **Validate**:
   1. Use forward reasoning to prove G.
   2. Use backward reasoning from ¬G to confirm it is not provable.
"""


SYSTEM_MODIFICATION_PROMPT = """
You are an expert in formal logic and Prover9. You will receive:

1. A list of assumptions (facts and rules).  
2. An initial goal G, which is provable.
3. The negation of the goal ¬G, which is currently not provable.

**Your objective** is to **edit the assumptions** so that ¬G becomes provable (but G is not).

Your task is to produce a **non-simple minimal edit set** that:

• **Removes** only those facts or rules that directly enable a proof of G.  
• **Negates** existing facts or rules as needed - never add ¬G as a standalone fact, though you may add a rule implying ¬G.  
• **Adds** only the smallest number of new facts or rules (all derived from original assumptions) to **indirectly** derive ¬G.  
• **You are encouraged to use non-obvious but logically derivable rules** — rules not explicitly present but inferable from the existing assumptions — to construct a valid proof of ¬G.  
• **Ensures** that after editing:  
  1. Use forward reasoning to derive ¬G.  
  2. Use backward reasoning from G fails (G is no longer derivable).

**Step-by-step approach**:
1. **Analyze** the assumptions and confirm that G is provable and ¬G is not.  
2. **Break** G’s derivation by removing or negating the minimal set of those assumptions.  
3. **Introduce** new facts or rules—based on original assumptions—to build a proof of ¬G.  
4. **Validate**:  
   - Use forward reasoning to prove ¬G.  
   - Use backward reasoning from G to confirm it cannot be derived.
"""

SYSTEM_INVARIANT_MODIFICATION_PROMPT = """
You are an expert in formal logic and Prover9. You will receive:

1. A list of assumptions (facts and rules).  
2. An initial goal G, which is currently provable.  
3. The negation of the goal ¬G, which is currently not provable.

**Your objective** is to edit the assumptions in a minimal way that preserves the current reasoning outcome:
– G must remain provable.  
– ¬G must remain unprovable.

Your task is to produce a **non-simple minimal edit set** that:

• **Adds** logically redundant facts or rules that are already entailed by the original assumptions.  
• **Rewrites** existing rules into logically equivalent forms (e.g., implication distribution, de Morgan’s laws, etc.) without changing the semantics. 
• **You are encouraged to use non-obvious but logically derivable rules** — statements that don’t change the deductive power but offer alternative expressions.  
• **Does not** remove any assumption, reorder rules, or rename variables.  
• **Does not** introduce any new path that could derive ¬G or block the proof of G.

**Ensure** that after editing:
   1. Forward reasoning still derives G.  
   2. Backward reasoning from ¬G still fails (¬G remains not derivable).

**Step-by-step approach**:
1. **Analyze** the assumptions and confirm that G is provable and ¬G is not.  
2. **Apply** logically neutral and semantically safe edits as described above.  
3. **Validate**:
   - Use forward reasoning to prove G.  
   - Use backward reasoning from ¬G to confirm it cannot be derived.

"""

ASSUMPTION_PROMPT = Template(
    """
#####################################################
Initial Assumptions:
{{assumptions}}

Initial Goal G: 
{{initial_goal}}

#####################################################
Goal ¬G:
{{neg_goal}}

"""
)

FORWARD_REASONING_PROMPT = Template(
    """
Prover9 was not able to prove {{neg_goal}}, which is the intended goal. 
Perform forward reasoning using the current assumptions to determine why {{neg_goal}} is not derivable. 
Identify the missing, conflicting, or overly permissive facts and rules that prevent the proof. 
Then, propose **non-simple minimal edits** (i.e., avoid directly inserting the goal) to the assumptions that would allow {{neg_goal}} to be correctly derived.
After applying the edits, re-run forward reasoning to confirm that {{neg_goal}} is now provable and that {{initial_goal}} is no longer derivable
"""
)


BACKWARD_REASONING_PROMPT_VERSION_0 = Template(
    """
Prover9 was able to prove {{initial_goal}}, but this result is incorrect — it should not be provable. 
Perform backward reasoning starting from {{initial_goal}} to identify which current assumptions led to its derivation. 
Trace the proof steps and locate the specific facts or rules responsible. 
Then, propose **non-simple minimal edits** (i.e., inserting the negation directly) to the assumptions to prevent {{initial_goal}} from being provable.
After applying the edits, perform forward reasoning using the updated assumptions to confirm that {{initial_goal}} is no longer derivable, but that {{neg_goal}} is provable.
"""
)

BACKWARD_REASONING_PROMPT_VERSION_1 = Template(
    """
Prover9 Proof:
{{ proof }}

Prover9 was able to prove {{initial_goal}}, but this result is incorrect — it should not be provable.

Your task is to:
1. Carefully examine the Prover9 proof to identify the sequence of steps that led to the derivation of {{initial_goal}}.
2. Use **backward reasoning**, starting from {{initial_goal}}, to trace which facts and rules were used at each step and which assumptions directly or indirectly contributed to the conclusion.
3. Based on your analysis, propose a set of **non-simple minimal edits** to the assumptions that would break the proof chain — i.e., prevent {{initial_goal}} from being provable — **without simply deleting the goal or inserting its negation as a fact**.
   - You may negate or adjust specific rules.
   - You may remove enabling assumptions.
   - You may introduce new rules that imply {{neg_goal}}, but not {{initial_goal}}.

4. Finally, perform **forward reasoning** with the updated assumptions to confirm that:
   - {{initial_goal}} is **no longer provable**.
   - {{neg_goal}} is **now derivable**.

Think step-by-step, and make sure your proposed changes are both logically valid and minimal.
"""
)


PROVER9_INPUT_GENERATION_PROMPT = Template(
    """
Once you have applied the **non-simple minimal edits**, generate a valid Prover9 input along with its equivalent FOL syntax using the updated assumptions.
**Do not remove or add any rules or facts beyond those specified in the non-simple minimal edits.**

Your output must include:
- A complete `formulas(assumptions).` block with the updated facts and rules.
- A `formulas(goals).` block with the new goal: {{neg_goal}}.
- Ensure that the original goal {{initial_goal}} is not included anywhere in the assumptions or goals.

Please provide the output in a **valid** JSON format, structured as follows:
{
    "prover9_input": {
        "formulas(assumptions)": ["¬p_1(Novah)", "all x (p_3(x) -> p_2(x))", ...],
        "formulas(goals)": ["¬p_2(Novah)",...]
    }
    "fol_input": {
        "formulas(assumptions)": ["¬p_1(Novah)", "∀x (p_3(x) → p_2(x))", ...],
        "formulas(goals)": ["¬p_2(Novah)",...]
    }
}

For the `fol_input`, use the following standard **FOL symbols**:
- Universal quantifier: `∀`
- Existential quantifier: `∃` (only if logically necessary)
- Negation: `¬`
- Conjunction: `∧`
- Disjunction: `∨`
- Implication: `→`
- Exclusive disjunction: `⊕`
- Biconditional: `↔`

If an expression uses XOR (⊕) in the FOL block (e.g., `p(x) ⊕ q(x)`), translate it in the Prover9 block as `-(p(x) <-> q(x))`.  
Always enclose the entire negated biconditional in parentheses when it appears inside larger expressions, like conjunctions or implications.  
For example:

- FOL: `(p(x) ⊕ q(x)) ∧ r(x)`  
- Prover9: `(-(p(x) <-> q(x))) & r(x)`

**Do not** use Prover9 syntax (e.g., `all`, `->`) in the `fol_input`. Think step by step and make sure both inputs are consistent.
"""
)


EDITS_MADE_PROMPT = Template(
    """

Initial Assumptions:
{{initial_assumptions}}

Updated Assumptions:
{{updated_assumptions}}

Based on the initial and updated assumptions in this conversation history, compare the initial and modified assumptions and output a JSON object with exactly these four keys:
{
    "removed_facts": [¬p_3(Bob), p4(Alice),...],
    "removed_rules: [∀x (p_3(x) → p_2(x)),...],
    "added_facts": [p_2(Kevin),...],
    "added_rules": [∃x (p_3(x) → p_2(x)),...]
}

Use the following standard **FOL symbols** inside each string:
- Universal quantifier: `∀`
- Existential quantifier: `∃` (only if logically necessary)
- Negation: `¬`
- Conjunction: `∧`
- Disjunction: `∨`
- Implication: `→`
- Exclusive disjunction: `⊕`
- Biconditional: `↔`

All values must be strings representing individual facts or rules in valid FOL syntax.
**Do not** include explanations or any additional fields. **Only return the JSON object as shown**.
"""
)


def generate_prover9_input_prompting(
    initital_goal, neg_goal, instructor_client, history
):

    prover9_input_generation_prompt = PROVER9_INPUT_GENERATION_PROMPT.render(
        initial_goal=initital_goal, neg_goal=neg_goal
    )
    history.append({"role": "user", "content": prover9_input_generation_prompt})
    response = instructor_client.chat.completions.create(
        model=MODEL_OUTPUT_GENERATION,
        temperature=0.7,
        response_model=Prover9InputStructure,
        messages=history,
        max_retries=3,
    )

    return response, history


def edits_made_prompting(
    initial_assumptions, updated_assumptions, instructor_client, history
):
    user_prompt = EDITS_MADE_PROMPT.render(
        initial_assumptions=initial_assumptions,
        updated_assumptions=updated_assumptions,
    )
    history.append({"role": "user", "content": user_prompt})

    response = instructor_client.chat.completions.create(
        model=MODEL_OUTPUT_GENERATION,
        temperature=0.7,
        messages=history,
        max_retries=3,
        response_model=UpdateStructure,
    )

    # The model's reply should be pure JSON
    return response


def assumptions_modification_prompting(
    assumptions, initial_goal, neg_goal, client, history
):
    user_prompt = ASSUMPTION_PROMPT.render(
        assumptions=assumptions, initial_goal=initial_goal, neg_goal=neg_goal
    )

    history.append({"role": "system", "content": SYSTEM_MODIFICATION_PROMPT})
    history.append({"role": "user", "content": user_prompt})

    response = client.chat.completions.create(
        extra_body={"provider": {"sort": "throughput"}},
        model=MODEL,
        temperature=0.7,
        messages=history,
    )

    response = response.choices[0].message.content
    history.append({"role": "assistant", "content": response})

    return response, history


def backward_reasoning_prompting(proof, initial_goal, neg_goal, client, history):
    print("RETRY with BACKWARD REASONING")

    user_prompt = BACKWARD_REASONING_PROMPT_VERSION_1.render(
        proof=proof, initial_goal=initial_goal, neg_goal=neg_goal
    )
    history.append({"role": "user", "content": user_prompt})
    response = client.chat.completions.create(
        extra_body={"provider": {"sort": "throughput"}},
        model=MODEL,
        temperature=0.7,
        messages=history,
    )

    response = response.choices[0].message.content
    # print("RETRY with BACKWARD REASONING:", response)

    history.append({"role": "assistant", "content": response})

    return response, history


def forward_reasoning_prompting(neg_goal, initial_goal, client, history):
    """
    Generates a prompt for forward reasoning in a logic system.
    """
    print("RETRY with FORWARD REASONING")
    user_prompt = FORWARD_REASONING_PROMPT.render(
        neg_goal=neg_goal, initial_goal=initial_goal
    )

    history.append({"role": "user", "content": user_prompt})

    response = client.chat.completions.create(
        extra_body={"provider": {"sort": "throughput"}},
        model=MODEL,
        temperature=0.7,
        messages=history,
    )

    response = response.choices[0].message.content
    # print("RETRY with FORWARD REASONING:", response)
    history.append({"role": "assistant", "content": response})

    return response, history


def uncertain_modification_prompting(
    assumptions, initial_goal, neg_goal, client, history
):
    user_prompt = ASSUMPTION_PROMPT.render(
        assumptions=assumptions, initial_goal=initial_goal, neg_goal=neg_goal
    )

    history.append({"role": "system", "content": SYSTEM_UNCERTAIN_MODIFICATION_PROMPT})
    history.append({"role": "user", "content": user_prompt})

    response = client.chat.completions.create(
        extra_body={"provider": {"sort": "throughput"}},
        model=MODEL,
        temperature=0.7,
        messages=history,
    )

    response = response.choices[0].message.content
    history.append({"role": "assistant", "content": response})

    return response, history


def invariant_modification_prompting(
    assumptiuons, initial_goal, neg_goal, client, history
):
    user_prompt = ASSUMPTION_PROMPT.render(
        assumptions=assumptiuons, initial_goal=initial_goal, neg_goal=neg_goal
    )

    history.append({"role": "system", "content": SYSTEM_INVARIANT_MODIFICATION_PROMPT})
    history.append({"role": "user", "content": user_prompt})

    response = client.chat.completions.create(
        extra_body={"provider": {"sort": "throughput"}},
        model=MODEL,
        temperature=0.7,
        messages=history,
    )

    response = response.choices[0].message.content
    history.append({"role": "assistant", "content": response})

    return response, history
