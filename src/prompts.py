"""Library module (imported, not run directly) — prompt templates.

Prompt templates for the prompt-engineering benchmark (Experiments 2-6).

Five styles, all returning the SAME strict JSON schema so they are directly
comparable. The only thing that changes between experiments is HOW we ask the
model to think.

Styles:
  zero_shot     -> Exp 2  : plain instruction, no examples, no reasoning
  few_shot      -> Exp 3  : a few labelled examples per category
  cot           -> Exp 4  : forced step-by-step reasoning before the label
  persona       -> Exp 5  : role-conditioned ("you are a moderator")
  few_shot_cot  -> Exp 6  : examples + step-by-step reasoning combined

Few-shot examples are hand-written (NOT drawn from the dataset) so there is
zero risk of leaking eval tweets into the prompt.
"""
from config import LABELS, LABEL_DESCRIPTIONS

# Canonical, machine-parseable output contract shared by every prompt.
OUTPUT_SCHEMA = (
    "Respond with ONLY a JSON object (no markdown, no extra text) of the form:\n"
    "{\n"
    '  "label": one of ' + str(LABELS) + ",\n"
    '  "target_group": the group targeted (e.g. "women", "Muslims", "elderly") or "none",\n'
    '  "intent": "harmful" or "neutral",\n'
    '  "intensity": "none" | "low" | "medium" | "high",\n'
    '  "explanation": one concise sentence justifying the label\n'
    "}"
)

CATEGORY_BLOCK = "Categories:\n" + "\n".join(
    f"- {lab}: {desc}" for lab, desc in LABEL_DESCRIPTIONS.items()
)

# Hand-written few-shot exemplars (synthetic, illustrative, no dataset leakage).
FEW_SHOT_EXAMPLES = [
    ('"Women like you should just stay in the kitchen, you are useless."',
     '{"label": "gender", "target_group": "women", "intent": "harmful", "intensity": "high", "explanation": "Demeans women with a sexist stereotype and insult."}'),
    ('"All muslims are terrorists and should be banned."',
     '{"label": "religion", "target_group": "Muslims", "intent": "harmful", "intensity": "high", "explanation": "Hateful generalization attacking a religious group."}'),
    ('"Go back to your own country you dirty immigrant."',
     '{"label": "ethnicity", "target_group": "immigrants", "intent": "harmful", "intensity": "high", "explanation": "Xenophobic slur targeting ethnicity/nationality."}'),
    ('"Stupid old geezer cant even use a phone, just retire already."',
     '{"label": "age", "target_group": "elderly", "intent": "harmful", "intensity": "medium", "explanation": "Ageist mockery of an older person."}'),
    ('"Nobody likes you, you are a worthless loser, just disappear."',
     '{"label": "other_cyberbullying", "target_group": "an individual", "intent": "harmful", "intensity": "high", "explanation": "Personal harassment not tied to a protected trait."}'),
    ('"Thanks so much for the help today, you really made my week!"',
     '{"label": "not_cyberbullying", "target_group": "none", "intent": "neutral", "intensity": "none", "explanation": "Friendly, supportive message with no abuse."}'),
]


def _few_shot_block() -> str:
    lines = ["Here are labelled examples:"]
    for tweet, ans in FEW_SHOT_EXAMPLES:
        lines.append(f"Tweet: {tweet}\nAnswer: {ans}")
    return "\n\n".join(lines)


def build_prompt(style: str, tweet: str) -> tuple[str | None, str]:
    """Return (system_prompt, user_prompt) for the given style and tweet."""
    base_task = (
        "Classify the following tweet into exactly one cyberbullying category.\n\n"
        + CATEGORY_BLOCK + "\n\n" + OUTPUT_SCHEMA
    )

    if style == "zero_shot":
        system = "You are a precise text classifier."
        user = f"{base_task}\n\nTweet: \"{tweet}\""
        return system, user

    if style == "few_shot":
        system = "You are a precise text classifier."
        user = f"{base_task}\n\n{_few_shot_block()}\n\nNow classify this tweet:\nTweet: \"{tweet}\""
        return system, user

    if style == "cot":
        system = "You are a careful content-moderation reasoner."
        user = (
            f"{base_task}\n\n"
            "Think step by step BEFORE answering:\n"
            "1. Who or what group is being addressed or referenced?\n"
            "2. What is the tone and language style (insulting, sarcastic, neutral)?\n"
            "3. Is there harmful intent toward a person or group?\n"
            "4. If harmful, which protected trait (religion/age/gender/ethnicity) or is it general?\n"
            "Reason internally, then output ONLY the final JSON object.\n\n"
            f"Tweet: \"{tweet}\""
        )
        return system, user

    if style == "persona":
        system = (
            "You are an experienced social-media content-moderation officer who also "
            "advises school counsellors. You are fair, context-aware, and protective of "
            "vulnerable groups. You never over-flag friendly banter, but you reliably "
            "catch coded slurs and targeted harassment."
        )
        user = f"{base_task}\n\nAs the moderator, classify this tweet:\nTweet: \"{tweet}\""
        return system, user

    if style == "few_shot_cot":
        system = "You are a careful content-moderation reasoner."
        user = (
            f"{base_task}\n\n{_few_shot_block()}\n\n"
            "For the new tweet, reason step by step (target, tone, intent, trait) "
            "internally, then output ONLY the final JSON object.\n\n"
            f"Tweet: \"{tweet}\""
        )
        return system, user

    raise ValueError(f"Unknown prompt style: {style}")


STYLES = ["zero_shot", "few_shot", "cot", "persona", "few_shot_cot"]
# Maps each style to the experiment number in the research plan.
STYLE_TO_EXP = {
    "zero_shot": "exp2",
    "few_shot": "exp3",
    "cot": "exp4",
    "persona": "exp5",
    "few_shot_cot": "exp6",
}
