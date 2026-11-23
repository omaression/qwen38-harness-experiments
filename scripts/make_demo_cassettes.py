#!/usr/bin/env python3
"""Write the three synthetic cassette sets of the demo, deterministically.

    python scripts/make_demo_cassettes.py            write cassettes/{baseline,candidate,candidate-regressed}
    python scripts/make_demo_cassettes.py --check    regenerate into a temporary directory and compare the
                                                     bytes with the committed sets: exit 0 when identical,
                                                     1 on any difference
    python scripts/make_demo_cassettes.py --out DIR  write the three sets under DIR instead

NOTHING HERE IS A MODEL. Every response is assembled by this script from the task suite in
tasks/demo: a task's reference_code stands for a correct fix, its wrong_code for an overfitted
one, and a few hand-written variants, sentence templates and deliberately broken replies stand
for the rest. The manifests say "synthetic": true, every gate report repeats it per arm, and the
CI summary opens with a banner saying so. The sets exist to show how the harness decides; they
say nothing about how good any model is. Recording a real model (`llm-eval record`) replaces them.

THE ARMS
    baseline             demo-model-a      bugfix/v1, judge/v1   the current baseline
    candidate            demo-model-a      bugfix/v2, judge/v2   a change that should promote
    candidate-regressed  demo-model-small  bugfix/v2, judge/v2   a change the gate must reject

THE RULES. An item's behaviour is chosen by a rule over i, the index of its task among the
sorted task ids of its split (dev and held-out are counted separately), and n, the number of
tasks in that split. The first rule that applies wins.

  bugfix (one item per task)
    baseline    i % 3 == 1                         overfit: the task's wrong_code
                otherwise                          fix: the task's reference_code
    candidate   i % 3 == 1 and (i // 3) % 3 == 0   overfit (one in three of the baseline's stays)
                i == n - 1                         overfit (a fix the baseline had: a loss)
                otherwise                          fix
    regressed   the task is tagged "parsing"       eval: a hand-written fix that calls eval()
                i == n - 1                         truncated: an unclosed <think>, finish "length"
                i % 11 == 2                        prose: the fix, with prose around the JSON
                i % 11 == 5                        missing key: {"patched_code": ...} alone
                i % 3 == 1                         overfit
                otherwise                          fix

  judge (two items per task: the reference patch, label true, and the wrong one, label false)
    baseline    reference                          true
                wrong, i % 7 == 3                  false
                wrong                              true (a rubber stamp)
    candidate   reference, i == n - 2              false (a correct patch rejected)
                reference                          true
                wrong, i % 5 == 2                  true (fooled by a plausible patch)
                wrong                              false
    regressed   reference, i % 10 == 4             truncated
                reference, i % 3 == 0              false
                reference                          true
                wrong, i % 20 == 7                 prose: true, with prose around the JSON
                wrong, i % 20 == 17                the verdict as the string "true"
                wrong, i % 3 == 2                  false
                wrong                              true

  Thinking: with v = 0 for bugfix items and judge reference items and v = 1 for judge wrong
  items, a reply opens with a closed <think> block when (i + v) % 2 == 0 for the baseline,
  always for the candidate, and when (i + v) % 2 == 1 for the regressed arm (never on prose,
  missing-key or string-verdict replies, which are pure format failures). Its length in words
  is drawn from sha256("<arm>|<item id>") within the arm's range (see ARMS below), always
  10 to 150 words. Explanations, reasons and thinking are built from short templates and the
  task's first tag; the thinking quotes the shown test as the prompt renders it.

  Usage and latency are derived from the text: completion_tokens = ceil(characters / 4),
  prompt_tokens likewise from the messages (plus 4 per message), and latency from a per-model
  linear rate. A truncated reply holds exactly 4 * max_tokens characters (a repetitive loop
  cut mid-word, as a model that loops until max_tokens leaves it), so its completion_tokens is
  max_tokens.

WHAT THE RULES GIVE, by construction (tests/test_demo.py checks it through the real gate):

  held-out (21 tasks)   bugfix resolved: baseline 14, candidate 17 (4 gains, 1 loss),
                        regressed 10 (4 losses; 5 schema failures, 2 of them leniently valid;
                        1 eval() patch). judge balanced accuracy: baseline 0.571 (approves
                        39 of 42), candidate 0.881, regressed 0.429 (4 schema failures).
  dev (13 tasks)        bugfix resolved: baseline 9, candidate 10, regressed 4 (3 eval()
                        patches). judge: baseline 0.577, candidate 0.846, regressed 0.423.

  The candidate promotes and the regressed candidate is rejected on both splits: bugfix fails
  primary, schema_validity and new_sast_violations; judge fails primary and schema_validity.

Changing a rule, a template or a task changes the cassettes, and with them the baseline's
fingerprint in the ledger: regenerate, then re-create ledger/ledger.jsonl (README, "Changing the
demo").
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
import shutil
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from llm_eval_harness import prompts, sast, tasks  # noqa: E402
from llm_eval_harness.cassettes import MANIFEST_FILE, CassetteWriter  # noqa: E402
from llm_eval_harness.features import get_feature  # noqa: E402
from llm_eval_harness.records import ALL, DEV, FEATURES, HELDOUT, ChatRequest, ChatResponse, Item, Suite, Task  # noqa: E402

SUITE_DIR = ROOT / "tasks" / "demo"
DEFAULT_OUT = ROOT / "cassettes"
CHARS_PER_TOKEN = 4


@dataclass(frozen=True)
class Arm:
    """One cassette set: its name (and directory), the model it plays, its prompts, its thinking
    lengths per feature (inclusive word ranges) and the latency rate of its model."""

    name: str
    model: str
    prompts: dict[str, str]
    think_words: dict[str, tuple[int, int]]
    latency: tuple[float, float, float]  # seconds: fixed, per completion token, per prompt token
    role: str


ARMS: tuple[Arm, ...] = (
    Arm("baseline", "demo-model-a", {"bugfix": "bugfix/v1", "judge": "judge/v1"},
        {"bugfix": (10, 60), "judge": (10, 40)}, (0.35, 0.018, 0.0004),
        "the current baseline: prompts v1 on demo-model-a, which often overfits the shown test and whose judge "
        "approves nearly every patch"),
    Arm("candidate", "demo-model-a", {"bugfix": "bugfix/v2", "judge": "judge/v2"},
        {"bugfix": (20, 80), "judge": (40, 150)}, (0.35, 0.018, 0.0004),
        "a candidate that should promote: prompts v2 on demo-model-a, which overfits less, and a judge that "
        "checks inputs beyond the shown test"),
    Arm("candidate-regressed", "demo-model-small", {"bugfix": "bugfix/v2", "judge": "judge/v2"},
        {"bugfix": (10, 40), "judge": (10, 40)}, (0.2, 0.006, 0.0001),
        "a candidate the gate must reject: prompts v2 on demo-model-small, which breaks the reply format, "
        "reaches for eval() on parsing tasks and judges worse than chance"),
)

# ---- hand-written eval() variants -------------------------------------------------------------
# The body of a fix that "works" by evaluating its input as Python: plausible, short, and exactly
# what the SAST gate exists to refuse. Each replaces the body of the reference fix under its def
# line and docstring. Every task tagged "parsing" needs one here.

EVAL_BODIES: dict[str, str] = {
    "evaluate-sum": "    return int(eval(expression))\n",
    "parse-int-list": (
        "    body = text.strip()\n"
        '    if not (body.startswith("[") and body.endswith("]")):\n'
        '        raise ValueError(f"not a list: {text!r}")\n'
        "    return [int(value) for value in eval(body)]\n"
    ),
    "parse-point": (
        "    try:\n"
        "        x, y = eval(text)\n"
        "    except Exception:\n"
        '        raise ValueError(f"not a point: {text!r}") from None\n'
        "    return int(x), int(y)\n"
    ),
    "parse-ratio": (
        "    try:\n"
        "        return float(eval(text))\n"
        "    except ZeroDivisionError:\n"
        '        raise ValueError(f"zero denominator in {text!r}") from None\n'
    ),
}

# ---- text templates -----------------------------------------------------------------------------
# What a bug class looks like in one clause ("The bug was that <phrase>."), by a task's first tag.

TAG_PHRASES: dict[str, str] = {
    "accumulator-init": "the accumulator started from the wrong value",
    "aliasing": "the input was modified through an alias",
    "boundary": "a value at the boundary was handled wrongly",
    "case-handling": "upper and lower case were not treated alike",
    "dict-handling": "the dictionary was updated in the wrong way",
    "early-return": "the function returned before it had checked every element",
    "empty-input": "an input that is too short was not handled",
    "integer-division": "integer division cut off the fractional part",
    "loop-condition": "the loop stopped before the last step",
    "mutable-default": "the default list was created once and shared between calls",
    "off-by-one": "a range stopped one step too early",
    "ordering": "the order of the input was not kept",
    "parsing": "the text was not parsed the way the docstring describes",
    "recursion-base-case": "the recursion's base case returned the wrong value",
    "rounding": "the result was rounded to the wrong precision",
    "set-handling": "a set operation lost what the result needs",
    "sign-handling": "negative numbers were not accepted",
    "slicing": "a slice had the wrong bounds",
    "sort-key": "the sort used the wrong key",
    "swapped-arguments": "two arguments were passed in the wrong order",
    "swapped-operands": "the operands of an operation were swapped",
    "unused-variable": "a value was computed and then never used",
    "whitespace-handling": "whitespace was not handled the way the docstring says",
    "wrong-comparison": "a comparison used the wrong operator",
    "wrong-exception": "the wrong exception type was raised",
    "wrong-operator": "the wrong operator was used",
    "wrong-variable": "the wrong variable was used",
}
GENERIC_PHRASE = "the function did not do what its docstring says"


def _phrase(task: Task) -> str:
    return TAG_PHRASES.get(task.tags[0], GENERIC_PHRASE) if task.tags else GENERIC_PHRASE


def _shown(task: Task) -> str:
    """The first shown test, exactly as the prompt renders it."""
    return prompts.render_check(task.shown_tests[0])


def _pick(options: list[str], key: str) -> str:
    return options[_hash(key) % len(options)]


def _hash(key: str) -> int:
    return int(hashlib.sha256(key.encode("utf-8")).hexdigest()[:8], 16)


def _explanation(kind: str, task: Task, key: str) -> str:
    phrase, ep = _phrase(task), task.entry_point
    if kind == "fix":
        return _pick([
            f"The bug was that {phrase}. The patch corrects that and leaves the rest of {ep} unchanged.",
            f"{phrase[0].upper() + phrase[1:]}, which is why the test failed. Fixed in place; the signature is unchanged.",
            f"Fixed {ep}: {phrase}.",
        ], key)
    if kind == "overfit":
        return _pick([
            f"The failing test now passes: {ep} returns the expected result for its input.",
            "Handled the input of the failing test, so the test passes now.",
        ], key)
    if kind == "eval":
        return f"Parsing the text by hand missed cases, so {ep} now evaluates the expression with Python directly."
    raise ValueError(kind)


def _bugfix_thoughts(kind: str, task: Task, careful: bool) -> list[str]:
    ep, shown = task.entry_point, _shown(task)
    if kind == "overfit":
        return [
            f"The test `{shown}` fails, so {ep} has to return the expected value for this input.",
            "The quickest way to make it pass is to handle this case directly.",
            "The rest of the function can stay as it is.",
            "That should make the failing test pass.",
        ]
    if kind == "eval":
        return [
            f"The test `{shown}` fails because the text is not parsed the way the docstring describes.",
            "Instead of fixing the parsing by hand, Python can evaluate the text directly.",
            "That covers every case the docstring mentions in one line.",
        ]
    thoughts = [
        f"The test `{shown}` fails, so I need to find out where {ep} goes wrong.",
        f"Reading the function next to its docstring, the problem seems to be that {_phrase(task)}.",
        "I will trace the shown input through the code to confirm that this is the cause.",
        "That confirms it: changing this one place should make the test pass.",
    ]
    if careful:
        thoughts += [
            "The docstring describes more inputs than this test, so the fix has to hold for all of them, not only the one shown.",
            "I check an empty input, the boundaries and a typical case in my head before answering.",
            "None of them needs a special case.",
        ]
    return thoughts + [
        "The rest of the function looks right, so I keep it as it is.",
        "The answer is the whole corrected function, with the same name and signature.",
    ]


def _judge_thoughts(kind: str, task: Task) -> list[str]:
    ep, shown = task.entry_point, _shown(task)
    if kind == "stamp":  # approves without checking
        return [
            f"The patch changes {ep}, and the test `{shown}` should pass with it.",
            "The change is small and looks like a reasonable fix for the bug.",
            "I do not see anything wrong with it.",
        ]
    opening = f"The test `{shown}` failed on the original {ep}; first I check that the patch fixes it."
    if kind == "approve":
        return [
            opening,
            "It does, and the change is in the place where the bug was.",
            "Next I try other inputs the docstring describes: an empty input, the boundaries and a typical case.",
            "The patched function handles them the way the docstring says.",
            "There is no special case for the input of the test.",
            "So the patch is correct for every input I tried.",
        ]
    if kind == "reject":
        return [
            opening,
            "The shown test passes with the patch.",
            "Next I try other inputs the docstring describes, not only the one in the test.",
            "For some of them the patched function does not do what the docstring says.",
            "So the patch makes the test pass without fixing the bug.",
        ]
    if kind == "doubt":
        return [
            opening,
            "The shown test passes with the patch.",
            "I am not sure the change covers every input the docstring describes.",
            "Without a clear reason to trust it, I will not approve it.",
        ]
    raise ValueError(kind)


def _think(thoughts: list[str], words: int) -> str:
    """Whole sentences, in order, while they fit in `words` (the first one always)."""
    chosen: list[str] = []
    count = 0
    for sentence in thoughts:
        size = len(sentence.split())
        if chosen and count + size > words:
            break
        chosen.append(sentence)
        count += size
    return " ".join(chosen)


def _loop(task: Task, chars: int) -> str:
    """A thinking loop that never closes, cut at exactly `chars` characters (mid-word)."""
    shown = _shown(task)
    cycle = (
        f"Wait, let me check `{shown}` again. "
        "The code looks right, but the result is still not what the test expects. "
        "Maybe the problem is somewhere else. "
    )
    text = "<think>\n"
    while len(text) < chars:
        text += cycle
    return text[:chars]


# ---- the behaviours -----------------------------------------------------------------------------


def bugfix_behaviour(arm: str, task: Task, i: int, n: int) -> str:
    """See THE RULES (bugfix)."""
    if arm == "baseline":
        return "overfit" if i % 3 == 1 else "fix"
    if arm == "candidate":
        if i % 3 == 1 and (i // 3) % 3 == 0:
            return "overfit"
        return "overfit" if i == n - 1 else "fix"
    if arm == "candidate-regressed":
        if "parsing" in task.tags:
            return "eval"
        if i == n - 1:
            return "truncated"
        if i % 11 == 2:
            return "prose"
        if i % 11 == 5:
            return "missing-key"
        return "overfit" if i % 3 == 1 else "fix"
    raise ValueError(f"unknown arm {arm!r}")


def judge_behaviour(arm: str, variant: str, i: int, n: int) -> str:
    """See THE RULES (judge). Returns "true", "false", "truncated", "prose" or "string"."""
    reference = variant == "reference"
    if arm == "baseline":
        return "true" if reference or i % 7 != 3 else "false"
    if arm == "candidate":
        if reference:
            return "false" if i == n - 2 else "true"
        return "true" if i % 5 == 2 else "false"
    if arm == "candidate-regressed":
        if reference:
            if i % 10 == 4:
                return "truncated"
            return "false" if i % 3 == 0 else "true"
        if i % 20 == 7:
            return "prose"
        if i % 20 == 17:
            return "string"
        return "false" if i % 3 == 2 else "true"
    raise ValueError(f"unknown arm {arm!r}")


def eval_variant(task: Task) -> str:
    """The task's hand-written eval() fix: the reference's def line and docstring, then the
    EVAL_BODIES body. Refuses a variant the SAST scan would not block (the demo depends on it)."""
    if task.id not in EVAL_BODIES:
        raise SystemExit(f"make_demo_cassettes: task {task.id!r} is tagged 'parsing' but EVAL_BODIES has no variant for it")
    tree = ast.parse(task.reference_code)
    func = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == task.entry_point)
    first = func.body[0]
    has_doc = isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant) and isinstance(first.value.value, str)
    lines = task.reference_code.splitlines(keepends=True)
    end = first.end_lineno if has_doc else first.lineno - 1
    code = "".join(lines[func.lineno - 1:end]) + EVAL_BODIES[task.id]
    if not sast.blocking(sast.added_findings(task.buggy_code, code)):
        raise SystemExit(f"make_demo_cassettes: the eval() variant of {task.id!r} has no block-class SAST finding")
    return code


# ---- responses ----------------------------------------------------------------------------------


@dataclass(frozen=True)
class Context:
    """What a response is made from: the arm, the item and its rule inputs, the request."""

    arm: Arm
    item: Item
    i: int
    n: int
    request: ChatRequest

    @property
    def key(self) -> str:
        return f"{self.arm.name}|{self.item.id}"

    def thinks(self) -> bool:
        v = 1 if self.item.variant == "wrong" else 0
        if self.arm.name == "baseline":
            return (self.i + v) % 2 == 0
        if self.arm.name == "candidate":
            return True
        return (self.i + v) % 2 == 1

    def think_words(self) -> int:
        lo, hi = self.arm.think_words[self.item.feature]
        return lo + _hash(self.key) % (hi - lo + 1)


def _with_think(ctx: Context, thoughts: list[str], answer: str) -> str:
    if not ctx.thinks():
        return answer
    return f"<think>\n{_think(thoughts, ctx.think_words())}\n</think>\n\n{answer}"


def _respond(ctx: Context, content: str, finish_reason: str = "stop") -> ChatResponse:
    prompt_chars = sum(len(m.content) for m in ctx.request.messages)
    prompt_tokens = math.ceil(prompt_chars / CHARS_PER_TOKEN) + 4 * len(ctx.request.messages)
    completion_tokens = max(1, math.ceil(len(content) / CHARS_PER_TOKEN))
    fixed, per_completion, per_prompt = ctx.arm.latency
    latency = round(fixed + per_completion * completion_tokens + per_prompt * prompt_tokens, 3)
    return ChatResponse(
        content=content,
        reasoning=None,
        finish_reason=finish_reason,
        usage={"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens, "total_tokens": prompt_tokens + completion_tokens},
        latency_s=latency,
    )


def _truncated(ctx: Context) -> ChatResponse:
    return _respond(ctx, _loop(ctx.item.task, CHARS_PER_TOKEN * ctx.request.settings.max_tokens), "length")


def bugfix_response(ctx: Context) -> ChatResponse:
    task = ctx.item.task
    kind = bugfix_behaviour(ctx.arm.name, task, ctx.i, ctx.n)
    if kind == "truncated":
        return _truncated(ctx)
    if kind == "eval":
        code = eval_variant(task)
    elif kind == "overfit":
        code = task.wrong_code
    else:  # fix, prose, missing-key: the right code, in a good or a broken reply
        code = task.reference_code
    explanation = _explanation("fix" if kind in ("prose", "missing-key") else kind, task, ctx.key)
    answer = json.dumps({"explanation": explanation, "patched_code": code})
    if kind == "prose":
        return _respond(ctx, f"Sure, here is the corrected function.\n\n{answer}\n\nThis makes the failing test pass.")
    if kind == "missing-key":
        return _respond(ctx, json.dumps({"patched_code": code}))
    careful = ctx.arm.prompts["bugfix"] != "bugfix/v1"
    return _respond(ctx, _with_think(ctx, _bugfix_thoughts(kind, task, careful), answer))


def judge_response(ctx: Context) -> ChatResponse:
    item = ctx.item
    kind = judge_behaviour(ctx.arm.name, item.variant or "", ctx.i, ctx.n)
    if kind == "truncated":
        return _truncated(ctx)
    if kind == "prose":
        return _respond(ctx, 'The patch looks correct to me.\n{"verdict": true}')
    if kind == "string":
        return _respond(ctx, json.dumps({"verdict": "true", "reason": "The patch fixes the failing test."}))
    verdict = kind == "true"
    if ctx.arm.name == "baseline":
        thoughts = _judge_thoughts("stamp" if verdict else "reject", item.task)
        answer = {"verdict": True} if verdict else {"verdict": False, "reason": "The patch only makes the shown test pass."}
    else:
        if verdict:
            thought_kind, reason = "approve", "The patched function does what the docstring says for the inputs I checked."
        elif item.variant == "reference":
            thought_kind, reason = "doubt", "Not sure the patch covers every input the docstring describes."
        else:
            thought_kind, reason = "reject", "The patched function still fails for inputs the docstring describes."
        thoughts = _judge_thoughts(thought_kind, item.task)
        answer = {"verdict": verdict, "reason": reason}
    return _respond(ctx, _with_think(ctx, thoughts, json.dumps(answer)))


RESPONDERS: dict[str, Callable[[Context], ChatResponse]] = {"bugfix": bugfix_response, "judge": judge_response}


# ---- writing ------------------------------------------------------------------------------------


def _positions(suite: Suite) -> dict[str, tuple[int, int]]:
    """task id -> (i, n): its index among the sorted task ids of its split, and the split's size."""
    out: dict[str, tuple[int, int]] = {}
    for split in (DEV, HELDOUT):
        ids = [t.id for t in tasks.tasks_in_split(suite, split)]
        out.update({task_id: (i, len(ids)) for i, task_id in enumerate(ids)})
    return out


def description(arm: Arm) -> str:
    return (
        "Synthetic responses: written for the demo by scripts/make_demo_cassettes.py, not produced by a model. "
        f"They play {arm.role}."
    )


def write_sets(out: Path, suite: Suite) -> dict[str, dict[str, dict[str, int]]]:
    """Write every arm under `out`; returns {arm: {feature: {behaviour: items}}} for the log."""
    positions = _positions(suite)
    counts: dict[str, dict[str, dict[str, int]]] = {}
    for arm in ARMS:
        root = out / arm.name
        root.mkdir(parents=True, exist_ok=True)
        counts[arm.name] = {}
        for name in FEATURES:
            feature = get_feature(name)
            template = prompts.get_template(arm.prompts[name], name)
            settings = feature.default_settings
            writer = CassetteWriter(
                root, arm=arm.name, feature=name, prompt=template.id, model=arm.model, settings=settings,
                suite_name=suite.name, suite_sha256=suite.sha256, synthetic=True, description=description(arm),
            )
            tally: dict[str, int] = {}
            for item in feature.items(suite, ALL):  # sorted by item id
                i, n = positions[item.task.id]
                request = prompts.build_request(template, item, arm.model, settings)
                writer.write(request, RESPONDERS[name](Context(arm, item, i, n, request)))
                behaviour = (bugfix_behaviour(arm.name, item.task, i, n) if name == "bugfix"
                             else f"{item.variant}:{judge_behaviour(arm.name, item.variant or '', i, n)}")
                tally[f"{item.split}/{behaviour}"] = tally.get(f"{item.split}/{behaviour}", 0) + 1
            writer.close()
            counts[arm.name][name] = dict(sorted(tally.items()))
    return counts


def _set_files(directory: Path) -> dict[str, bytes]:
    """The files that make up a cassette set (what cassettes.fingerprint covers), by name."""
    if not directory.is_dir():
        return {}
    return {p.name: p.read_bytes() for p in sorted(directory.iterdir())
            if p.is_file() and (p.name == MANIFEST_FILE or p.suffix == ".jsonl")}


def compare_sets(generated: Path, committed: Path) -> list[str]:
    """Every difference between the generated and the committed sets, as sentences that name
    the files relative to the sets' directory ("candidate/bugfix.jsonl")."""
    problems = []
    for arm in ARMS:
        new, old = _set_files(generated / arm.name), _set_files(committed / arm.name)
        if not old:
            problems.append(f"{arm.name}/: no cassette set there")
            continue
        for name in sorted(new.keys() | old.keys()):
            if name not in old:
                problems.append(f"{arm.name}/{name}: missing from the committed set")
            elif name not in new:
                problems.append(f"{arm.name}/{name}: not written by this script")
            elif new[name] != old[name]:
                problems.append(f"{arm.name}/{name}: differs from what this script writes")
    return problems


def install(generated: Path, out: Path) -> None:
    """Replace each arm's cassette files under `out` with the generated ones (other files in the
    arm directories are left alone)."""
    for arm in ARMS:
        target = out / arm.name
        target.mkdir(parents=True, exist_ok=True)
        for name in _set_files(target):
            (target / name).unlink()
        for name in _set_files(generated / arm.name):
            shutil.copyfile(generated / arm.name / name, target / name)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Write the synthetic demo cassettes (see the module docstring).")
    parser.add_argument("--check", action="store_true", help="compare a fresh generation with the committed sets instead of writing")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT, help="the directory that holds the sets (default: cassettes/)")
    args = parser.parse_args(argv)
    suite = tasks.load_suite(SUITE_DIR)
    with tempfile.TemporaryDirectory(prefix="demo-cassettes-") as tmp:
        generated = Path(tmp)
        counts = write_sets(generated, suite)
        if args.check:
            problems = compare_sets(generated, args.out)
            for problem in problems:
                print(f"make_demo_cassettes: {problem}", file=sys.stderr)
            if problems:
                print("make_demo_cassettes: the committed cassettes are not what this script writes: run it without --check", file=sys.stderr)
                return 1
            print(f"make_demo_cassettes: the {len(ARMS)} committed cassette sets are identical to a fresh generation")
            return 0
        install(generated, args.out)
    for arm, features in counts.items():
        for feature, tally in features.items():
            shown = ", ".join(f"{key} {value}" for key, value in tally.items())
            print(f"{arm:20} {feature:7} {shown}")
    print(f"make_demo_cassettes: wrote {len(ARMS)} synthetic cassette sets for suite {suite.name} ({suite.sha256[:12]})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
