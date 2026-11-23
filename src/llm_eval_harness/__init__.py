"""llm_eval_harness: an evaluation harness for an LLM feature, with a regression gate for CI.

The feature under test is an LLM bug-fix assistant. Given a buggy Python function (with its
docstring) and one failing test, the model returns an optional <think>...</think> block followed
by a strict JSON object {"explanation": str, "patched_code": str}. A second, smaller feature is an
LLM patch-correctness judge that returns {"verdict": bool}. The judge is scored against labels
that come from running the tests.

The modules, in the order data flows through them (docs/DESIGN.md has the whole picture):

    tasks        the task suite (JSONL) and its dev / held-out split by sha256 bucket
    prompts      versioned prompt templates that never change once published; one request per item
    clients      a live OpenAI-compatible client (urllib), a replay client and a recorder
    cassettes    recorded responses on disk, keyed by item and checked against the request hash
    sandbox      model-written code in a temporary directory, a child process and a deadline
    verdicts     think-block validity, strict JSON schema validity and the strict execution verdict
    sast         an AST-based scan for unsafe or harness-gaming constructs
    features     the two features: their items and how a reply is scored
    validation   the checks a task suite must pass before anything is scored on it
    runner       one arm (a cassette set) evaluated on a split, recording, and the gate run
    metrics      rates with Wilson intervals, classifier metrics, the paired comparison
    gate         the hard gates, the non-inferiority rule and the JSON report
    report       the report's JSON form and a markdown summary for CI
    ledger       a hash-chained record of evaluations, promotions and rollbacks
    cli          the `llm-eval` command

    records      the shared data records (Task, Item, ChatRequest, ItemResult, ArmRun, ...)
    util         canonical JSON, hashing, JSONL files, display paths, logging to stderr
"""

__version__ = "0.1.0"

__all__ = ["__version__"]
