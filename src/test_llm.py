"""
test_llm.py — Quick end-to-end sanity check for llm.py
Tests all three core functions: summarise_document, answer_question, simplify_passage

Run:
    python test_llm.py
    python test_llm.py --verbose      # show full LLM responses
"""

import os
import sys
import time
import argparse
from dotenv import load_dotenv

load_dotenv()

VERBOSE = "--verbose" in sys.argv or "-v" in sys.argv

# ── Colour helpers (works on Windows 10+ and all Unix) ───────────────────────
GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
CYAN   = "\033[96m"
BOLD   = "\033[1m"
RESET  = "\033[0m"

def ok(msg):  print(f"  {GREEN}✅  PASS{RESET}  {msg}")
def fail(msg): print(f"  {RED}❌  FAIL{RESET}  {msg}")
def info(msg): print(f"  {CYAN}ℹ    {RESET}  {msg}")
def warn(msg): print(f"  {YELLOW}⚠    {RESET}  {msg}")
def header(msg): print(f"\n{BOLD}{msg}{RESET}\n" + "─" * 55)

# ── Sample data ───────────────────────────────────────────────────────────────
SAMPLE_TEXT = """
The Braille system was invented by Louis Braille in 1824.
Louis was blind from the age of three after a childhood accident.
He created a system of raised dots that blind people can read with their fingertips.
Each character is made up of a cell of six dots arranged in two columns of three.
Today, Braille is used worldwide and is available in almost every language.
It is used in books, menus, medicine labels, and elevator buttons.
Braille literacy is closely linked to better employment and independence for blind people.
"""

SAMPLE_CHUNKS = [
    ("Louis Braille invented the Braille system in 1824 after losing his sight at age three.", 0.82),
    ("Each Braille character uses a cell of six raised dots in a 2×3 grid.", 0.78),
    ("Braille is available in almost every language and used worldwide.", 0.71),
    ("Braille literacy is linked to higher employment rates among blind people.", 0.65),
]

WEAK_CHUNKS = [
    ("The weather in Paris is often mild in spring.", 0.21),
    ("French cuisine includes baguettes and croissants.", 0.18),
]

results = {"pass": 0, "fail": 0}

def run(label, fn):
    """Run a test function, catch errors, print result."""
    print(f"\n  {BOLD}→ {label}{RESET}")
    try:
        t0 = time.time()
        response = fn()
        elapsed = time.time() - t0

        if not response or not response.strip():
            fail(f"Empty response ({elapsed:.1f}s)")
            results["fail"] += 1
            return

        word_count = len(response.split())
        ok(f"{word_count} words returned in {elapsed:.1f}s")

        if VERBOSE:
            print()
            for line in response.strip().splitlines():
                print(f"      {line}")
            print()

        results["pass"] += 1

    except Exception as e:
        fail(str(e))
        results["fail"] += 1


# ── Pre-flight checks ─────────────────────────────────────────────────────────
header("PRE-FLIGHT")

api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    fail("GROQ_API_KEY not set — add it to your .env file")
    sys.exit(1)
ok(f"GROQ_API_KEY found  ({api_key[:8]}{'*' * (len(api_key) - 8)})")

try:
    from groq import Groq
    ok("groq package installed")
except ImportError:
    fail("groq not installed — run: pip install groq")
    sys.exit(1)

try:
    from llm import summarise_document, answer_question, simplify_passage, META_QUESTIONS, LOW_SCORE_THRESHOLD
    ok("llm.py imported successfully")
    info(f"META_QUESTIONS has {len(META_QUESTIONS)} entries")
    info(f"LOW_SCORE_THRESHOLD = {LOW_SCORE_THRESHOLD}")
except SyntaxError as e:
    fail(f"Syntax error in llm.py → {e}")
    sys.exit(1)
except ImportError as e:
    fail(f"Import error → {e}")
    sys.exit(1)


# ── Test 1: summarise_document ────────────────────────────────────────────────
header("TEST 1 — summarise_document")

run("Standard summary", lambda: summarise_document(SAMPLE_TEXT, mode="standard"))
run("Simple / ELI10 summary", lambda: summarise_document(SAMPLE_TEXT, mode="simple"))
run("Bullet-point summary", lambda: summarise_document(SAMPLE_TEXT, mode="bullet"))


# ── Test 2: answer_question (normal RAG path) ─────────────────────────────────
header("TEST 2 — answer_question  (strong chunks)")

run(
    "Direct factual question",
    lambda: answer_question("Who invented Braille?", SAMPLE_CHUNKS),
)
run(
    "Detail question",
    lambda: answer_question("How many dots are in a Braille cell?", SAMPLE_CHUNKS),
)
run(
    "With chat history",
    lambda: answer_question(
        "Is it used in other languages?",
        SAMPLE_CHUNKS,
        chat_history=[
            {"role": "user",      "content": "Tell me about Braille."},
            {"role": "assistant", "content": "Braille is a tactile writing system for blind people."},
        ],
    ),
)


# ── Test 3: Meta-question guard ───────────────────────────────────────────────
header("TEST 3 — answer_question  (meta-question guard)")

for q in ["what is in the pdf", "summarize", "give me a summary", "overview"]:
    run(
        f'Meta → "{q}"',
        lambda q=q: answer_question(q, SAMPLE_CHUNKS),
    )


# ── Test 4: Low-score fallback guard ─────────────────────────────────────────
header("TEST 4 — answer_question  (low-score fallback)")

run(
    "Weak chunks (avg ~0.20) → should fall back to summarise",
    lambda: answer_question("What is Braille?", WEAK_CHUNKS),
)


# ── Test 5: simplify_passage ──────────────────────────────────────────────────
header("TEST 5 — simplify_passage")

hard_text = (
    "The proliferation of assistive technologies has engendered a paradigm shift "
    "in accessibility discourse, necessitating a reevaluation of heuristic frameworks "
    "employed by rehabilitative practitioners."
)
run("Simplify complex passage", lambda: simplify_passage(hard_text))


# ── Test 6: Edge cases ────────────────────────────────────────────────────────
header("TEST 6 — Edge cases")

run(
    "Very short text summary",
    lambda: summarise_document("Braille helps blind people read.", mode="standard"),
)
run(
    "No chat history (None)",
    lambda: answer_question("What is Braille used for?", SAMPLE_CHUNKS, chat_history=None),
)
run(
    "Empty chat history ([])",
    lambda: answer_question("What is Braille used for?", SAMPLE_CHUNKS, chat_history=[]),
)


# ── Final report ──────────────────────────────────────────────────────────────
total = results["pass"] + results["fail"]
header("RESULTS")
print(f"  Passed : {GREEN}{results['pass']}/{total}{RESET}")
if results["fail"]:
    print(f"  Failed : {RED}{results['fail']}/{total}{RESET}")
    print(f"\n  {YELLOW}Tip: run with --verbose to see full LLM responses.{RESET}")
else:
    print(f"\n  {GREEN}{BOLD}All tests passed — your LLM pipeline is working correctly! 🎉{RESET}")

print()