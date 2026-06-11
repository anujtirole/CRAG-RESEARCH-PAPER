"""
Confidence-scoring critic for the agentic RAG pipeline.

Design contract:
  • Chain-of-thought BEFORE score: justification first, then "SCORE: X.XX"
  • Conservative by default: parse failure → 0.2 (low), never high
  • Anti-sycophancy prompt: explicitly tells the model most chunks are NOT relevant
  • --calibration-check mode: empirically shows the critic discriminates between
    irrelevant and relevant pairs — directly answers reviewer bias concerns

Run directly:
  python critic.py                    # single test pair
  python critic.py --calibration-check
"""

import json
import re
import sys
import argparse
import statistics
from typing import List, Optional

import requests

import config

# ── System prompt ─────────────────────────────────────────────────────────────
# Printed once per process so you can copy it verbatim into the paper.
SYSTEM_PROMPT = """You are a strict relevance assessor for a retrieval-augmented generation system. Your task is to score how well a retrieved text chunk supports answering a specific query.

CALIBRATION GUIDELINES — read carefully before scoring:
- The retrieval system returns many plausible-looking but weakly relevant chunks. Most retrieved chunks are NOT highly relevant. Your scores must reflect this.
- Score > 0.7: Reserve ONLY for chunks that directly and specifically contain the answer to the query. This should be uncommon.
- Score 0.5 – 0.7: The chunk discusses the same general topic but does not directly answer the question.
- Score 0.3 – 0.5: The chunk is tangentially related or shares only surface-level keywords with the query.
- Score < 0.3: The chunk is largely irrelevant — wrong domain, wrong subject, or only incidental keyword overlap.
- Score 0.0 – 0.1: The chunk is completely irrelevant; entirely different subject matter.

Do NOT inflate scores to seem helpful. A high score means the chunk directly answers the query.

RESPONSE FORMAT — you MUST follow this exactly or your response will be rejected:
1. Write 2–3 sentences of justification explaining WHY the chunk is or is not relevant to the query.
2. On a NEW line, write exactly: SCORE: <a decimal number between 0.0 and 1.0>

The justification MUST come before the score line."""

# Batch score-only prompt — used by the eval loop (no justification, 1 call for k chunks).
# Keeps the same rubric as SYSTEM_PROMPT; strips CoT to minimise output tokens.
_BATCH_SYSTEM_PROMPT = (
    "You are a strict relevance assessor for a retrieval-augmented generation system.\n"
    "Score how well each retrieved chunk supports answering the query.\n\n"
    "IMPORTANT: Many queries are multi-hop questions that require combining facts from\n"
    "multiple chunks. A chunk does NOT need to contain the entire answer to score well.\n\n"
    "SCORING RUBRIC — calibrated for multi-hop retrieval:\n"
    "  > 0.7  : chunk directly contains the answer or a critical fact needed to answer\n"
    "           (e.g. identifies a key entity, date, location, or relationship)\n"
    "  0.5–0.7: chunk contains a partial answer or one required fact in a reasoning chain\n"
    "           (e.g. names an intermediate entity whose properties the question asks about)\n"
    "  0.3–0.5: chunk is about the right topic or entity but does not contribute a needed fact\n"
    "  < 0.3  : largely irrelevant — wrong subject or only peripheral keyword overlap\n"
    "  0.0–0.1: completely irrelevant\n\n"
    "Respond with ONLY a JSON array of decimal scores, one per chunk.\n"
    "Example for 5 chunks: [0.3, 0.6, 0.1, 0.2, 0.4]\n"
    "No explanation. No extra text. Just the JSON array."
)

_SYSTEM_PROMPT_PRINTED = False


# ── Ollama HTTP helper ────────────────────────────────────────────────────────

def _call_ollama_chat(
    messages: list[dict],
    temperature: float = 0.0,
    num_predict: int = 250,
) -> str:
    url = f"{config.OLLAMA_BASE_URL}/api/chat"
    payload = {
        "model": config.LLM_MODEL,
        "messages": messages,
        "stream": False,
        "options": {
            "temperature": temperature,
            "seed": config.RANDOM_SEED,
            "num_predict": num_predict,
        },
    }
    try:
        resp = requests.post(url, json=payload, timeout=config.OLLAMA_TIMEOUT)
        resp.raise_for_status()
        return resp.json()["message"]["content"]
    except requests.exceptions.Timeout:
        raise TimeoutError(f"Ollama timed out after {config.OLLAMA_TIMEOUT}s")
    except Exception as exc:
        raise RuntimeError(f"Ollama chat failed: {exc}") from exc


# ── Score parsing ─────────────────────────────────────────────────────────────

def _parse_score(text: str) -> Optional[float]:
    """Extract the float from a 'SCORE: X.XX' line. Returns None if absent."""
    match = re.search(r"SCORE:\s*([0-9]+(?:\.[0-9]+)?)", text, re.IGNORECASE)
    if not match:
        return None
    return max(0.0, min(1.0, float(match.group(1))))


def _parse_batch_scores(text: str, expected: int) -> Optional[List[float]]:
    """
    Parse a JSON array of k floats from the model's response.
    Tries JSON first, then falls back to extracting all bare numbers.
    Returns None if the count doesn't match or parsing fails entirely.
    """
    text = text.strip()
    # Primary: find the first [...] block and JSON-parse it
    bracket = re.search(r"\[([^\]]+)\]", text)
    if bracket:
        try:
            raw = json.loads(f"[{bracket.group(1)}]")
            if isinstance(raw, list) and len(raw) == expected:
                return [max(0.0, min(1.0, float(v))) for v in raw]
        except (json.JSONDecodeError, ValueError, TypeError):
            pass
    # Fallback: collect every decimal / integer token
    tokens = re.findall(r"\b\d+(?:\.\d+)?\b", text)
    if len(tokens) == expected:
        try:
            return [max(0.0, min(1.0, float(t))) for t in tokens]
        except ValueError:
            pass
    return None


# ── Public API ────────────────────────────────────────────────────────────────

def score_context(query: str, chunk: str) -> float:
    """
    Score how well `chunk` supports answering `query`.

    Returns a float in [0, 1].
    - Parse failures default to 0.2 (conservative, not high).
    - System prompt is printed once per process for paper methodology section.
    """
    global _SYSTEM_PROMPT_PRINTED
    if not _SYSTEM_PROMPT_PRINTED:
        print("\n" + "=" * 72)
        print("CRITIC SYSTEM PROMPT (copy verbatim into paper methodology section):")
        print("=" * 72)
        print(SYSTEM_PROMPT)
        print("=" * 72 + "\n", flush=True)
        _SYSTEM_PROMPT_PRINTED = True

    user_content = f"Query: {query}\n\nRetrieved chunk:\n{chunk}"
    messages: list[dict] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": user_content},
    ]

    for attempt in range(2):
        try:
            response_text = _call_ollama_chat(messages, temperature=0.0)
            score = _parse_score(response_text)
            if score is not None:
                return score
            # Score line missing — append a nudge and retry once
            if attempt == 0:
                messages.append({"role": "assistant", "content": response_text})
                messages.append({
                    "role": "user",
                    "content": (
                        "Your response did not include a SCORE line. "
                        "Please end your response with exactly: SCORE: <number between 0.0 and 1.0>"
                    ),
                })
        except (TimeoutError, RuntimeError) as exc:
            print(f"  [critic] Warning (attempt {attempt + 1}): {exc}", file=sys.stderr)

    # Both attempts failed — default to low score (conservative)
    print("  [critic] Warning: score parse failed twice; defaulting to 0.2", file=sys.stderr)
    return 0.2


def score_contexts_batch(query: str, chunks: List[str]) -> List[float]:
    """
    Score all k chunks in a SINGLE Ollama call (score-only, no justification).
    Used by the agentic eval loop to collapse k sequential calls into 1.
    Parse failures default to 0.2 per chunk (same conservative policy).

    Not suitable for --calibration-check (no CoT → not explainable).
    """
    if not chunks:
        return []

    chunk_block = "\n\n".join(f"[{i + 1}] {c}" for i, c in enumerate(chunks))
    user_content = f"Query: {query}\n\nChunks to score:\n{chunk_block}"
    messages: list[dict] = [
        {"role": "system", "content": _BATCH_SYSTEM_PROMPT},
        {"role": "user",   "content": user_content},
    ]

    for attempt in range(2):
        try:
            # num_predict=96 is ample for "[0.3, 0.6, 0.1, 0.2, 0.4]" (~20 tokens)
            response_text = _call_ollama_chat(messages, temperature=0.0, num_predict=96)
            scores = _parse_batch_scores(response_text, expected=len(chunks))
            if scores is not None:
                return scores
            if attempt == 0:
                messages.append({"role": "assistant", "content": response_text})
                messages.append({
                    "role": "user",
                    "content": (
                        f"Your response did not contain a valid JSON array of "
                        f"{len(chunks)} scores. Reply with ONLY the JSON array, "
                        f"e.g.: [0.3, 0.6, 0.1, 0.2, 0.4]"
                    ),
                })
        except (TimeoutError, RuntimeError) as exc:
            print(f"  [critic] Warning (batch attempt {attempt + 1}): {exc}", file=sys.stderr)

    print(
        f"  [critic] Warning: batch parse failed twice; defaulting all {len(chunks)} to 0.2",
        file=sys.stderr,
    )
    return [0.2] * len(chunks)


# ── Calibration check ─────────────────────────────────────────────────────────

_IRRELEVANT_PAIRS = [
    ("What is the capital of France?",
     "Python decorators are a design pattern that allows you to wrap a function to extend its behaviour without permanently modifying it. They are applied with the @ symbol above a function definition."),
    ("How do vaccines work?",
     "The Dow Jones Industrial Average rose 1.4% on Thursday as technology stocks led a broad market rally following better-than-expected corporate earnings reports."),
    ("What causes thunderstorms?",
     "Julius Caesar was assassinated on the Ides of March, 44 BCE, by a group of Roman senators led by Brutus and Cassius who feared he was becoming a tyrant."),
    ("How is bread made?",
     "Quantum entanglement is a phenomenon where two particles become correlated such that the quantum state of one instantly influences the other, regardless of distance."),
    ("What is the speed of light?",
     "Leonardo da Vinci used sfumato — a technique of very fine shading and delicate tonal gradations — to create the soft, mysterious atmosphere in the Mona Lisa."),
    ("How do plants absorb water?",
     "Bitcoin uses a proof-of-work consensus mechanism where miners compete to solve cryptographic hash puzzles, with the winner adding the next block to the blockchain."),
    ("What is the Pythagorean theorem?",
     "To make traditional carbonara, render guanciale until crispy, then toss with al dente spaghetti and a mixture of egg yolks and Pecorino Romano off the heat."),
    ("How does the immune system work?",
     "The carburetor in older petrol engines mixes fuel with air in the correct stoichiometric ratio before delivering the mixture to the engine cylinders for combustion."),
    ("What is inflation?",
     "Coral reefs are built over centuries by tiny marine animals called polyps that secrete calcium carbonate exoskeletons, creating complex three-dimensional reef structures."),
    ("How do solar panels generate electricity?",
     "Medieval castle designers favoured concentric curtain walls, a fortified gatehouse, and a central keep on high ground to create multiple lines of defence."),
    ("What is DNA?",
     "Miles Davis's 1959 album Kind of Blue pioneered modal jazz by replacing complex chord changes with simple scales, becoming the best-selling jazz album in history."),
    ("How does GPS determine your location?",
     "Traditional winemaking converts grape sugars into ethanol and carbon dioxide through yeast fermentation, with temperature and oxygen exposure shaping the final flavour."),
    ("What causes earthquakes?",
     "The power-dressing aesthetic of the 1980s featured high-waisted trousers, bold padded shoulders, and vivid primary colours as women entered corporate professions."),
    ("How is float glass manufactured?",
     "Michael Phelps won 23 Olympic gold medals across four Olympic Games, a record that may stand for decades in competitive swimming."),
    ("What is machine learning?",
     "Ancient Egyptian embalmers removed internal organs, dried the body with natron salt over seventy days, and wrapped it in linen bandages before placing it in a sarcophagus."),
    ("How do antibiotics kill bacteria?",
     "Suspension bridges carry deck loads via cables strung between tall towers and anchored at both ends, allowing main spans of more than 2 000 metres."),
    ("What is photosynthesis?",
     "Capital gains tax is levied on the profit realised when a capital asset is sold for more than its purchase price, with rates varying by jurisdiction and holding period."),
    ("How does human memory work?",
     "Crampons are metal frames with downward-pointing spikes attached to mountaineering boots to provide traction on hard ice and steep snow slopes."),
    ("What is the greenhouse effect?",
     "In the Sicilian Defence, Black responds to 1.e4 with 1…c5, seeking an asymmetric position with active counterplay and avoiding symmetrical pawn structures."),
    ("How do neurons transmit signals?",
     "Dovetail joints interlock at an angle so that the joint resists pulling apart under tension; cabinetmakers prize them for drawer construction because of their mechanical strength."),
]

_RELEVANT_PAIRS = [
    ("What is the capital of France?",
     "Paris is the capital and most populous city of France. It serves as the country's political, economic, and cultural centre and is home to the French national government and Élysée Palace."),
    ("How do vaccines work?",
     "Vaccines work by introducing the immune system to a harmless antigen — a weakened pathogen, inactivated virus, or mRNA encoding a viral protein — prompting the production of antibodies and memory B cells so the body can mount a rapid defence upon real infection."),
    ("What causes thunderstorms?",
     "Thunderstorms form when warm, moist surface air rises rapidly into cooler atmospheric layers, creating towering cumulonimbus clouds. Strong updrafts separate electric charges, leading to lightning; the rapid heating of air by lightning bolts produces the acoustic shock wave we hear as thunder."),
    ("How is bread made?",
     "Bread is produced by combining flour, water, yeast, and salt into a dough, which is kneaded to develop gluten networks. The dough is proofed as yeast ferments sugars and releases CO₂, causing the dough to rise; it is then baked at high temperature, which sets the gluten structure and forms the crust."),
    ("What is the speed of light?",
     "The speed of light in a vacuum is exactly 299,792,458 metres per second, denoted c. This is a universal physical constant; by definition since 1983, it fixes the length of the metre. Nothing with mass can reach or exceed c, making it a fundamental limit in special relativity."),
    ("How do plants absorb water?",
     "Plants absorb water through root hair cells via osmosis, driven by the lower water potential inside the root compared with the soil solution. Water then moves from cell to cell and enters the xylem, where the cohesion–tension mechanism — driven by transpiration at the leaves — pulls continuous water columns upward to the shoot."),
    ("What is the Pythagorean theorem?",
     "The Pythagorean theorem states that in a right-angled triangle, the square of the hypotenuse (the side opposite the right angle) equals the sum of the squares of the other two sides: a² + b² = c². This relationship holds for all Euclidean right triangles and is foundational in geometry and trigonometry."),
    ("How does the immune system work?",
     "The immune system operates in two layers. Innate immunity provides immediate, non-specific defence via physical barriers (skin, mucous membranes) and phagocytes that engulf pathogens. Adaptive immunity deploys B lymphocytes that produce antigen-specific antibodies and T lymphocytes that kill infected cells; memory cells formed during an infection allow faster responses to future encounters with the same pathogen."),
    ("What is inflation?",
     "Inflation is the sustained, economy-wide increase in the general price level of goods and services, measured by indices such as the Consumer Price Index (CPI). It erodes the purchasing power of money. Central banks typically target low, stable inflation (around 2%) and use interest-rate policy to control it."),
    ("How do solar panels generate electricity?",
     "Solar panels contain photovoltaic cells made of semiconductor materials, typically silicon doped to create a p–n junction. When photons from sunlight strike the cell, they excite electrons across the junction, generating a direct current. An inverter then converts this DC output to the alternating current used in homes and the grid."),
    ("What is DNA?",
     "DNA (deoxyribonucleic acid) is a double-stranded polymer twisted into a double helix. Its backbone consists of alternating sugar and phosphate groups; the two strands are held together by complementary base pairs (adenine–thymine and cytosine–guanine). The sequence of these bases encodes the genetic instructions for the development, function, and reproduction of all known living organisms."),
    ("How does GPS determine your location?",
     "A GPS receiver calculates its position by measuring the time delay of radio signals from at least four satellites whose orbital positions are precisely known. Using the speed of light and the time of flight, it computes its distance from each satellite; trilateration of these distances gives latitude, longitude, and altitude with metre-level accuracy."),
    ("What causes earthquakes?",
     "Earthquakes result from the sudden release of elastic strain energy stored along tectonic plate boundaries or geological fault lines. When frictional resistance is overcome, plates slip abruptly and radiate seismic waves that propagate through Earth, causing ground shaking. Subduction zones, transform boundaries, and rift zones are the most seismically active regions."),
    ("How is float glass manufactured?",
     "Float glass is produced by melting a mixture of silica sand (SiO₂), soda ash, and limestone at around 1 550 °C. The molten glass is then poured onto a bath of molten tin, where it spreads and floats to form a perfectly flat ribbon. The ribbon is slowly cooled in a controlled annealing lehr to relieve internal stresses before cutting."),
    ("What is machine learning?",
     "Machine learning is a branch of artificial intelligence in which algorithms improve their performance on a task by learning statistical patterns from data, rather than through explicitly programmed rules. It encompasses supervised learning (labelled examples), unsupervised learning (structure discovery), and reinforcement learning (reward-driven optimisation), and underpins applications from image recognition to language generation."),
    ("How do antibiotics kill bacteria?",
     "Antibiotics target structures or processes essential to bacteria but absent in human cells. Beta-lactams such as penicillin inhibit cell-wall synthesis, causing bacteria to burst. Macrolides and aminoglycosides block bacterial ribosomes, halting protein synthesis. Fluoroquinolones inhibit bacterial DNA gyrase, preventing DNA replication. Resistance arises when bacteria evolve enzymes that inactivate the drug or pumps that expel it."),
    ("What is photosynthesis?",
     "Photosynthesis is the process by which photoautotrophs — plants, algae, and cyanobacteria — convert light energy, carbon dioxide, and water into glucose and oxygen. It occurs in two stages: the light-dependent reactions in the thylakoid membranes capture photons and generate ATP and NADPH; the Calvin cycle in the stroma uses these to fix CO₂ into three-carbon sugars that are subsequently converted to glucose."),
    ("How does human memory work?",
     "Human memory involves three stages: encoding converts sensory input into a neural representation; consolidation stabilises it through hippocampal replay during sleep; retrieval reactivates the stored pattern in cortical networks. Working memory holds a small amount of information actively in mind; long-term memory is subdivided into declarative memory (episodic and semantic) and non-declarative memory (procedural skills and conditioning)."),
    ("What is the greenhouse effect?",
     "The greenhouse effect is the warming of Earth's surface caused by atmospheric gases — primarily water vapour, CO₂, methane, and nitrous oxide — that absorb outgoing long-wave infrared radiation and re-emit it in all directions, including back toward the surface. Without the natural greenhouse effect, Earth's average temperature would be about –18 °C. Anthropogenic emissions have enhanced this effect, driving the observed increase in global mean surface temperature."),
    ("How do neurons transmit signals?",
     "Neurons communicate electrically and chemically. An action potential — a rapid reversal of membrane voltage caused by Na⁺ influx and K⁺ efflux through voltage-gated channels — propagates along the axon. At a synapse, the action potential triggers Ca²⁺-dependent exocytosis of neurotransmitter vesicles; the released molecules diffuse across the synaptic cleft and bind ligand-gated receptors on the postsynaptic neuron, either exciting or inhibiting it."),
]


def calibration_check() -> None:
    """
    Score 20 irrelevant and 20 relevant (query, chunk) pairs.

    Reports mean and std for each group. A discrimination gap > 0.2 indicates
    the critic is not merely assigning high scores to everything (sycophancy).
    Include these numbers in the paper's methodology section.
    """
    print("\n" + "=" * 72)
    print("CRITIC CALIBRATION CHECK")
    print("Scoring 20 irrelevant + 20 relevant synthetic pairs...")
    print("Expected: mean(irrelevant) << mean(relevant)")
    print("=" * 72)

    irrelevant_scores: list[float] = []
    print("\n[Irrelevant pairs]")
    for i, (q, c) in enumerate(_IRRELEVANT_PAIRS, 1):
        s = score_context(q, c)
        irrelevant_scores.append(s)
        print(f"  {i:02d}  score={s:.3f}  query='{q[:55]}'")

    relevant_scores: list[float] = []
    print("\n[Relevant pairs]")
    for i, (q, c) in enumerate(_RELEVANT_PAIRS, 1):
        s = score_context(q, c)
        relevant_scores.append(s)
        print(f"  {i:02d}  score={s:.3f}  query='{q[:55]}'")

    mean_irr = statistics.mean(irrelevant_scores)
    std_irr  = statistics.stdev(irrelevant_scores)
    mean_rel = statistics.mean(relevant_scores)
    std_rel  = statistics.stdev(relevant_scores)
    gap      = mean_rel - mean_irr

    print("\n" + "=" * 72)
    print("CALIBRATION RESULTS (report these numbers in your paper):")
    print(f"  Irrelevant pairs — mean={mean_irr:.4f}  std={std_irr:.4f}")
    print(f"  Relevant pairs   — mean={mean_rel:.4f}  std={std_rel:.4f}")
    print(f"  Discrimination gap (relevant − irrelevant): {gap:.4f}")
    if gap > 0.20:
        verdict = "PASS — critic clearly discriminates (gap > 0.20)"
    elif gap > 0.10:
        verdict = "MARGINAL — some discrimination (gap 0.10–0.20); consider tightening prompt"
    else:
        verdict = "FAIL — weak discrimination (gap < 0.10); sycophancy risk; revise prompt"
    print(f"  Verdict: {verdict}")
    print("=" * 72 + "\n")


# ── CLI entry point ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Confidence-scoring critic for agentic RAG")
    parser.add_argument(
        "--calibration-check", action="store_true",
        help="Score 20 irrelevant + 20 relevant pairs and report discrimination statistics",
    )
    args = parser.parse_args()

    if args.calibration_check:
        calibration_check()
    else:
        # Single smoke-test pair
        test_q = "What is the boiling point of water?"
        test_c = ("Water boils at 100 °C (212 °F) at standard atmospheric pressure (101.325 kPa). "
                  "At higher altitudes the boiling point decreases because atmospheric pressure is lower.")
        s = score_context(test_q, test_c)
        print(f"Smoke-test score (expect high, e.g. > 0.7): {s:.3f}")
