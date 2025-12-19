import math
import re
from collections import Counter
from datetime import datetime
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit
from qiskit.circuit.library import MCMTGate, ZGate
from qiskit_algorithms import Grover, AmplificationProblem
from qiskit.primitives import StatevectorSampler

matplotlib.use("Agg")


def grover_oracle(marked_states):
    if not isinstance(marked_states, list):
        marked_states = [marked_states]

    n = len(marked_states[0])
    qc = QuantumCircuit(n)

    for target in marked_states:
        rev = target[::-1]
        zero_inds = [i for i, b in enumerate(rev) if b == "0"]

        if zero_inds:
            qc.x(zero_inds)

        qc.append(MCMTGate(ZGate(), n - 1, 1), list(range(n)))

        if zero_inds:
            qc.x(zero_inds)

    return qc


def classical_find_occurrences(genome, motif):
    L = len(motif)
    windows = [genome[i:i+L] for i in range(len(genome) - L + 1)]
    good = [i for i, w in enumerate(windows) if w == motif]
    return windows, good


def grover_once(genome, motif):
    windows, good_indices = classical_find_occurrences(genome, motif)
    N = len(windows)
    if not good_indices:
        raise ValueError("Motif nao aparece no genoma.")

    n = math.ceil(math.log2(N))
    if n == 0:
        n = 1

    good_states = [format(i, f"0{n}b") for i in good_indices]
    oracle = grover_oracle(good_states)

    problem = AmplificationProblem(oracle, is_good_state=good_states)
    grover = Grover(sampler=StatevectorSampler())
    result = grover.amplify(problem)

    bits = result.top_measurement
    idx = int(bits, 2)
    return idx, good_indices, windows


def run_trials(genome, motif, trials=50):
    hits = 0
    counts = Counter()

    sample_good = None
    for _ in range(trials):
        idx, good_indices, windows = grover_once(genome, motif)
        counts[idx] += 1
        if idx in good_indices:
            hits += 1
        sample_good = good_indices

    acc = hits / trials
    return acc, counts, sample_good


def build_results(genome, motif, trials=50):
    windows, good_indices = classical_find_occurrences(genome, motif)
    acc, counts, _ = run_trials(genome, motif, trials=trials)
    total_shots = sum(counts.values())

    results = []
    for i, window in enumerate(windows):
        count = counts.get(i, 0)
        probability = (count / total_shots) * 100 if total_shots else 0.0
        results.append(
            {
                "index": i,
                "window": window,
                "count": count,
                "probability": probability,
                "is_target": i in good_indices,
            }
        )

    histogram = [
        {
            "index": idx,
            "count": count,
            "is_target": idx in good_indices,
        }
        for idx, count in sorted(counts.items())
    ]

    return {
        "accuracy": acc,
        "windows": windows,
        "good_indices": good_indices,
        "results": results,
        "histogram": histogram,
        "total_shots": total_shots,
    }


def _slugify(name):
    normalized = re.sub(r"[^a-zA-Z0-9]+", "_", name.strip().lower())
    return normalized.strip("_") or "scenario"


def _save_probability_chart(results, output_path, title):
    indices = [f"Idx {r['index']}" for r in results]
    probabilities = [r["probability"] for r in results]
    colors = ["#10b981" if r["is_target"] else "#e2e8f0" for r in results]

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(indices, probabilities, color=colors)
    ax.set_title(title)
    ax.set_ylabel("Probability (%)")
    ax.set_xlabel("Window index")
    ax.set_ylim(0, 100)
    ax.tick_params(axis="x", labelrotation=90, labelsize=7)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def _save_histogram_chart(histogram, output_path, title):
    indices = [f"Idx {r['index']}" for r in histogram]
    counts = [r["count"] for r in histogram]
    colors = ["#0ea5e9" if r["is_target"] else "#cbd5f5" for r in histogram]

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(indices, counts, color=colors)
    ax.set_title(title)
    ax.set_ylabel("Measurements")
    ax.set_xlabel("Window index")
    ax.tick_params(axis="x", labelrotation=90, labelsize=7)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def _save_results_table(results, output_path, title):
    headers = ["Index", "Window", "Count", "Prob (%)", "Target"]
    rows = [
        [
            r["index"],
            r["window"],
            r["count"],
            f"{r['probability']:.1f}",
            "yes" if r["is_target"] else "no",
        ]
        for r in results
    ]

    row_count = max(len(rows), 1)
    fig_height = min(0.35 * row_count + 1.5, 30)
    fig, ax = plt.subplots(figsize=(12, fig_height))
    ax.axis("off")
    table = ax.table(
        cellText=rows,
        colLabels=headers,
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.2)
    ax.set_title(title, pad=12)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def save_reports(genome, motif, name, output_dir, trials=50):
    results = build_results(genome, motif, trials=trials)
    slug = _slugify(name)

    output_dir.mkdir(parents=True, exist_ok=True)

    _save_probability_chart(
        results["results"],
        output_dir / f"{slug}_probability.png",
        f"{name} - Probability Distribution",
    )
    _save_histogram_chart(
        results["histogram"],
        output_dir / f"{slug}_histogram.png",
        f"{name} - Execution Histogram",
    )
    _save_results_table(
        results["results"],
        output_dir / f"{slug}_results_table.png",
        f"{name} - Results Table",
    )

    return results


def scenario(genome, motif, name, output_dir=None):
    print("\n==============================")
    print(f"CENARIO: {name}")
    print("genome:", genome)
    print("motif :", motif)

    windows, good = classical_find_occurrences(genome, motif)
    print("N janelas:", len(windows))
    print("Indices bons (classico):", good)

    acc, counts, _ = run_trials(genome, motif, trials=50)
    print("taxa de acerto (50 execucoes):", acc)
    print("top 5 indices mais medidos:", counts.most_common(5))

    if output_dir is not None:
        save_reports(genome, motif, name, output_dir, trials=50)


if __name__ == "__main__":
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(f"resultados_{timestamp}")

    scenario(
        genome="ACGTTACGTAACGTTACGTA",
        motif="ACGTT",
        name="M=2 (duas ocorrencias)",
        output_dir=output_dir,
    )

    scenario(
        genome="TTTTTACGTAACGTTACGTA",
        motif="ACGTT",
        name="M=1 (uma ocorrencia)",
        output_dir=output_dir,
    )

    scenario(
        genome="ACGTTACGTTACGTTACGTT",
        motif="ACGTT",
        name="M=4 (quatro ocorrencias)",
        output_dir=output_dir,
    )

    print(f"\nRelatorios salvos em: {output_dir.resolve()}")
