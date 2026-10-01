"""
dataset.py -- the JUDGE task definition: CAD renders in, a JSON verdict out.

The multimodal counterpart of the SQL task. Eight orthographic renders of a
Build123d model plus the original text request go in; the target is the
reference judge's verdict as strict JSON (score 1-10, issues, suggestions, and
a checklist of {question, pass, detail}).

WHY THIS IS A SEPARATE MODULE, NOT A CHANGE TO tasks/sql/dataset.py

The SQL task used to live in a directory called `common/`, named as though it
were task-neutral. It never was: it *is* the SQL task, and it is what produced
the numbers in docs/RESULTS.md. Issue #18 settled the move to `tasks/sql/` +
`tasks/judge/`, and that has now landed.

Nothing here imports from tasks/sql and nothing there imports from here. That
is deliberate: the two tasks must be able to drift apart without either one's
published numbers moving. Shared code between tasks would recreate exactly the
trap `common/` was.

THINKING IS OFF, DELIBERATELY

#18 decided this with a reason rather than inheriting it from the SQL flow:
none of the targets contain a <think> block and no system prompt mentions
thinking, so there is no reasoning channel to supervise; and both the candidate
and the reference labels were produced with thinking off, so a later comparison
against them is only valid under the same setting. vLLM's default for this
checkpoint is thinking ON (#17), so this has to be forced at every call site,
never assumed.

WHAT THE PROCESSOR DOES TO THE TOKEN COUNT

The chat template accepts the export's OpenAI-style content parts as they are
-- `{"type": "image_url", "image_url": {"url": ...}}` renders to one
<|vision_start|><|image_pad|><|vision_end|> triple per image. The *expansion*
of that single placeholder into per-patch tokens happens inside the processor
call, once real pixels are supplied, so a token count taken from the rendered
text alone undercounts by three orders of magnitude.

Measured on this checkpoint: patch_size 16 with merge_size 2 means one image
token per 32x32 pixels, so a 768x768 render costs exactly 576 tokens and a
512x512 render exactly 256. No resizing happens at these sizes -- the image
processor bounds *area*, not edge length (65,536 to 16,777,216 pixels), and
768x768 = 589,824 sits inside that window.
"""

import json
import os
import statistics

DEFAULT_DATASET_DIR = "/mnt/data/slurm-llm-templates/datasets/judge-sft-rc0"
DEFAULT_MODEL_PATH = "/mnt/data/slurm-llm-templates/models/Qwen3.8-27B"

# Never /tmp: it is node-local, so a worker rank cannot see what the launcher
# wrote there. Shared artifacts live under /mnt/data.
assert not DEFAULT_DATASET_DIR.startswith("/tmp")


def load_rows(dataset_dir=DEFAULT_DATASET_DIR):
    """Read samples.jsonl. One row per line, in file order."""
    with open(os.path.join(dataset_dir, "samples.jsonl")) as handle:
        return [json.loads(line) for line in handle if line.strip()]


def image_paths(row):
    """
    The row's image paths, in the order the user turn refers to them.

    Taken from the message content rather than from row["images"], because the
    content order is what the template interleaves with the view labels
    ("Front view:", "Back view:", ...) and therefore what the model actually
    sees. The two agree in the rc0 export; reading the content means they
    cannot silently stop agreeing.

    BOTH part shapes are accepted, and that is not defensive padding. The chat
    template REWRITES the export's OpenAI-style
        {"type": "image_url", "image_url": {"url": ...}}
    into
        {"type": "image", "url": ...}
    and it does so IN PLACE on the dict it was handed. render_prompt() and
    render_full() now deep-copy to stop that, but a row that has been through
    any other template call still arrives in the rewritten shape, and matching
    only "image_url" would quietly return [] for it.

    Empty is an error, never a result: every row in this task carries eight
    views. Without this raise, [] flows into the processor and dies ~6 frames
    down in image_transforms.py as `IndexError: list index out of range`,
    which points at transformers rather than at the row.
    """
    parts = row["messages"][1]["content"]
    paths = []
    for part in parts:
        if part.get("type") == "image_url":
            paths.append(part["image_url"]["url"])
        elif part.get("type") == "image" and "url" in part:
            paths.append(part["url"])
    if not paths:
        raise ValueError(
            f"row {row.get('id')}: no image parts in the user turn. The row was "
            "probably mutated by a chat-template call that rewrote its content "
            "in place -- render from a copy (see render_prompt)."
        )
    return paths


def load_images(row, dataset_dir=DEFAULT_DATASET_DIR):
    from PIL import Image

    images = []
    for rel in image_paths(row):
        path = os.path.join(dataset_dir, rel)
        if not os.path.exists(path):
            raise FileNotFoundError(f"row {row['id']}: missing image {rel}")
        images.append(Image.open(path).convert("RGB"))
    return images


# The item-source taxonomy, held as data the code checks rather than a string
# buried in a comprehension. This is the ONE place a manifest change can silently
# alter what gets held out, and it has already changed once: `auto-C` did not
# exist in the first export and appeared in the second.
#
# The failure it guards against is not hypothetical. With the source compared
# against a bare literal, renaming or respelling "agreed" upstream empties the
# eligible pool, sample(..., min(n_eval, 0)) picks nothing, and train_eval_split
# returns every row as training with ZERO drift-guard rows. The run then
# completes, reports no preservation signal, and says nothing about why.
KNOWN_ITEM_SOURCES = {
    "agreed": "both judges agreed; carries no correction signal",
    "adjudicated": "a human overturned or confirmed the incumbent",
    "auto-C": "the one-sided rule confirmed the incumbent",
}

# The per-item decision codes. `R` reverses the incumbent's verdict, `C` confirms
# it, and an agreed item carries no decision at all. This is the field that
# separates the items which actually teach something from the ones that only say
# "you were already right": on the rc0 export, 68 items are R and 521 are C or
# agreed, which is 12% correcting -- NOT the 136 you get by counting everything
# that is merely ineligible for the carve.
KNOWN_ITEM_DECISIONS = {
    "R": "reversed -- the incumbent's verdict was overturned",
    "C": "confirmed -- the incumbent's verdict stood",
    None: "no adjudication; the judges agreed",
}
CORRECTING_DECISIONS = {"R"}


def item_decision_census(dataset_dir=DEFAULT_DATASET_DIR):
    """
    {decision: count} over every item, refusing to guess about a new code.

    Guarded the same way as the source vocabulary and for the same reason: the
    correcting-item count is what sizes rank and epochs, so a decision code this
    code does not understand must stop the run rather than be silently bucketed.
    """
    counts = {}
    for sample in load_manifest(dataset_dir)["samples"]:
        for item in sample.get("items", []):
            counts[item.get("decision")] = counts.get(item.get("decision"), 0) + 1
    unknown = sorted(str(k) for k in set(counts) - set(KNOWN_ITEM_DECISIONS))
    if unknown:
        raise ValueError(
            f"manifest.json in {dataset_dir} contains item decision code(s) this "
            f"code does not recognise: {unknown}.\n"
            f"Known: {sorted(str(k) for k in KNOWN_ITEM_DECISIONS)}.\n"
            "Add each to KNOWN_ITEM_DECISIONS, and to CORRECTING_DECISIONS if it "
            "means the incumbent was overturned. The correcting count sizes rank "
            "and epochs, so mis-bucketing one inflates or deflates the recipe."
        )
    return counts


def correcting_item_count(dataset_dir=DEFAULT_DATASET_DIR):
    """How many items actually overturn the incumbent. 68 on the rc0 export."""
    census = item_decision_census(dataset_dir)
    return sum(v for k, v in census.items() if k in CORRECTING_DECISIONS)


# Only all-agreed rows may be held out. A row carrying an adjudicated or auto-C
# item is one where a human (or the rule) ruled on the incumbent, and those items
# ARE the training signal -- holding one back spends correction signal to measure
# something a held-out loss cannot see anyway.
HELD_OUT_ELIGIBLE_SOURCES = {"agreed"}

# The drift-guard carve is defined exactly once, here, so a training run and
# anything that later inspects the split cannot disagree about which rows were
# held back.
SPLIT_SEED = 42
DEFAULT_N_EVAL = 10


def load_manifest(dataset_dir=DEFAULT_DATASET_DIR):
    """The export's manifest.json."""
    with open(os.path.join(dataset_dir, "manifest.json")) as handle:
        return json.load(handle)


def item_source_census(dataset_dir=DEFAULT_DATASET_DIR):
    """
    {source: count} over every checklist item, refusing to guess about a new one.

    An unknown source is fatal rather than silently treated as non-agreed,
    because "silently treated as" is how the drift guard disappears. If an
    export introduces a source, decide whether it is eligible for the carve and
    add it to KNOWN_ITEM_SOURCES -- the decision is cheap, the silence is not.
    """
    counts = {}
    for sample in load_manifest(dataset_dir)["samples"]:
        for item in sample.get("items", []):
            counts[item.get("source")] = counts.get(item.get("source"), 0) + 1

    unknown = sorted(str(k) for k in set(counts) - set(KNOWN_ITEM_SOURCES))
    if unknown:
        raise ValueError(
            f"manifest.json in {dataset_dir} contains item source(s) this code "
            f"does not recognise: {unknown}.\n"
            f"Known: {sorted(KNOWN_ITEM_SOURCES)}.\n"
            "Decide whether each new source is eligible for the agreed-only "
            "drift-guard carve and add it to KNOWN_ITEM_SOURCES (and to "
            "HELD_OUT_ELIGIBLE_SOURCES if it carries no correction signal). "
            "Refusing rather than guessing: treating an unknown source as "
            "non-agreed shrinks the carve silently, and a renamed 'agreed' "
            "would empty it entirely."
        )
    return counts


def agreed_only_ids(dataset_dir=DEFAULT_DATASET_DIR):
    """
    Row ids whose every checklist item is eligible to be held out.

    On the rc0 export that is 314 of the 402 rows. The census runs first, so a
    manifest carrying an unseen source fails here rather than quietly changing
    what this returns.
    """
    item_source_census(dataset_dir)  # raises on an unrecognised source
    return [
        s["id"]
        for s in load_manifest(dataset_dir)["samples"]
        if s.get("items")
        and all(i.get("source") in HELD_OUT_ELIGIBLE_SOURCES for i in s["items"])
    ]


def drop_auto_c_rows(rows, dataset_dir=DEFAULT_DATASET_DIR):
    """
    (kept_rows, dropped_ids) -- remove rows whose only non-agreed items are auto-C.

    auto-C items are confirmations of the incumbent made by a triage model at
    high confidence and never seen by a human. They teach nothing the 892
    agreed items do not, and if the triage model shares the incumbent's
    false-pass tendency they reinforce it. Dropping them is defensible; so is
    keeping them. This makes the choice one environment variable.

    Rows that carry BOTH an auto-C item and a human adjudication are KEPT: the
    loss is on the whole verdict, there is no per-item weighting, and dropping
    such a row would throw away the human label to avoid the machine one. On
    the rc1 export that is 51 rows dropped and 6 kept, costing 0 human items.
    """
    manifest = load_manifest(dataset_dir)
    by_id = {s["id"]: s for s in manifest["samples"]}
    dropped = set()
    for row in rows:
        items = by_id.get(row["id"], {}).get("items", [])
        sources = {i.get("source") for i in items}
        if "auto-C" in sources and sources <= {"auto-C", "agreed"}:
            dropped.add(row["id"])
    return [r for r in rows if r["id"] not in dropped], sorted(dropped)


def train_eval_split(rows, dataset_dir=DEFAULT_DATASET_DIR, n_eval=DEFAULT_N_EVAL):
    """
    (train_rows, eval_rows) -- every row trains except a small agreed-only carve.

    This is a DRIFT GUARD, not a holdout, and the difference matters enough to
    say twice. The real evaluation lives on the consuming side, on a held-out set
    disjoint from this export by example id and by prompt. Nothing measured here
    is an accuracy, and a loss computed on agreed-only rows measures
    PRESERVATION: it should sit near-flat, and a rise means the adapter is
    drifting off what the base already agreed with. The per-epoch TRAIN loss is
    the one expected to move.

    Deterministic: sorted ids, then a seeded shuffle, so two runs of the same
    export carve the same rows without writing a split file.

    Asserts that it carved what it was asked for. Silently returning an empty
    carve would leave the run with no preservation signal and nothing in the log
    saying so.
    """
    import random

    eligible = sorted(set(agreed_only_ids(dataset_dir)) & {r["id"] for r in rows})
    want = min(n_eval, len(eligible))
    picked = set(random.Random(SPLIT_SEED).sample(eligible, want))
    train = [r for r in rows if r["id"] not in picked]
    evaluation = [r for r in rows if r["id"] in picked]

    if n_eval > 0 and not evaluation:
        raise ValueError(
            f"asked for {n_eval} drift-guard rows and carved none. "
            f"{len(eligible)} of {len(rows)} rows are eligible "
            f"({sorted(HELD_OUT_ELIGIBLE_SOURCES)} items only). Either the "
            "manifest and samples.jsonl disagree about ids, or no row is "
            "all-agreed. Training without a preservation signal is a decision, "
            "not a default -- pass n_eval=0 to make it one."
        )
    if len(evaluation) != want:
        raise ValueError(
            f"carve is {len(evaluation)} rows, expected {want}; manifest ids and "
            "samples.jsonl ids disagree"
        )
    return train, evaluation


def load_processor(model_path=DEFAULT_MODEL_PATH):
    from transformers import AutoProcessor

    return AutoProcessor.from_pretrained(model_path)


def render_prompt(processor, row):
    """
    The inference-time prompt: system + user, generation prompt appended.

    Thinking off -- see the module docstring. With enable_thinking=False and
    add_generation_prompt=True the template emits a pre-filled empty think
    block, exactly as it does on the SQL path; masking it is #22's problem, not
    this module's.

    Deep-copied because apply_chat_template MUTATES the messages it is given,
    rewriting each {"type": "image_url", "image_url": {...}} part into
    {"type": "image", "url": ...}. The row survives one render and loses its
    images to the next caller -- which is invisible in a measurement pass that
    touches every row once, and fatal in training, where every row is rendered
    again on the second epoch.
    """
    import copy

    return processor.apply_chat_template(
        copy.deepcopy(row["messages"][:-1]),
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )


def render_full(processor, row):
    """
    The full training sequence: prompt plus the target verdict.

    Deep-copied for the same reason as render_prompt().
    """
    import copy

    return processor.apply_chat_template(
        copy.deepcopy(row["messages"]),
        tokenize=False,
        add_generation_prompt=False,
        enable_thinking=False,
    )


def build_example(processor, row, dataset_dir=DEFAULT_DATASET_DIR):
    """
    One row -> model-ready tensors.

    Returns the processor's own output: input_ids, attention_mask,
    pixel_values, image_grid_thw and mm_token_type_ids. Labels are not built
    here -- the assistant-only mask is #22, and putting a half-considered mask
    here would be the drift this repo keeps warning about.
    """
    images = load_images(row, dataset_dir)
    return processor(
        text=[render_full(processor, row)], images=images, return_tensors="pt"
    )


def measure_row(processor, row, dataset_dir=DEFAULT_DATASET_DIR):
    """
    Token accounting for one row, split into visual and text.

    answer_tokens is measured on text alone (full render minus prompt render,
    both untokenized-by-the-image-processor), which is exact because the
    assistant turn contains no images and therefore no expansion.
    """
    batch = build_example(processor, row, dataset_dir)
    ids = batch["input_ids"][0]
    image_pad_id = processor.tokenizer.convert_tokens_to_ids("<|image_pad|>")
    visual = int((ids == image_pad_id).sum())
    total = int(ids.numel())

    tok = processor.tokenizer
    prompt_text_tokens = len(tok(render_prompt(processor, row), add_special_tokens=False)["input_ids"])
    full_text_tokens = len(tok(render_full(processor, row), add_special_tokens=False)["input_ids"])

    grid = batch["image_grid_thw"].tolist()
    return {
        "id": row["id"],
        "total_tokens": total,
        "visual_tokens": visual,
        "text_tokens": total - visual,
        "answer_tokens": full_text_tokens - prompt_text_tokens,
        "n_images": len(grid),
        "image_sizes": [f"{h * 16}x{w * 16}" for _, h, w in grid],
        "tokens_per_image": [int(h * w / 4) for _, h, w in grid],
    }


def summarize(per_row):
    """min / median / p95 / max over a list of measure_row() results."""

    def stats(key):
        values = sorted(r[key] for r in per_row)
        n = len(values)
        # Nearest-rank p95: the smallest value at or above 95% of the rows.
        # With n=152 that is values[144]. Stated explicitly because "p95" is
        # ambiguous enough that two implementations disagree by a row or two,
        # and max_seq_len gets justified by this number.
        p95 = values[min(n - 1, (95 * n + 99) // 100 - 1)]
        return {
            "min": values[0],
            "median": int(statistics.median(values)),
            "p95": p95,
            "max": values[-1],
        }

    return {k: stats(k) for k in ("total_tokens", "visual_tokens", "text_tokens", "answer_tokens")}


def main():
    import argparse

    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument("--dataset-dir", default=DEFAULT_DATASET_DIR)
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    parser.add_argument(
        "--out",
        default=None,
        help="write per-row measurements here as JSON (recommended: the row "
        "count is not settled upstream, so an aggregate alone does not survive "
        "a re-export)",
    )
    parser.add_argument("--limit", type=int, default=-1)
    args = parser.parse_args()

    rows = load_rows(args.dataset_dir)
    if 0 <= args.limit < len(rows):
        rows = rows[: args.limit]
    print(f"{len(rows)} rows from {args.dataset_dir}")

    processor = load_processor(args.model_path)
    per_row = []
    for i, row in enumerate(rows, 1):
        per_row.append(measure_row(processor, row, args.dataset_dir))
        if i % 20 == 0 or i == len(rows):
            print(f"  measured {i}/{len(rows)}")

    summary = summarize(per_row)
    print()
    print(f"{'':>16}{'min':>8}{'median':>8}{'p95':>8}{'max':>8}")
    for key, s in summary.items():
        print(f"{key:>16}{s['min']:>8}{s['median']:>8}{s['p95']:>8}{s['max']:>8}")

    counts = {}
    for r in per_row:
        counts[r["n_images"]] = counts.get(r["n_images"], 0) + 1
    print(f"\nimages per row: {counts}")
    sizes = {}
    for r in per_row:
        for s in r["image_sizes"]:
            sizes[s] = sizes.get(s, 0) + 1
    print(f"image sizes: {sizes}")

    for limit in (8192, 10240, 12288, 16384):
        over = sum(1 for r in per_row if r["total_tokens"] > limit)
        print(f"max_seq_len {limit:>6}: {over} of {len(per_row)} rows truncated")

    if args.out:
        with open(args.out, "w") as handle:
            json.dump({"summary": summary, "rows": per_row}, handle, indent=1)
        print(f"\nper-row measurements -> {args.out}")


if __name__ == "__main__":
    main()
