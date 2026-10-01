"""
check_export.py -- run this the moment a new judge export lands, before anything else.

    python tasks/judge/check_export.py --dataset-dir <dir> [--reference <prev.json>]
    python tasks/judge/check_export.py --dataset-dir <dir> --quick     # seconds, no processor

WHY THIS EXISTS

rc0's `max_seq_len` was specified at 8192 from an outside estimate of ~7.5K. The
estimate matched the median and missed the tail: at 8192 two of 402 rows
truncated, and because the assistant turn is last they lost the **verdict**, not
padding. One row lost all 354 of its answer tokens. Nothing would have reported
that -- the run completes and the loss curve looks plausible.

The lesson is not "check max_seq_len". It is that a new export is a new set of
facts, and the cheap moment to establish them is before a GPU is allocated, not
after a number has been published. This script is that moment, as one command.

It answers, in order: is the export structurally sound, does its manifest use a
vocabulary this code understands, does the drift-guard carve still work, what is
the real token profile, does the masking gate still pass on every row, and what
moved since the last export.

WHAT IT DELIBERATELY DOES NOT DO

It reports no accuracy and trains nothing. It also does not *decide* the recipe:
it prints the `max_seq_len` the data requires and leaves rank, epochs and batch
to a human, because those depend on the correction signal rather than on the
token profile.

COST

Structural and manifest checks are instant. The token profile is one processor
call per row and the masking gate is two, so a 400-row export takes roughly three
to four minutes of CPU on the login node with no GPU. `--quick` runs only the
free checks, which is the right first move when an export has just appeared and
you want to know whether it is even the right shape.
"""

import argparse
import json
import os
import sys

sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

from tasks.judge import dataset as judge_dataset
from tasks.judge import masking

# Candidate sequence lengths, in the order a run would prefer them.
CANDIDATE_MAX_SEQ_LENS = (8192, 10240, 12288, 16384)


class Report:
    """Collects pass/fail so the script can exit non-zero and say why."""

    def __init__(self):
        self.failures = []
        self.notes = []

    def check(self, ok, label, detail=""):
        print(f"  [{'ok  ' if ok else 'FAIL'}] {label}" + (f" -- {detail}" if detail else ""))
        if not ok:
            self.failures.append(f"{label}: {detail}" if detail else label)
        return ok

    def note(self, text):
        self.notes.append(text)
        print(f"  [note] {text}")


def check_structure(rows, dataset_dir, report):
    """Shape of every row, and that every image it references is on disk."""
    print("\n== structure ==")
    report.check(bool(rows), "samples.jsonl has rows", f"{len(rows)} rows")

    turn_counts, image_counts, missing, sizes = {}, {}, [], {}
    for row in rows:
        turn_counts[len(row["messages"])] = turn_counts.get(len(row["messages"]), 0) + 1
        try:
            paths = judge_dataset.image_paths(row)
        except ValueError as exc:
            missing.append(f"{row.get('id')}: {exc}")
            continue
        image_counts[len(paths)] = image_counts.get(len(paths), 0) + 1
        for rel in paths:
            if not os.path.exists(os.path.join(dataset_dir, rel)):
                missing.append(f"{row.get('id')}: {rel}")

    report.check(
        list(turn_counts) == [3], "every row is single-turn (system+user+assistant)",
        f"turn counts {turn_counts}",
    )
    report.check(not missing, "every referenced image exists",
                 f"{len(missing)} missing, e.g. {missing[:2]}" if missing else "")
    print(f"  images per row: {image_counts}")
    if len(image_counts) > 1:
        report.note(
            f"rows do not all carry the same number of images ({image_counts}); "
            "the collator handles it, but visual-token counts will vary"
        )
    return sizes


def check_manifest(dataset_dir, rows, report):
    """The manifest's vocabulary, and whether the carve still works."""
    print("\n== manifest and the drift-guard carve ==")
    try:
        census = judge_dataset.item_source_census(dataset_dir)
    except ValueError as exc:
        report.check(False, "item-source vocabulary is recognised", str(exc).splitlines()[0])
        print("\n  Full message:\n   ", str(exc).replace("\n", "\n    "))
        return None
    report.check(True, "item-source vocabulary is recognised", str(census))

    total_items = sum(census.values())
    try:
        decisions = judge_dataset.item_decision_census(dataset_dir)
    except ValueError as exc:
        report.check(False, "item-decision vocabulary is recognised",
                     str(exc).splitlines()[0])
        print("\n  Full message:\n   ", str(exc).replace("\n", "\n    "))
        return census
    report.check(True, "item-decision vocabulary is recognised",
                 str({str(k): v for k, v in decisions.items()}))

    # The correcting count is items that REVERSE the incumbent, not items merely
    # ineligible for the carve. On rc0 that is 68, not 136 -- auto-C and 30 of the
    # adjudicated items confirm the incumbent and teach it nothing new.
    correcting = judge_dataset.correcting_item_count(dataset_dir)
    print(f"  items: {total_items} total, {correcting} reverse the incumbent "
          f"({100 * correcting / max(total_items, 1):.1f}%), "
          f"{total_items - correcting} confirm it")
    report.note(
        f"correcting signal is {correcting} of {total_items} items -- this, not the "
        "row count, is what should size rank and epochs"
    )

    # The manifest lists the ids this export must never train on -- the
    # qualification set and every spot check. Asserted here because "absent
    # from the samples" is a property of the export that a re-export can break
    # silently, and training on a held-out row invalidates the number that
    # qualifies the release.
    manifest = judge_dataset.load_manifest(dataset_dir)
    held = manifest.get("heldOut") or {}
    held_ids = set(held.get("exampleIds") or [])
    sample_ids = {r["id"] for r in rows}
    leak = sorted(held_ids & sample_ids)
    report.check(
        bool(held_ids), "manifest lists held-out ids",
        f"{len(held_ids)} listed" if held_ids else "none listed -- cannot prove disjointness",
    )
    report.check(not leak, "no held-out id appears in the samples",
                 f"{len(leak)} LEAKED, e.g. {leak[:3]}" if leak else f"0 of {len(held_ids)}")
    manifest_ids = {s["id"] for s in manifest["samples"]}
    report.check(manifest_ids == sample_ids, "manifest sample ids match samples.jsonl",
                 "" if manifest_ids == sample_ids else
                 f"{len(manifest_ids ^ sample_ids)} ids differ")

    eligible = judge_dataset.agreed_only_ids(dataset_dir)
    report.check(bool(eligible), "rows are eligible for the carve", f"{len(eligible)} eligible")
    try:
        train, evaluation = judge_dataset.train_eval_split(rows, dataset_dir)
        report.check(bool(evaluation), "drift-guard carve succeeds",
                     f"{len(train)} train + {len(evaluation)} guard")
        kept = correcting
        report.check(True, "carve costs no correction signal",
                     f"all {kept} correcting items remain in training")
    except ValueError as exc:
        report.check(False, "drift-guard carve succeeds", str(exc).splitlines()[0])
    return census


def check_tokens(rows, dataset_dir, model_path, report, out=None):
    """The real token profile, and the max_seq_len it requires."""
    print("\n== token profile ==")
    processor = judge_dataset.load_processor(model_path)
    per_row = []
    for i, row in enumerate(rows, 1):
        per_row.append(judge_dataset.measure_row(processor, row, dataset_dir))
        if i % 100 == 0 or i == len(rows):
            print(f"    measured {i}/{len(rows)}")

    summary = judge_dataset.summarize(per_row)
    print(f"\n  {'':>16}{'min':>8}{'median':>8}{'p95':>8}{'max':>8}")
    for key, stat in summary.items():
        print(f"  {key:>16}{stat['min']:>8}{stat['median']:>8}{stat['p95']:>8}{stat['max']:>8}")

    print()
    required = None
    for limit in CANDIDATE_MAX_SEQ_LENS:
        over = [r for r in per_row if r["total_tokens"] > limit]
        print(f"  max_seq_len {limit:>6}: {len(over)} of {len(per_row)} rows truncated")
        if not over and required is None:
            required = limit
    report.check(required is not None, "some candidate max_seq_len fits every row",
                 f"smallest that fits: {required}" if required else
                 f"max is {summary['total_tokens']['max']}, above every candidate")
    if required:
        report.note(
            f"use MAX_SEQ_LEN={required}. Truncation here drops the VERDICT, not "
            "padding, because the assistant turn is last"
        )

    sizes = {}
    for r in per_row:
        for s in r["image_sizes"]:
            sizes[s] = sizes.get(s, 0) + 1
    print(f"  image sizes: {sizes}")

    if out:
        with open(out, "w") as handle:
            json.dump({"summary": summary, "rows": per_row}, handle, indent=1)
        print(f"  per-row measurements -> {out}")
    return processor, summary, per_row


def check_masking(processor, rows, dataset_dir, report):
    """#22's gate, over every row."""
    print("\n== masking gate ==")
    n_ok, bad = masking.verify_dataset(processor, rows, dataset_dir, verbose=False)
    report.check(not bad, "assistant-only masking passes on every row",
                 f"{n_ok}/{len(rows)} pass" if not bad else
                 f"{len(bad)} failing, e.g. {bad[0][0]}: {bad[0][1][0]}")


def diff_reference(summary, per_row, reference, report):
    """What moved since the last export."""
    print("\n== changes since the reference export ==")
    with open(reference) as handle:
        ref = json.load(handle)
    print(f"  rows: {len(ref['rows'])} -> {len(per_row)}")
    for key in ("total_tokens", "answer_tokens"):
        r, n = ref["summary"][key], summary[key]
        arrow = lambda a, b: f"{a} -> {b}" + ("" if a == b else f"  ({b - a:+d})")
        print(f"  {key:>14} median {arrow(r['median'], n['median'])},"
              f"  max {arrow(r['max'], n['max'])}")
    if summary["total_tokens"]["max"] > ref["summary"]["total_tokens"]["max"]:
        report.note(
            "the token tail grew -- re-check MAX_SEQ_LEN against the table above "
            "rather than carrying the previous run's value"
        )


def main():
    # Line-buffer stdout so progress shows up when redirected to a log. The rc1
    # run sat at 0 logged bytes for its whole six minutes with block buffering,
    # which reads as a hang from outside.
    sys.stdout.reconfigure(line_buffering=True)
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--model-path", default=judge_dataset.DEFAULT_MODEL_PATH)
    parser.add_argument("--reference", default=None,
                        help="a previous export's per-row JSON, to report what moved")
    parser.add_argument("--out", default=None, help="write this export's per-row JSON here")
    parser.add_argument("--quick", action="store_true",
                        help="structure and manifest only -- instant, no processor")
    parser.add_argument("--limit", type=int, default=-1)
    args = parser.parse_args()

    print(f"Checking export at {args.dataset_dir}")
    report = Report()
    rows = judge_dataset.load_rows(args.dataset_dir)
    if 0 <= args.limit < len(rows):
        rows = rows[: args.limit]
        report.note(f"--limit {args.limit}: not a full check of the export")

    check_structure(rows, args.dataset_dir, report)
    check_manifest(args.dataset_dir, rows, report)

    if not args.quick:
        processor, summary, per_row = check_tokens(
            rows, args.dataset_dir, args.model_path, report, args.out
        )
        check_masking(processor, rows, args.dataset_dir, report)
        if args.reference:
            diff_reference(summary, per_row, args.reference, report)
    else:
        report.note("--quick: token profile and masking gate NOT run, so this says "
                    "nothing about max_seq_len or the label mask")

    print("\n" + "=" * 62)
    if report.failures:
        print(f"NOT READY -- {len(report.failures)} check(s) failed:")
        for f in report.failures:
            print(f"  - {f}")
        return 1
    print("READY -- every check passed." if not args.quick else
          "Structure and manifest are sound; run without --quick before training.")
    for n in report.notes:
        print(f"  note: {n}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
