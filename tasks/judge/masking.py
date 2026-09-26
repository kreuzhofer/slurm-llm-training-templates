"""
masking.py -- assistant-only loss for multi-image judge rows, and the gate.

Separate from dataset.py on purpose. dataset.py turns a row into tensors and
accounts for their length; this module decides which of those positions the
loss is computed on, and refuses to let a run start if it cannot prove the
answer. Those are different reasons to change, and only one of them needs a
self-test that fails at import time.

WHY THIS IS ITS OWN TICKET (#22)

The SQL path masks by token arithmetic over text: tokenize the prompt, count,
mask that many (tasks/sql/dataset.py::build_example). Carrying that here is the
single most dangerous thing anyone could do to this task, because the prompt
text contains eight `<|image_pad|>` placeholders that are ONE token each as
text and 576 tokens each after the image processor runs. Counting the text
undercounts the prompt by ~4,600 positions per row, so the "answer" the model
is supervised on would start ~4,600 tokens early -- in the middle of the
rendered views -- and the labels would be visual tokens.

Nothing about that crashes. The run completes, the loss curve looks plausible,
and the adapter has learned on the wrong positions. This repo has been bitten
by exactly this shape once already: `config.use_cache = False` was a silent
no-op on the multimodal class and cost a 16-GPU job (#20). The response there
was to set every level and then ASSERT the effective one at load. Same here.

HOW THE BOUNDARY IS DEFINED

Not by searching for the assistant header, and not by counting text. By the
only definition that cannot drift from inference:

    the supervised span begins exactly where the inference-time prompt ends.

dataset.render_prompt() is what a served model is given; dataset.render_full()
is that same string plus the verdict (verified: the prompt is a strict text
prefix of the full render, asserted per row below). Run the SAME processor over
BOTH with the SAME images and the prompt's length in tokens is the boundary,
post-expansion, exactly. Two processor calls per row instead of one; on 402
rows that is ~90 extra seconds of CPU, which is not a price worth thinking
about for removing an entire class of silent error.

The template emits a pre-filled empty think block into the assistant turn
(`<|im_start|>assistant\\n<think>\\n\\n</think>\\n\\n`, thinking off -- see
dataset.py). It lands inside the prompt render, so it is masked for free, and
it must be: the server emits those tokens itself at inference, and supervising
them teaches the model to produce a SECOND empty think block after the one it
was just handed. The SQL path masks it for the same reason.

WHAT IS CHECKED, AND WHY EACH ONE EARNS ITS PLACE

check_prefix     the prompt ids are elementwise a prefix of the full ids.
                 Catches a template or tokenizer change that makes the text
                 prefix stop being a token prefix.
check_think      the prompt ends with the pre-filled think block. Catches
                 thinking being silently turned back on.
check_no_visual  no `<|image_pad|>` survives inside the supervised span. This
                 is THE multi-image check: an off-by-one-block boundary puts
                 576 visual tokens under loss, and this is what notices.
check_single_end the supervised span holds exactly one `<|im_end|>`, at its
                 end. Catches a boundary that swallowed a turn.
check_decode     the decoded supervised span equals the verdict text. The
                 direct statement of what the ticket asked for.

All five run over every row, not one sample, and report per row.
"""

import sys

IGNORE_INDEX = -100

# Thinking off, so the assistant turn opens pre-filled and empty. Written out
# rather than rebuilt from the tokenizer because it is what we are asserting
# ABOUT the template; deriving it from the template would make the check
# vacuous.
EMPTY_THINK_TAIL = "<think>\n\n</think>\n\n"


def _ids(batch):
    """input_ids of a single-row processor batch, as a flat Python list."""
    return batch["input_ids"][0].tolist()


def build_labels(processor, row, dataset_dir=None, _perturb=0):
    """
    One row -> (batch, labels), loss on the assistant turn and nothing else.

    Returns the processor's own batch (input_ids, attention_mask, pixel_values,
    image_grid_thw, mm_token_type_ids) alongside a labels list of the same
    length, IGNORE_INDEX everywhere except the verdict.

    `_perturb` shifts the boundary by that many tokens. It exists so the
    verification below can prove it catches a wrong offset instead of
    asserting that it would; nothing else may pass it.
    """
    from tasks.judge import dataset as D

    kwargs = {} if dataset_dir is None else {"dataset_dir": dataset_dir}
    images = D.load_images(row, **kwargs)

    prompt_text = D.render_prompt(processor, row)
    full_text = D.render_full(processor, row)

    # The same processor and the same images over both renders. This is the
    # whole trick: the expansion that makes text arithmetic wrong is applied
    # identically to both sides, so it cancels.
    prompt_batch = processor(text=[prompt_text], images=images, return_tensors="pt")
    full_batch = processor(text=[full_text], images=images, return_tensors="pt")

    prompt_ids = _ids(prompt_batch)
    input_ids = _ids(full_batch)

    im_end = processor.tokenizer.convert_tokens_to_ids("<|im_end|>")

    start = len(prompt_ids) + _perturb
    # End at the verdict's own <|im_end|> rather than at the sequence end: the
    # template appends a trailing newline after it, and a token that follows
    # the stop token can never be generated, so supervising it teaches nothing
    # and costs a position.
    end = max(i for i, t in enumerate(input_ids) if t == im_end) + 1

    labels = [IGNORE_INDEX] * len(input_ids)
    labels[start:end] = input_ids[start:end]

    return full_batch, labels, {
        "prompt_ids": prompt_ids,
        "input_ids": input_ids,
        "start": start,
        "end": end,
        "prompt_text": prompt_text,
        "full_text": full_text,
    }


def verify_row(processor, row, dataset_dir=None, _perturb=0):
    """
    Run all five checks on one row. Returns (ok, [failure strings]).

    Never raises on a check failure -- the caller decides whether a bad row is
    fatal -- but a row that cannot be built at all still raises, because that
    is a different problem from a mask that is wrong.
    """
    _, labels, ctx = build_labels(processor, row, dataset_dir, _perturb=_perturb)

    tok = processor.tokenizer
    prompt_ids, input_ids = ctx["prompt_ids"], ctx["input_ids"]
    start, end = ctx["start"], ctx["end"]
    image_pad = tok.convert_tokens_to_ids("<|image_pad|>")
    im_end = tok.convert_tokens_to_ids("<|im_end|>")

    failures = []

    # check_prefix
    if input_ids[: len(prompt_ids)] != prompt_ids:
        n = min(len(prompt_ids), len(input_ids))
        i = next((k for k in range(n) if prompt_ids[k] != input_ids[k]), n)
        failures.append(
            f"prompt is not a token prefix of the full render (diverges at {i}); "
            "the chat template or the tokenizer changed"
        )

    # check_think
    if not ctx["prompt_text"].endswith(EMPTY_THINK_TAIL):
        failures.append(
            "prompt does not end with the pre-filled empty think block; "
            "thinking may have been re-enabled, and the think tokens would be supervised"
        )

    # check_no_visual -- the multi-image check
    if not 0 <= start <= end <= len(input_ids):
        failures.append(f"span [{start}, {end}) is outside the sequence")
    else:
        stray = sum(1 for t in input_ids[start:end] if t == image_pad)
        if stray:
            failures.append(
                f"{stray} visual tokens fall inside the supervised span; "
                "the boundary is wrong by at least one image block"
            )

    # check_single_end
    ends = [i for i in range(start, end) if input_ids[i] == im_end]
    if ends != [end - 1]:
        failures.append(
            f"supervised span holds {len(ends)} <|im_end|> tokens at {ends}, "
            f"expected exactly one at {end - 1}"
        )

    # check_decode -- what the ticket actually asked to see
    if start < end:
        got = tok.decode(input_ids[start:end])
        want = ctx["full_text"][len(ctx["prompt_text"]):]
        if not (want.startswith(got) and want[len(got):] in ("", "\n")):
            failures.append(
                "decoded supervised span is not the verdict text\n"
                f"      got:  {got[:110]!r}…\n"
                f"      want: {want[:110]!r}…"
            )
    else:
        failures.append("supervised span is empty")

    if sum(1 for x in labels if x != IGNORE_INDEX) != max(0, end - start):
        failures.append("label count does not match the span width")

    return not failures, failures


def verify_dataset(processor, rows, dataset_dir=None, _perturb=0, verbose=True):
    """
    Every row, pass/fail each. Returns (n_ok, [(row_id, failures), ...]).

    Over all rows rather than a sample because the failure this guards against
    is data-dependent: a row with a differently sized render, or one whose
    verdict happens to contain a token that looks like a boundary, is exactly
    the row a sample would miss.
    """
    bad = []
    for i, row in enumerate(rows, 1):
        ok, failures = verify_row(processor, row, dataset_dir, _perturb=_perturb)
        if not ok:
            bad.append((row.get("id", f"#{i}"), failures))
        if verbose and (i % 50 == 0 or i == len(rows)):
            print(f"  verified {i}/{len(rows)}  ({len(bad)} failing)")
    return len(rows) - len(bad), bad


def assert_masking_is_sane(processor, rows, dataset_dir=None, sample=None):
    """
    The load-time gate. Call this from the training script BEFORE the first step.

    Raises AssertionError with the offending rows named. `sample` limits the
    check to the first N rows for a fast smoke path; leave it None for a real
    run, where the few minutes this costs are noise against the run itself and
    the alternative is discovering a wrong mask after the fact -- or never.
    """
    subset = rows if sample is None else rows[:sample]
    n_ok, bad = verify_dataset(processor, subset, dataset_dir, verbose=False)
    if bad:
        detail = "\n".join(
            f"  {rid}: {'; '.join(f)}" for rid, f in bad[:5]
        )
        raise AssertionError(
            f"assistant-only masking failed on {len(bad)} of {len(subset)} rows. "
            f"Training would optimise the wrong positions and would NOT crash.\n{detail}"
            + (f"\n  … and {len(bad) - 5} more" if len(bad) > 5 else "")
        )
    return n_ok


def pad_and_mask(batch_labels, pad_to, ignore=IGNORE_INDEX):
    """
    Right-pad a batch of label lists to a common length with IGNORE_INDEX.

    Trivial, and here rather than in the collator so that "padding contributes
    nothing to the loss" is stated and tested in the same place as the rest of
    the masking rules, instead of being an implicit property of whichever
    collator happens to be in use.
    """
    out = []
    for labels in batch_labels:
        if len(labels) > pad_to:
            raise ValueError(f"labels of length {len(labels)} exceed pad_to={pad_to}")
        out.append(list(labels) + [ignore] * (pad_to - len(labels)))
    return out


def main():
    import argparse

    from tasks.judge import dataset as D

    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument("--dataset-dir", default=D.DEFAULT_DATASET_DIR)
    parser.add_argument("--model-path", default=D.DEFAULT_MODEL_PATH)
    parser.add_argument("--limit", type=int, default=-1)
    parser.add_argument(
        "--perturb",
        type=int,
        default=0,
        help="shift the boundary by N tokens; the run is expected to FAIL, "
        "which is how we demonstrate the check catches a wrong offset rather "
        "than assuming it",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="print the decoded supervised span of the first row",
    )
    args = parser.parse_args()

    # Padding, checked before anything expensive: three label lists of
    # different lengths must come back rectangular with IGNORE_INDEX in every
    # position that was not real, and with no real label disturbed.
    padded = pad_and_mask([[1, 2, 3], [4], [5, 6]], pad_to=4)
    assert padded == [
        [1, 2, 3, IGNORE_INDEX],
        [4, IGNORE_INDEX, IGNORE_INDEX, IGNORE_INDEX],
        [5, 6, IGNORE_INDEX, IGNORE_INDEX],
    ], padded
    assert sum(1 for r in padded for x in r if x != IGNORE_INDEX) == 6
    try:
        pad_and_mask([[1, 2, 3]], pad_to=2)
    except ValueError:
        pass
    else:
        raise AssertionError("pad_and_mask silently truncated labels past pad_to")
    print("padding: 3 ragged rows -> rectangular, 6 real labels kept, "
          "over-length refused")

    rows = D.load_rows(args.dataset_dir)
    if 0 <= args.limit < len(rows):
        rows = rows[: args.limit]
    print(f"{len(rows)} rows from {args.dataset_dir}")

    processor = D.load_processor(args.model_path)

    if args.show:
        _, labels, ctx = build_labels(processor, rows[0], args.dataset_dir)
        ids = ctx["input_ids"]
        kept = [t for t, l in zip(ids, labels) if l != IGNORE_INDEX]
        print(f"\nrow {rows[0].get('id')}: {len(ids)} tokens, "
              f"supervised [{ctx['start']}, {ctx['end']}) = {len(kept)} tokens")
        print("--- masked tail of the prompt ---")
        print(repr(processor.tokenizer.decode(ids[ctx["start"] - 12:ctx["start"]])))
        print("--- supervised span (first 300 chars) ---")
        print(processor.tokenizer.decode(kept)[:300])
        print()

    n_ok, bad = verify_dataset(processor, rows, args.dataset_dir, _perturb=args.perturb)

    print(f"\n{n_ok}/{len(rows)} rows pass all five checks")
    for rid, failures in bad[:5]:
        print(f"  FAIL {rid}")
        for f in failures:
            print(f"    - {f}")
    if len(bad) > 5:
        print(f"  … and {len(bad) - 5} more failing rows")

    if args.perturb:
        # Inverted: with a deliberate offset, passing is the failure.
        if bad:
            print(f"\nEXPECTED: --perturb {args.perturb} made {len(bad)} rows fail. "
                  "The check catches a wrong offset.")
            return 0
        print(f"\nBUG: --perturb {args.perturb} changed the boundary and every row "
              "still passed. The checks are not actually checking.")
        return 1

    return 0 if not bad else 1


if __name__ == "__main__":
    sys.exit(main())
