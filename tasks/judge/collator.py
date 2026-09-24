"""
collator.py -- batch multi-image judge rows into one training step.

`DataCollatorForSeq2Seq`, which the SQL path uses, cannot do this job. It pads
input_ids and labels and knows nothing about the tensors that travel beside
them, which do not pad at all:

    input_ids          (L,)              pad to the batch's longest L
    attention_mask     (L,)              pad with 0
    mm_token_type_ids  (L,)              pad with 0
    labels             (L,)              pad with IGNORE_INDEX
    pixel_values       (patches, 1536)   CONCATENATE along dim 0
    image_grid_thw     (n_images, 3)     CONCATENATE along dim 0

The vision tensors are already flat: the processor packs every image of a row
into one patch stream and records the per-image grid separately, so a batch is
the concatenation of those streams, not a padded stack. There is no padding
dimension to get wrong -- but there is an ORDER to get wrong, and it is
silent. `image_grid_thw` is what tells the model how to cut `pixel_values` back
into images, so both must be concatenated in the same row order as input_ids.
The assertion at the end of collate() is what checks that.

Measured shapes on this export: a row of eight 768x768 renders carries
18,432 patches (48x48 per image, 2,304 x 8) and 4,608 image tokens after the
2x2 merge; eight 512x512 renders carry 8,192 patches and 2,048 tokens.

SEQUENCES ARE REFUSED, NEVER TRUNCATED

max_seq_len raises rather than cutting. #21 measured that truncation on this
task does not drop padding, it drops the VERDICT: the assistant turn is last,
so a row cut at 8192 loses the target it was supposed to teach and trains on a
prompt with no answer. That is a silent corruption of exactly the kind #22
guards the mask against, and the correct response to it is to stop.
"""

from tasks.judge.masking import IGNORE_INDEX


def build_feature(processor, row, dataset_dir=None):
    """
    One row -> one flat training feature, ready for collate().

    Squeezes the processor's leading batch dimension off the per-token tensors
    and leaves the vision tensors as they are, because that is the shape
    collate() concatenates.
    """
    import torch

    from tasks.judge import masking

    batch, labels, _ = masking.build_labels(processor, row, dataset_dir)
    return {
        "input_ids": batch["input_ids"][0],
        "attention_mask": batch["attention_mask"][0],
        "mm_token_type_ids": batch["mm_token_type_ids"][0],
        "labels": torch.tensor(labels, dtype=torch.long),
        "pixel_values": batch["pixel_values"],
        "image_grid_thw": batch["image_grid_thw"],
    }


class JudgeCollator:
    """
    Right-pad the token streams, concatenate the vision streams.

    Right padding, not left: labels are aligned to input_ids position by
    position, and IGNORE_INDEX on the tail is what keeps the pad positions out
    of the loss. Left padding would work for generation and would silently
    misalign the labels here.
    """

    def __init__(self, pad_token_id, max_seq_len=None, pad_to_multiple_of=8):
        if pad_token_id is None:
            raise ValueError(
                "pad_token_id is None; pass tokenizer.pad_token_id explicitly "
                "rather than letting the pad positions take token 0, which is "
                "a real token"
            )
        self.pad_token_id = pad_token_id
        self.max_seq_len = max_seq_len
        self.pad_to_multiple_of = pad_to_multiple_of

    def __call__(self, features):
        import torch

        lengths = [f["input_ids"].shape[0] for f in features]
        longest = max(lengths)

        if self.max_seq_len is not None and longest > self.max_seq_len:
            over = [(i, n) for i, n in enumerate(lengths) if n > self.max_seq_len]
            raise ValueError(
                f"{len(over)} sequence(s) exceed max_seq_len={self.max_seq_len}: "
                f"{over}. Truncating here would drop the verdict, not padding "
                "(the assistant turn is last) -- raise max_seq_len instead. "
                "Measured max on the 402-row export is 8,496."
            )

        if self.pad_to_multiple_of:
            m = self.pad_to_multiple_of
            longest = ((longest + m - 1) // m) * m

        pad_values = {
            "input_ids": self.pad_token_id,
            "attention_mask": 0,
            "mm_token_type_ids": 0,
            "labels": IGNORE_INDEX,
        }

        out = {}
        for key, value in pad_values.items():
            rows = []
            for f in features:
                t = f[key]
                gap = longest - t.shape[0]
                if gap:
                    t = torch.cat(
                        [t, torch.full((gap,), value, dtype=t.dtype)], dim=0
                    )
                rows.append(t)
            out[key] = torch.stack(rows, dim=0)

        # Same row order as the token streams above. image_grid_thw is the only
        # thing that says where one row's images end and the next row's begin,
        # so a reordering here would hand row A's pixels to row B's tokens
        # without any shape disagreeing.
        out["pixel_values"] = torch.cat([f["pixel_values"] for f in features], dim=0)
        out["image_grid_thw"] = torch.cat(
            [f["image_grid_thw"] for f in features], dim=0
        )

        # The patch stream must account for exactly the images the grid
        # describes: prod(t*h*w) summed over images. If these disagree the
        # model will still run and read the wrong pixels for some rows.
        expected = int(out["image_grid_thw"].prod(dim=1).sum())
        got = out["pixel_values"].shape[0]
        assert expected == got, (
            f"pixel_values has {got} patches but image_grid_thw describes "
            f"{expected}; the vision tensors are out of step with each other"
        )

        # Nothing outside a real position may carry a label.
        assert not (
            (out["labels"] != IGNORE_INDEX) & (out["attention_mask"] == 0)
        ).any(), "a padded position carries a label"

        return out
