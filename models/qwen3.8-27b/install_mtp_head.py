"""
install_mtp_head.py -- put a trained MTP head into a merged checkpoint, in place.

    python install_mtp_head.py <merged_dir> <mtp_head.safetensors>

merge_judge_lora.py writes the 15 mtp.* tensors into a shard of their own (the
last one, 849 MB), because the model class dropped them and they are restored
after the save. That layout makes installing a trained head a one-shard rewrite
rather than a second 51 GiB merge: overwrite that shard with the same tensor
names, leave the index alone, and verify.

Refuses rather than guesses at every step. The shard the index maps mtp.* to
must contain ONLY mtp.* tensors (otherwise rewriting it would drop something
else); the head file's names and shapes must match exactly (half a trained
drafter is worse than either); and after writing, every tensor is read back and
compared to the head file bit for bit.
"""

import json
import os
import sys


def main():
    if len(sys.argv) != 3:
        print(__doc__.strip().splitlines()[2]); return 2
    merged_dir, head_path = sys.argv[1], sys.argv[2]

    import torch
    from safetensors import safe_open
    from safetensors.torch import save_file

    index_path = os.path.join(merged_dir, "model.safetensors.index.json")
    with open(index_path) as handle:
        index = json.load(handle)
    wmap = index["weight_map"]
    mtp_names = sorted(n for n in wmap if n.startswith("mtp."))
    if not mtp_names:
        raise SystemExit(f"{merged_dir} carries no mtp.* tensors; nothing to replace")

    shards = {wmap[n] for n in mtp_names}
    if len(shards) != 1:
        raise SystemExit(f"mtp.* tensors span {len(shards)} shards {sorted(shards)}; "
                         "expected one -- was this merged by merge_judge_lora.py?")
    shard = shards.pop()
    others = sorted(n for n, s in wmap.items() if s == shard and not n.startswith("mtp."))
    if others:
        raise SystemExit(f"shard {shard} also holds {len(others)} non-mtp tensors, e.g. "
                         f"{others[:3]}; rewriting it would drop them. Refusing.")

    with safe_open(head_path, framework="pt") as f:
        head = {n: f.get_tensor(n) for n in f.keys()}
    if set(head) != set(mtp_names):
        raise SystemExit(
            f"{head_path} does not match the checkpoint's MTP tensors.\n"
            f"  in head, not needed: {sorted(set(head) - set(mtp_names))}\n"
            f"  needed, not in head: {sorted(set(mtp_names) - set(head))}")
    shard_path = os.path.join(merged_dir, shard)
    with safe_open(shard_path, framework="pt") as f:
        for n in mtp_names:
            cur = f.get_slice(n).get_shape()
            if tuple(cur) != tuple(head[n].shape):
                raise SystemExit(f"{n}: checkpoint {tuple(cur)} vs head {tuple(head[n].shape)}")

    print(f"installing {len(head)} tensors into {shard} ...")
    save_file({n: head[n].contiguous() for n in mtp_names}, shard_path,
              metadata={"format": "pt"})

    # The index's total_size changes only if the shard's byte size did.
    index.setdefault("metadata", {})["total_size"] = sum(
        os.path.getsize(os.path.join(merged_dir, f)) for f in set(wmap.values()))
    with open(index_path, "w") as handle:
        json.dump(index, handle, indent=1)

    with safe_open(shard_path, framework="pt") as f:
        bad = [n for n in mtp_names if not torch.equal(f.get_tensor(n), head[n])]
    if bad:
        raise SystemExit(f"read-back mismatch on {len(bad)} tensors: {bad[:3]}")
    print(f"verified: {len(mtp_names)}/{len(mtp_names)} tensors read back bit-identical "
          f"to {os.path.basename(head_path)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
