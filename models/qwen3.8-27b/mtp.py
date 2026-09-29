"""
mtp.py -- a TRAINABLE multi-token-prediction head for Qwen3.8, ported from vLLM.

transformers 5.16.1 has no MTP implementation for this architecture at all. The
only occurrences of `mtp` in modeling_qwen3_5.py are two ignore-regexes, on the
SHARED base class, so neither model class instantiates the head and no adapter
can attach to it. That is why rc0 ships the base drafter copied verbatim.

Measured consequence of that transplant: the trunk's final hidden state -- the
exact tensor this head consumes -- moved to cosine 0.788 after the LoRA merge,
and acceptance length fell to 2.93 against 4.39 for a matched head. Fixing it at
the root needs a head that can be trained, which is this file.

WHAT IT IS

Structurally unremarkable once it can be built: a projection, one ordinary
full-attention decoder layer, and three RMSNorms. ~424M parameters, 1.6% of the
model. Ported from vLLM 0.28.0's inference-only `Qwen3_5MultiTokenPredictor`
(vllm/model_executor/models/qwen3_5_mtp.py), which is the reference for both the
module shape and the weight names.

    embed        = embed_tokens(next_ids)        SHARED with the trunk
    x            = fc(cat[pre_fc_norm_embedding(embed),
                          pre_fc_norm_hidden(trunk_hidden)])
    x            = layers[0](x)                  one full-attention layer
    x            = norm(x)
    logits       = lm_head(x)                    SHARED with the trunk

Two things a naive port gets wrong, both taken from the vLLM source rather than
guessed:

  * `mtp.layers.0` is a FULL-ATTENTION layer, while 48 of the trunk's 64 layers
    are Gated DeltaNet. transformers builds the block type from
    `config.layer_types[layer_idx]`, so the config handed to it must report
    full_attention at the MTP index or the wrong layer type is constructed.
  * `embed_tokens` and `lm_head` are SHARED, not owned. The checkpoint's 15
    `mtp.*` tensors contain neither, which is the corroborating evidence, and
    the config says `mtp_use_dedicated_embeddings: false`.

THE OBJECTIVE, AND WHY THE ALIGNMENT IS THE WHOLE GAME

The trunk predicts x_{t+1} from h_t. This head predicts **x_{t+2}** from h_t
paired with the embedding of x_{t+1}. Off-by-one here does not crash: it trains
to a plausible-looking loss on the wrong target and produces a drafter that
proposes confidently wrong tokens, which shows up only as a low acceptance rate
much later.

So the alignment has a cheap, decisive test, and `verify_alignment()` runs it: a
PRETRAINED head on its OWN base trunk should already score far below ln(vocab)
~= 11.9. If the shift is wrong it cannot, because it is being asked to predict a
token it was never trained to predict. Loss near chance means the shift is wrong,
not that the head is bad.

Everything is kept at full sequence length rather than sliced, so the rotary
embeddings and attention mask captured from the trunk stay valid without any
parallel slicing -- the shift lives entirely in which target each position is
scored against.
"""

import torch
from torch import nn
from transformers.models.qwen3_5.modeling_qwen3_5 import (
    Qwen3_5DecoderLayer,
    Qwen3_5RMSNorm,
)

IGNORE_INDEX = -100

# How far ahead this head predicts. The trunk does +1; MTP does +2. Named rather
# than written as a literal 2 in three places, because every one of them has to
# agree and a silent disagreement is the failure described above.
PREDICT_OFFSET = 2


def _mtp_layer_config(text_config):
    """
    A copy of the text config that reports full_attention at the MTP index.

    `Qwen3_5DecoderLayer.__init__` reads `config.layer_types[layer_idx]` to pick
    its block type, and the real config's layer_types has exactly
    num_hidden_layers entries -- so the MTP index is off the end. vLLM sidesteps
    this by passing layer_type explicitly; transformers has no such parameter,
    so the config is extended instead.
    """
    import copy

    cfg = copy.deepcopy(text_config)
    cfg.layer_types = list(cfg.layer_types) + ["full_attention"]
    return cfg


class Qwen3_5MTPHead(nn.Module):
    """The head itself. Owns no embedding and no lm_head -- both are the trunk's."""

    def __init__(self, text_config):
        super().__init__()
        cfg = _mtp_layer_config(text_config)
        self.layer_idx = text_config.num_hidden_layers
        hidden = text_config.hidden_size
        eps = text_config.rms_norm_eps

        self.fc = nn.Linear(hidden * 2, hidden, bias=False)
        self.pre_fc_norm_embedding = Qwen3_5RMSNorm(hidden, eps=eps)
        self.pre_fc_norm_hidden = Qwen3_5RMSNorm(hidden, eps=eps)
        self.layers = nn.ModuleList([Qwen3_5DecoderLayer(cfg, self.layer_idx)])
        self.norm = Qwen3_5RMSNorm(hidden, eps=eps)

        assert self.layers[0].block_type == "full_attention", (
            f"MTP layer built as {self.layers[0].block_type!r}; it must be "
            "full_attention. layer_types was not extended correctly."
        )

    def forward(self, trunk_hidden, next_embeds, position_embeddings,
                attention_mask=None, position_ids=None):
        x = torch.cat(
            [self.pre_fc_norm_embedding(next_embeds),
             self.pre_fc_norm_hidden(trunk_hidden)],
            dim=-1,
        )
        x = self.fc(x)
        x = self.layers[0](
            x,
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
            position_ids=position_ids,
        )
        return self.norm(x)


def load_base_mtp_weights(head, base_model_path, strict=True):
    """
    Load the checkpoint's 15 `mtp.*` tensors into `head`. Returns what was loaded.

    Strict by design. A partially initialised drafter trains to a plausible loss
    and drafts badly, which is indistinguishable from "the idea did not work"
    until someone measures acceptance -- so a missing tensor stops here.
    """
    import glob
    import json
    import os

    from safetensors import safe_open

    index_path = os.path.join(base_model_path, "model.safetensors.index.json")
    if os.path.exists(index_path):
        with open(index_path) as handle:
            weight_map = json.load(handle)["weight_map"]
        shards = {}
        for name, shard in weight_map.items():
            if name.startswith("mtp."):
                shards.setdefault(shard, []).append(name)
    else:
        shards = {os.path.basename(p): None
                  for p in glob.glob(os.path.join(base_model_path, "*.safetensors"))}

    tensors = {}
    for shard, names in shards.items():
        with safe_open(os.path.join(base_model_path, shard), framework="pt") as f:
            for name in (names if names is not None else f.keys()):
                if name.startswith("mtp."):
                    tensors[name[len("mtp."):]] = f.get_tensor(name)

    if not tensors:
        raise ValueError(
            f"no mtp.* tensors in {base_model_path}. This checkpoint has no MTP "
            "head to initialise from -- a merged artifact written by a model "
            "class that drops ^mtp.* will look exactly like this."
        )

    missing, unexpected = head.load_state_dict(tensors, strict=False)
    if strict and (missing or unexpected):
        raise ValueError(
            f"MTP weights do not match the module.\n"
            f"  missing from checkpoint: {sorted(missing)}\n"
            f"  unexpected in checkpoint: {sorted(unexpected)}"
        )
    return sorted(tensors)


class MTPTrainer(nn.Module):
    """
    Trunk + head, wired for the +2 objective. The trunk is frozen by default.

    Freezing the trunk is dgx-manager's suggestion and the right first step: it
    isolates the one variable that matters -- whether an aligned head recovers
    the acceptance rate -- and it is a strict subset of full co-training, so
    nothing here has to be rewritten to go further.

    The trunk's own rotary embeddings and causal mask are captured by hook rather
    than recomputed. This architecture uses mrope, whose position ids are not a
    plain arange once images are in the sequence, and reimplementing that to feed
    one extra layer would be a second thing that can silently disagree with the
    first. The same hook trick verified the drift probe's tensor.
    """

    def __init__(self, model, head, train_trunk=False):
        super().__init__()
        self.model = model
        self.head = head
        self.train_trunk = train_trunk
        if not train_trunk:
            for p in self.model.parameters():
                p.requires_grad = False

        text = model.model.language_model
        self._captured = {}
        self._hooks = [
            text.rotary_emb.register_forward_hook(
                lambda _m, _i, out: self._captured.__setitem__("pos_emb", out)
            )
        ]
        # A full-attention layer receives the mask this head needs. Hooking one
        # is cheaper and safer than rebuilding create_causal_mask's arguments.
        full_idx = next(
            i for i, t in enumerate(model.config.text_config.layer_types)
            if t == "full_attention"
        )
        self._hooks.append(
            text.layers[full_idx].register_forward_pre_hook(
                lambda _m, _a, kwargs: self._captured.update(
                    attention_mask=kwargs.get("attention_mask"),
                    position_ids=kwargs.get("position_ids"),
                ) or None,
                with_kwargs=True,
            )
        )

    def embed(self, input_ids):
        return self.model.model.language_model.embed_tokens(input_ids)

    def forward(self, input_ids=None, labels=None, **batch):
        self._captured.clear()
        trunk = self.model.model(
            input_ids=input_ids,
            **{k: v for k, v in batch.items() if k != "labels"},
        )
        hidden = trunk.last_hidden_state
        if not self.train_trunk:
            hidden = hidden.detach()

        for key in ("pos_emb", "attention_mask"):
            if key not in self._captured:
                raise RuntimeError(
                    f"did not capture {key} from the trunk; the model layout "
                    "changed and the head would run on recomputed values that "
                    "may not match the trunk's"
                )

        # At position t the head sees h_t and emb(x_{t+1}), and is scored against
        # x_{t+2}. Everything stays at full length; only the targets shift, so
        # the captured rotary and mask remain valid.
        pad_embed = torch.zeros_like(self.embed(input_ids[:, :1]))
        next_embeds = torch.cat([self.embed(input_ids[:, 1:]), pad_embed], dim=1)

        out = self.head(
            hidden,
            next_embeds,
            position_embeddings=self._captured["pos_emb"],
            attention_mask=self._captured["attention_mask"],
            position_ids=self._captured.get("position_ids"),
        )
        logits = self.model.lm_head(out)

        loss = None
        if labels is not None:
            tail = labels.new_full((labels.shape[0], PREDICT_OFFSET), IGNORE_INDEX)
            target = torch.cat([labels[:, PREDICT_OFFSET:], tail], dim=1)
            loss = nn.functional.cross_entropy(
                logits.reshape(-1, logits.shape[-1]).float(),
                target.reshape(-1),
                ignore_index=IGNORE_INDEX,
            )
        return {"loss": loss, "logits": logits}

    def close(self):
        for h in self._hooks:
            h.remove()
        self._hooks = []


@torch.no_grad()
def verify_alignment(trainer, batch, vocab_size):
    """
    Prove the +2 shift is right before training anything on it.

    A pretrained head on its own base trunk already knows this task, so its loss
    must sit far below chance (ln(vocab) ~= 11.9). If the shift were wrong the
    head would be asked for a token it was never trained to produce and would
    score near chance -- which is the only cheap way to tell a misaligned
    objective from a merely untrained one, since both train to something
    plausible-looking.

    Also scores the WRONG shifts for contrast. The correct one should win
    clearly; if it does not, do not train on this.
    """
    import math

    results = {}
    labels = batch["labels"]
    chance = math.log(vocab_size)
    for offset in (1, PREDICT_OFFSET, 3):
        tail = labels.new_full((labels.shape[0], offset), IGNORE_INDEX)
        target = torch.cat([labels[:, offset:], tail], dim=1)
        out = trainer(**{**batch, "labels": None})
        logits = out["logits"]
        loss = nn.functional.cross_entropy(
            logits.reshape(-1, logits.shape[-1]).float(),
            target.reshape(-1),
            ignore_index=IGNORE_INDEX,
        )
        results[offset] = float(loss)
    results["chance"] = chance
    return results
