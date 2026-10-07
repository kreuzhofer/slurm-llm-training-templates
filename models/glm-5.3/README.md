# GLM-5.3 inference on Slurm

Serve [GLM-5.3](https://huggingface.co/zai-org/GLM-5.3) with vLLM's
OpenAI-compatible API on one node (8 GPUs) or two nodes (16 GPUs), then reach
it from your laptop through an SSH tunnel to the login node.

| file | what it does |
|---|---|
| `download.sbatch` | Slurm job that downloads the weights to shared storage (CPU only) |
| `download.sh` | the download itself; also runs directly on a login node |
| `serve.sbatch` | Slurm job that starts the vLLM server on 1 or more nodes |
| `query.sh` | sends one chat request to the running server |

Two weight variants are available:

| variant | Hugging Face repo | size | stored in | served as |
|---|---|---|---|---|
| `fp8` (default) | `zai-org/GLM-5.3` | 756 GB | `$TEMPLATES_DIR/models/GLM-5.3` | `glm-5.3-fp8` |
| `nvfp4` | `nvidia/GLM-5.3-NVFP4` | 464 GB | `$TEMPLATES_DIR/models/GLM-5.3-NVFP4` | `glm-5.3-nvfp4` |

NVFP4 needs Blackwell GPUs (B200, B300, GB200, GB300).

Requirements:

- Nodes with 8 × NVIDIA B300 each, one node or two.
- A filesystem shared by the login node and all compute nodes, here
  `/mnt/data`, with room for the variants you download.
- Internet access from the compute nodes. They download the weights from
  Hugging Face, and vLLM downloads GPU kernels from NVIDIA on first start.

`$TEMPLATES_DIR` is `/mnt/data/slurm-llm-templates`. All commands below run on
the login node unless stated otherwise.

## 0. One-time setup

```bash
git clone https://github.com/kreuzhofer/slurm-llm-training-templates.git
cd slurm-llm-training-templates
bash cluster/setup.sh                       # venv (vLLM 0.28.0) + copy of the repo on shared storage
source /mnt/data/slurm-llm-templates/activate.sh
cd $TEMPLATES_DIR/repo/models/glm-5.3
```

## 1. Download the weights

Run once per variant. The job runs on one node without GPUs and resumes if
interrupted.

```bash
sbatch download.sbatch fp8        # or: nvfp4, both
squeue --me
tail -f $TEMPLATES_DIR/logs/glm53_download_<JOBID>.out
```

The job finishes with `Done.` in its log.

## 2. Start the server

```bash
sbatch serve.sbatch fp8          # or: nvfp4
```

The server uses one node with 8 GPUs. To run on two nodes, see
[Using two nodes](#using-two-nodes).

To queue download and server together, let the server wait for the download:

```bash
DL=$(sbatch --parsable download.sbatch fp8)
sbatch --dependency=afterok:$DL serve.sbatch fp8
```

## 3. Wait until it is ready

```bash
squeue --me -n glm53-serve -O JobID,State,BatchHost
grep -E 'Endpoint|Application startup complete' $TEMPLATES_DIR/logs/glm53_serve_<JOBID>.out
```

The head node is the job's `EXEC_HOST`. The `Endpoint` line also names it,
e.g. `http://worker-b300-1:8000/v1`. On two nodes this is not necessarily
the lower-numbered node. The server is ready once
`Application startup complete` appears.

Startup takes about 15 to 30 minutes, most of it reading the weights from
shared storage. The first start on a new cluster takes up to 20 minutes
longer while vLLM builds and downloads GPU kernels. They are cached under
`~/.cache` and reused by later starts.

## 4. Send a request from the login node

```bash
bash query.sh "In one sentence: what is a mixture-of-experts model?"
REASONING_EFFORT=low bash query.sh "Name the capital of France."
```

With more than one server running, pick one with `SERVE_JOBID=<JOBID>`.

Plain `curl` works the same way:

```bash
curl http://<head-node>:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model": "glm-5.3-fp8",
       "messages": [{"role": "user", "content": "Hello!"}],
       "reasoning_effort": "low"}'
```

## 5. Reach the server from your laptop

The compute nodes are not reachable from outside the cluster, so forward a
local port through the login node. On your laptop:

```bash
# <head-node> is the EXEC_HOST from step 3, e.g. worker-b300-1
ssh -i ~/.ssh/<your-private-key> -N -L 8000:<head-node>:8000 <user>@<login-node-address>
```

The login node accepts SSH key authentication only. Pass the private key that
belongs to your cluster account with `-i`. You can leave `-i` out if that key
is already your default or is configured for this host in `~/.ssh/config`.

Leave that running. In a second terminal on your laptop:

```bash
curl http://localhost:8000/v1/models
```

Any OpenAI-compatible client can now use `http://localhost:8000/v1`:

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="unused")
resp = client.chat.completions.create(
    model="glm-5.3-fp8",
    messages=[{"role": "user", "content": "Write a haiku about GPUs."}],
    reasoning_effort="low",
)
print(resp.choices[0].message.content)
```

If port 8000 is already taken on your laptop, use another local port, e.g.
`-L 18000:<head-node>:8000` and `http://localhost:18000/v1`. With two
separate servers, forward one local port to each, e.g.
`-L 8000:<first-node>:8000 -L 8001:<second-node>:8000`.

## 6. Stop the server

```bash
scancel <JOBID>
```

Server jobs stop on their own after 4 hours. Ask for more with
`sbatch --time=12:00:00 serve.sbatch ...`.

## Using two nodes

`--nodes=2` runs one server across two nodes. The model is split into two
pipeline stages, one per node, and the head node runs the HTTP endpoint.

```bash
sbatch --nodes=2 serve.sbatch fp8          # or: nvfp4
```

GLM-5.3 already fits on one node, so the second node adds KV cache, not room
for a bigger model. Each node holds only half of the weights, which leaves
more memory for the KV cache. Speculative decoding is off in this layout, so
a single request is slower than on one node.

| layout | KV cache, FP8 | KV cache, NVFP4 |
|---|---|---|
| one node | 3.2M tokens | 3.9M tokens |
| one server across two nodes | 8.7M tokens | 9.4M tokens |

Another option is two separate one-node servers behind a load balancer of
your choice: submit `sbatch serve.sbatch fp8` twice and point the load
balancer at both endpoints. Each server keeps speculative decoding, and a
failed node takes only one server down. Together they hold 2 × 3.2M (FP8) or
2 × 3.9M (NVFP4) KV cache tokens.

## Context length

`MAX_MODEL_LEN` caps a single request, prompt plus output. The default is
131072 tokens; the model supports up to 1048576. All running requests share
the KV cache, so a larger KV cache serves more requests at once, not longer
ones. One node already holds several full-length requests:

```bash
MAX_MODEL_LEN=1048576 sbatch serve.sbatch fp8
```

## Reasoning and tool calls

- GLM-5.3 always reasons before it answers. The reasoning comes back in
  `message.reasoning`, the answer in `message.content`.
- Control how much it reasons with `reasoning_effort`: `low`, `high`, or
  leave it out for the default (maximum).
- Do not send `chat_template_kwargs: {"enable_thinking": false}`. GLM-5.3 does
  not support it, and the reasoning then ends up inside `message.content`.
- Tool calling uses the standard OpenAI `tools` / `tool_calls` format.

## Options

Set these as environment variables in front of `sbatch`:

| variable | default | meaning |
|---|---|---|
| `PORT` | `8000` | HTTP port on the head node |
| `MAX_MODEL_LEN` | `131072` | maximum tokens per request, prompt plus output; up to 1048576 |
| `MTP_TOKENS` | `5` | tokens drafted per step by the built-in MTP head; `0` turns speculative decoding off. Always off for one server across two nodes |
| `SERVED_NAME` | `glm-5.3-<variant>` | model name clients send |
| `EXTRA_ARGS` | empty | extra `vllm serve` flags, passed through as-is |

Example:

```bash
MAX_MODEL_LEN=262144 PORT=8001 sbatch serve.sbatch nvfp4
```

## Logs

| job | log |
|---|---|
| download | `$TEMPLATES_DIR/logs/glm53_download_<JOBID>.out` |
| server | `$TEMPLATES_DIR/logs/glm53_serve_<JOBID>.out` (vLLM log), `.err` (progress bars, Slurm messages) |

## License

GLM-5.3 is released under the
[GLM-5.3 License](https://huggingface.co/zai-org/GLM-5.3/blob/main/LICENSE).
The NVFP4 checkpoint is additionally covered by the
[NVIDIA Open Model Agreement](https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-agreement/).
