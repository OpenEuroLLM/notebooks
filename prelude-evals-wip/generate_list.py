# /// script
# requires-python = ">=3.9"
# dependencies = [
#   "requests",
# ]
# ///
"""Generate models-to-evaluate.txt: checkpoints spaced ~200B tokens apart
for prelude, apertus, olmo3 and datamix.

Token counts for apertus/olmo3/datamix are approximate:
- apertus: token count is read directly from the branch name (stepX-tokensYB/T).
- olmo3: only stage1 (pretraining) branches are used, since stage1 alone covers
  5.93T of the ~6T total training budget (stage2/stage3 are short follow-up
  stages with no token count in the branch name). Tokens are estimated by
  linearly interpolating step -> tokens using the stage1 total (5.93T tokens
  at the final stage1 step).
- datamix: tokens are estimated by linearly interpolating step -> tokens using
  the model's total training budget (4T tokens at the final step, 953675, per
  the model card).
"""
import re

import requests

TARGET_SPACING_B = 200  # billions of tokens between selected checkpoints

PRELUDE_STEPS = [
    2400, 4800, 7200, 9600, 12000,
    24000, 48000, 72000, 96000, 120000,
    144000, 168000, 192000, 216000, 240000,
    264000, 288000, 312000, 336000, 360000,
    384000, 408000, 432000, 456000, 480000,
    504000, 528000, 552000, 576000, 600000,
    624000,
    648000,
]


def get_branches(repo_id):
    resp = requests.get(f"https://huggingface.co/api/models/{repo_id}/refs")
    resp.raise_for_status()
    return [b["name"] for b in resp.json().get("branches", [])]


def tokens_to_billions(value, unit):
    value = float(value)
    return value * 1000 if unit == "T" else value


def select_spaced(items, spacing_b=TARGET_SPACING_B):
    """items: list of (name, tokens_billions). Greedily keep the first
    checkpoint at/after each spacing_b-wide target, plus always the last one."""
    items = sorted(items, key=lambda x: x[1])
    selected = []
    next_target = 0
    for name, tokens in items:
        if tokens >= next_target:
            selected.append((name, tokens))
            next_target = tokens + spacing_b
    if selected[-1] != items[-1]:
        selected.append(items[-1])
    return selected


def prelude_checkpoints():
    return [f"openeurollm/prelude-checkpoints/iter_{s:07d}" for s in PRELUDE_STEPS]


def apertus_checkpoints():
    repo_id = "swiss-ai/Apertus-8B-2509"
    pattern = re.compile(r"^step\d+-tokens(\d+(?:\.\d+)?)([BT])$")
    items = []
    for branch in get_branches(repo_id):
        m = pattern.match(branch)
        if m:
            items.append((branch, tokens_to_billions(m.group(1), m.group(2))))
    selected = select_spaced(items)
    return [f"{repo_id}/{name}" for name, _ in selected]


def olmo3_checkpoints():
    repo_id = "allenai/Olmo-3-1025-7B"
    total_tokens_b = 5930  # 5.93T tokens over stage1, per the model card
    pattern = re.compile(r"^stage1-step(\d+)$")
    steps = []
    for branch in get_branches(repo_id):
        m = pattern.match(branch)
        if m:
            steps.append((branch, int(m.group(1))))
    max_step = max(step for _, step in steps)
    items = [(name, total_tokens_b * step / max_step) for name, step in steps]
    selected = select_spaced(items)
    return [f"{repo_id}/{name}" for name, _ in selected]


def datamix_checkpoints():
    repo_id = "openeurollm/datamix-9b-80-20"
    total_tokens_b = 4000  # 4T tokens total, per the model card
    final_step = 953675  # final step (main branch), per the model card
    pattern = re.compile(r"^iter_(\d+)$")
    steps = []
    for branch in get_branches(repo_id):
        m = pattern.match(branch)
        if m:
            steps.append((branch, int(m.group(1))))
    items = [(name, total_tokens_b * step / final_step) for name, step in steps]
    selected = select_spaced(items)
    return [f"{repo_id}/{name}" for name, _ in selected]


EXTRA_MODELS = [
    "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16",
    "Qwen/Qwen3-8B",
    "meta-llama/Llama-3.1-8B",
]


def main():
    groups = {
        "prelude": prelude_checkpoints(),
        "apertus": apertus_checkpoints(),
        "olmo3": olmo3_checkpoints(),
        "datamix": datamix_checkpoints(),
        "extra": list(EXTRA_MODELS),
    }

    lines = []
    for name, checkpoints in groups.items():
        print(f"{name}: {len(checkpoints)} checkpoints")
        lines.extend(checkpoints)

    with open("models-to-evaluate.txt", "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\nWrote {len(lines)} total model paths to models-to-evaluate.txt")


if __name__ == "__main__":
    main()
