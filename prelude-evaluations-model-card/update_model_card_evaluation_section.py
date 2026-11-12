# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "huggingface_hub",
# ]
# ///
"""Replace the Evaluation section of the HF model card with
model_card_evaluation.md.

The section is the "Evaluation" heading of the model card's README.md and
everything up to the next heading of the same or a higher level (headings
inside code blocks are ignored). Local images referenced in the markdown
(e.g. figures/plot_results_german_all.png) are uploaded to the model repo at
the same path and linked through their resolve/main URL.

The diff of the section is always shown; nothing is pushed unless
--commit-change is given, in which case README.md and the images are uploaded
in a single commit (needs a token with write access, e.g. `hf auth login`).

Usage:
  uv run update_model_card_evaluation_section.py                  # dry run
  uv run update_model_card_evaluation_section.py --commit-change
"""
import argparse
import difflib
import os
import re
import sys

from huggingface_hub import CommitOperationAdd, HfApi, hf_hub_download

HERE = os.path.dirname(os.path.abspath(__file__))
SECTION_MD = os.path.join(HERE, "model_card_evaluation.md")
REPO_ID = "openeurollm/prelude-base"
HEADING = "Evaluation"

HEADING_RE = re.compile(r"^(#{1,6})\s+(.*?)\s*#*\s*$")
IMAGE_RE = re.compile(r"(!\[[^\]]*\]\()([^)\s]+)(\))")


def find_section(lines: list[str], title: str) -> tuple[int, int]:
    """[start, end) line range of the section whose heading is `title`."""
    start = level = None
    fence = False
    for i, line in enumerate(lines):
        if line.lstrip().startswith(("```", "~~~")):
            fence = not fence
        if fence or not (m := HEADING_RE.match(line)):
            continue
        if start is None:
            if m[2].strip().lower() == title.lower():
                start, level = i, len(m[1])
        elif len(m[1]) <= level:
            return start, i
    if start is None:
        sys.exit(f"no '{title}' heading found in the model card")
    return start, len(lines)


def resolve_images(text: str, repo_id: str) -> tuple[str, list[str]]:
    """Point local image paths at the repo; return (text, local paths)."""
    images = []

    def repl(m):
        path = m[2]
        if re.match(r"^[a-z]+://", path):
            return m[0]
        if not os.path.exists(os.path.join(HERE, path)):
            sys.exit(f"image {path} referenced in {SECTION_MD} does not exist")
        images.append(path)
        url = f"https://huggingface.co/{repo_id}/resolve/main/{path}"
        return f"{m[1]}{url}{m[3]}"

    return IMAGE_RE.sub(repl, text), images


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--commit-change", action="store_true",
                        help="push the updated model card and images")
    parser.add_argument("--repo-id", default=REPO_ID)
    args = parser.parse_args()

    card_path = hf_hub_download(args.repo_id, "README.md")
    card = open(card_path).read()
    lines = card.splitlines(keepends=True)
    start, end = find_section(lines, HEADING)

    new_section, images = resolve_images(open(SECTION_MD).read(), args.repo_id)
    new_section = new_section.rstrip("\n") + "\n"
    if end < len(lines):
        new_section += "\n"            # blank line before the next heading
    old_section = "".join(lines[start:end])
    new_card = "".join(lines[:start]) + new_section + "".join(lines[end:])

    diff = list(difflib.unified_diff(
        old_section.splitlines(keepends=True),
        new_section.splitlines(keepends=True),
        fromfile=f"{args.repo_id}/README.md ({HEADING}, current)",
        tofile=f"{args.repo_id}/README.md ({HEADING}, new)"))
    print("".join(diff) if diff else "Evaluation section unchanged.")
    print(f"\nimages to upload: {', '.join(images) if images else 'none'}")

    if not args.commit_change:
        print("\ndry run; pass --commit-change to push")
        return
    ops = [CommitOperationAdd("README.md", new_card.encode())]
    ops += [CommitOperationAdd(p, os.path.join(HERE, p)) for p in images]
    info = HfApi().create_commit(
        args.repo_id, ops, commit_message="Update Evaluation section")
    print(f"\ncommitted: {info.commit_url}")


if __name__ == "__main__":
    main()
