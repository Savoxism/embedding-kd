"""Print the method's candidate width, so a baseline can be matched to it.

The diffusion quota is derived from the graph artifact at startup rather than
configured, so the total candidate width -- quota + hard + uniform -- is not a
number anyone can write down in advance. The budget-matched baseline in S1 needs
exactly that number, and hard-coding it would silently unmatch the comparison the
first time the corpus or the coverage target moves.

    python scripts/ablation/candidate_width.py cache/.../graph_base.pt 40 26
"""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.ggpkd.policy import derive_diffusion_quota  # noqa: E402


def main() -> int:
    if len(sys.argv) != 4:
        print(__doc__.strip(), file=sys.stderr)
        return 2
    artifact_path, hard_neg_k, random_neg_k = sys.argv[1], sys.argv[2], sys.argv[3]
    artifact = torch.load(artifact_path, map_location="cpu", weights_only=False)
    quota = derive_diffusion_quota(
        artifact["pool_probs"].numpy(), artifact["metadata"]["diffusion_scales"]
    )
    print(quota + int(hard_neg_k) + int(random_neg_k))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
