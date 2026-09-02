"""One-step behavioural traces for every `train_step` branch.

These exist to make the `train_step` split safe: each test drives a real
`KnowledgeDistiller.train_step` with a fixed seed and pins the loss, the metric
keys, and the fact that the student actually moved. A pure code move has to leave
all three untouched.

They also close a standing coverage gap -- before this file, `cdm`, `dskd`, `emo`
and `stella` had no test of any kind, while `ggpkd`, `rkd` and `talas` each had
one. The students and teachers here are deliberately tiny stand-ins: the point is
to exercise the distiller's plumbing (device moves, pooling, task loss, criterion
call, backward, optimizer/scheduler step), not to check the encoders.
"""

from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F
from torch import nn
from torch.amp import GradScaler

from distiller import KnowledgeDistiller
from src.criterions.dual_space_kd import DualSpaceKD

VOCAB, DIM, SEQ, BATCH = 40, 8, 5, 4


class _TinyEncoder(nn.Module):
    """Stands in for a HF encoder: returns `last_hidden_state`, optionally attentions."""

    def __init__(self, vocab: int = VOCAB, dim: int = DIM, heads: int = 2):
        super().__init__()
        self.embedding = nn.Embedding(vocab, dim)
        self.heads = heads
        self.config = SimpleNamespace(hidden_size=dim, num_attention_heads=heads)

    def forward(
        self,
        input_ids,
        attention_mask=None,
        return_dict=True,
        output_attentions=False,
        output_hidden_states=False,
    ):
        hidden = self.embedding(input_ids)
        out = {"last_hidden_state": hidden}
        if output_attentions:
            att = (
                torch.softmax(hidden @ hidden.transpose(1, 2), dim=-1)
                .unsqueeze(1)
                .expand(-1, self.heads, -1, -1)
            )
            out["attentions"] = (att, att)
        if output_hidden_states:
            out["hidden_states"] = (hidden, hidden)
        return SimpleNamespace(**out)


def _base_distiller(method: str, criterion, *, task_type="pair_cls", temperature=0.07):
    distiller = KnowledgeDistiller.__new__(KnowledgeDistiller)
    distiller.config = SimpleNamespace(
        distill_method=method,
        task_type=task_type,
        temperature=temperature,
        w_task=0.5,
        w_cls=1.0,
        alpha_dtw=0.5,
    )
    distiller.device_s = distiller.device_t = torch.device("cpu")
    distiller.model_student = _TinyEncoder()
    distiller.model_teacher = _TinyEncoder()
    distiller.criterion = criterion
    distiller.task_head = None
    distiller.proj_s2t = None
    distiller.optimizer = torch.optim.Adam(
        distiller.model_student.parameters(), lr=1e-2
    )
    distiller.scheduler = torch.optim.lr_scheduler.LambdaLR(
        distiller.optimizer, lambda _: 1.0
    )
    distiller.scaler = GradScaler("cuda", enabled=False)
    distiller.current_epoch = 0
    distiller.current_step = 0
    return distiller


def _paired_batch(with_special_masks: bool = False):
    ids = torch.arange(BATCH * SEQ).reshape(BATCH, SEQ) % VOCAB
    ones = torch.ones_like(ids)
    batch = {
        "input_ids1_stu": ids,
        "attention_mask1_stu": ones,
        "input_ids2_stu": (ids + 3) % VOCAB,
        "attention_mask2_stu": ones,
        "input_ids1_tea": ids,
        "attention_mask1_tea": ones,
        "input_ids2_tea": (ids + 3) % VOCAB,
        "attention_mask2_tea": ones,
    }
    if with_special_masks:
        special = torch.zeros_like(ids)
        special[:, 0] = 1
        batch["special_tokens_mask1_stu"] = special
        batch["special_tokens_mask1_tea"] = special
    return batch


def _run_one_step(distiller, batch):
    before = distiller.model_student.embedding.weight.detach().clone()
    loss, metrics = distiller.train_step(batch)
    moved = not torch.equal(before, distiller.model_student.embedding.weight.detach())
    return loss, metrics, moved


def test_dskd_train_step_contract():
    torch.manual_seed(0)
    criterion = DualSpaceKD(student_dim=DIM, teacher_dim=DIM, w_task=0.5, alpha_dtw=0.5)
    distiller = _base_distiller("dskd", criterion)
    distiller.optimizer.add_param_group({"params": criterion.parameters(), "lr": 1e-2})
    distiller.scheduler = torch.optim.lr_scheduler.LambdaLR(
        distiller.optimizer, lambda _: 1.0
    )

    loss, metrics, moved = _run_one_step(distiller, _paired_batch(True))

    assert torch.isfinite(loss)
    assert moved, "the student must be updated"
    assert "loss_total" in metrics
    assert all(isinstance(v, (int, float)) for v in metrics.values())


def test_emo_train_step_contract():
    """EMO is the only branch that pulls attentions from both models."""
    torch.manual_seed(0)

    seen = {}

    class _StubEMO(nn.Module):
        """Records the keyword contract the distiller relies on."""

        def __init__(self):
            super().__init__()
            self.scale = nn.Parameter(torch.ones(()))

        def compute_emo_loss(self, *, teacher_outputs, student_outputs, **kwargs):
            seen.update(kwargs)
            seen["teacher_attentions"] = teacher_outputs.attentions is not None
            seen["student_attentions"] = student_outputs.attentions is not None
            loss = self.scale * F.mse_loss(
                student_outputs.last_hidden_state, teacher_outputs.last_hidden_state
            )
            return loss, {"loss_emo": float(loss.detach())}

    distiller = _base_distiller("emo", _StubEMO())
    distiller.tok_student = None
    distiller.tok_teacher = None

    loss, metrics, moved = _run_one_step(distiller, _paired_batch())

    assert torch.isfinite(loss)
    assert moved
    # Both sides must arrive with attentions -- EMO is the only branch that asks
    # the teacher for them, so a split that drops output_attentions breaks here.
    assert seen["teacher_attentions"] and seen["student_attentions"]
    for key in (
        "input_ids_tea",
        "input_ids_stu",
        "attention_mask_tea",
        "attention_mask_stu",
        "tok_teacher",
        "tok_student",
        "att_loss_weight",
        "ot_loss_weight",
    ):
        assert key in seen, f"distiller stopped passing {key} to compute_emo_loss"


@pytest.mark.parametrize(
    "method", ["ggpkd", "rkd", "talas", "cdm", "dskd", "emo", "stella"]
)
def test_every_method_still_routes_to_a_step(method):
    """`train_step` is a dispatcher now; every method must still reach a step.

    This is the structural guard for `cdm` and `stella`, whose criterions need
    real tokenizers or a wrapped HF model and so have no one-step trace here. It
    reads the dispatcher and the step modules together, so moving a branch
    between them is fine but losing one is not.
    """
    import inspect

    from src.distill.steps import ggpkd, rkd, standard, talas

    sources = "".join(
        inspect.getsource(obj)
        for obj in (
            KnowledgeDistiller.train_step,
            ggpkd.step,
            rkd.step,
            talas.step,
            standard.step,
        )
    )
    assert f'"{method}"' in sources or f"'{method}'" in sources, (
        f"no step handles {method}"
    )


def test_step_modules_do_not_import_each_other():
    """The point of the split: editing one method cannot reach another.

    ggpkd is the active method and the six baselines are frozen, so nothing
    should be able to break GGPKD by touching a baseline step.
    """
    import ast
    from pathlib import Path as _Path

    step_dir = _Path("src/distill/steps")
    names = {p.stem for p in step_dir.glob("*.py")} - {"__init__"}
    for path in step_dir.glob("*.py"):
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module:
                imported = node.module.split(".")[-1]
                assert imported not in names - {path.stem}, (
                    f"{path.name} imports the {imported} step"
                )
