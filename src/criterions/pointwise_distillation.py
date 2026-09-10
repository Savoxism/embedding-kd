"""Pointwise embedding distillation: the measurement floor of the batch study.

    L = 1 - cos( W s_i , t_i )

One term, one example at a time. No pair, no batch, no support -- which is the
entire point of it. In the batch-intervention study this arm establishes how much
a benchmark moves when the *only* thing that changed is which texts travelled
together: whatever spread it shows across the three batch samplers is seed noise
and dataloader ordering, because its objective provably cannot read batch
membership. An in-batch relational arm that moves by several times that spread is
then moving for a reason, and the comparison does not rest on a prior belief about
how large seed noise is.

Read it as a floor, not as a competitor. It is expected to be flat, and a flat
result here is what makes the other arms interpretable.

`W` is a single linear map, needed because teacher and student dimensions differ
(1024/2560/4096 against 384/768) and a cosine is not defined across them. It is
trainable, and that is the standard formulation of pointwise embedding KD -- but
it is also why this arm cannot pin absolute similarity levels: any invertible
transform of the student space is absorbed by `W`. That limitation is the reason
the method distils relations instead, and it is stated here rather than left for a
reader to discover from the numbers.
"""

import torch
import torch.nn.functional as F
from torch import nn


class PointwiseDistillation(nn.Module):
    """Cosine alignment between the projected student vector and the teacher's.

    The forward signature matches `RelationalKnowledgeDistillation` so both share
    `src.distill.steps.rkd` unchanged: one student vector, one teacher vector, one
    optional task loss.
    """

    def __init__(
        self,
        *,
        student_dim: int,
        teacher_dim: int,
        task_weight: float = 0.0,
        eps: float = 1e-12,
    ):
        super().__init__()
        if task_weight < 0:
            raise ValueError("task_weight must be non-negative")
        self.task_weight = float(task_weight)
        self.eps = float(eps)
        # No bias: the loss is a cosine, so a constant shift of the projected
        # vector changes the objective without expressing anything about the
        # teacher, and it is the one parameter that could absorb a systematic
        # error instead of fixing it.
        self.projection = (
            nn.Identity()
            if student_dim == teacher_dim
            else nn.Linear(student_dim, teacher_dim, bias=False)
        )

    def forward(
        self,
        student: torch.Tensor,
        teacher: torch.Tensor,
        task_loss: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        projected = self.projection(student.float())
        target = teacher.float()
        cosine = F.cosine_similarity(projected, target, dim=-1, eps=self.eps)
        pointwise = (1.0 - cosine).mean()

        zero = pointwise * 0.0
        if self.task_weight > 0:
            if task_loss is None:
                raise ValueError("task_weight > 0 requires task_loss")
            task = task_loss.float()
        else:
            task = zero

        total = pointwise + self.task_weight * task
        return total, {
            "loss_total": float(total.detach()),
            "loss_pointwise": float(pointwise.detach()),
            "teacher_cos_mean": float(cosine.mean().detach()),
            "loss_task": float(task.detach()),
        }
