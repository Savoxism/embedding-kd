"""Helpers shared by the method specs."""


def attach_parameters(ctx, parameters, lr):
    """Give the optimizer a new param group and rebuild the schedule over it.

    Three criteria own trainable parameters, and adding a param group *after* the
    scheduler exists leaves `LambdaLR` holding one `lr_lambda` for two groups;
    torch >= 2.6 zips them with `strict=True`, so the first `scheduler.step()`
    raises. Every method that adds a group must therefore rebuild the scheduler in
    the same breath, and keeping the two steps apart is what let CDM drift into
    that bug once already.
    """
    ctx.optimizer.add_param_group({"params": parameters, "lr": lr})
    ctx.scheduler = ctx.build_scheduler()
