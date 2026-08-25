import sys

import pytest

from distiller import should_save_epoch
from main import get_config, parse_args


def test_save_every_cli_overrides_config_default(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        ["main.py", "--method", "talas", "--save_every", "3"],
    )

    args = parse_args()
    config = get_config(args.method, args)

    assert config.save_every == 3


def test_non_positive_save_every_is_rejected(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        ["main.py", "--method", "talas", "--save_every", "0"],
    )

    args = parse_args()
    with pytest.raises(ValueError, match="positive integer"):
        get_config(args.method, args)


def test_five_epoch_periodic_save_schedule_only_selects_epoch_three():
    selected = [
        epoch_index + 1
        for epoch_index in range(5)
        if should_save_epoch(epoch_index, save_every=3)
    ]

    assert selected == [3]
