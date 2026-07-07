from pathlib import Path

import yaml

from config import Server_config
from server_fastapi import resolve_server_port


def test_server_config_default_port_is_5100():
    assert Server_config().port == 5100


def test_server_config_from_dict_uses_explicit_port():
    assert Server_config.from_dict({"port": 5000}).port == 5000


def test_default_config_server_port_matches_default():
    with Path("default_config.yml").open(encoding="utf-8") as file:
        default_config = yaml.safe_load(file)

    assert default_config["server"]["port"] == 5100


def test_resolve_server_port_uses_config_port_when_cli_port_is_omitted():
    assert resolve_server_port(configured_port=5100, cli_port=None) == 5100
    assert resolve_server_port(configured_port=5001, cli_port=None) == 5001


def test_resolve_server_port_prefers_cli_port():
    assert resolve_server_port(configured_port=5100, cli_port=5110) == 5110
