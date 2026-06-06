from pathlib import Path

import pytest

from graspnt_rm.config import load_config, require_keys, validate_runtime_config


def test_load_config_reads_yaml(tmp_path):
    config_file = tmp_path / "config.yaml"
    config_file.write_text(
        """
safety:
  min_grasp_z: 0.05
""",
        encoding="utf-8",
    )

    config = load_config(config_file)

    assert config["safety"]["min_grasp_z"] == 0.05


def test_require_keys_reports_missing_path():
    with pytest.raises(KeyError, match="graspnet.root"):
        require_keys({"graspnet": {}}, ["graspnet.root"])


def test_validate_runtime_config_accepts_minimal_full_config():
    validate_runtime_config(
        {
            "graspnet": {
                "root": "/opt/graspnet",
                "checkpoint": "/opt/graspnet/checkpoint.tar",
            },
            "hand_eye": {
                "rotation": [1, 0, 0, 0, 1, 0, 0, 0, 1],
                "translation": [0.0, 0.0, 0.0],
            },
            "safety": {
                "gripper_length": 0.08,
                "min_grasp_z": 0.05,
            },
        }
    )


def test_validate_runtime_config_rejects_missing_graspnet_root():
    with pytest.raises(KeyError, match="graspnet.root"):
        validate_runtime_config(
            {
                "graspnet": {
                    "checkpoint": "/opt/graspnet/checkpoint.tar",
                },
                "hand_eye": {
                    "rotation": [1, 0, 0, 0, 1, 0, 0, 0, 1],
                    "translation": [0.0, 0.0, 0.0],
                },
                "safety": {
                    "gripper_length": 0.08,
                    "min_grasp_z": 0.05,
                },
            }
        )


def test_validate_runtime_config_rejects_missing_gripper_length():
    with pytest.raises(KeyError, match="safety.gripper_length"):
        validate_runtime_config(
            {
                "graspnet": {
                    "root": "/opt/graspnet",
                    "checkpoint": "/opt/graspnet/checkpoint.tar",
                },
                "hand_eye": {
                    "rotation": [1, 0, 0, 0, 1, 0, 0, 0, 1],
                    "translation": [0.0, 0.0, 0.0],
                },
                "safety": {
                    "min_grasp_z": 0.05,
                },
            }
        )
