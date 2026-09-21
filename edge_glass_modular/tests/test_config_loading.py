"""Regression tests: config loading (src/config.py).

Verifies the retained YAML configs parse into ExperimentConfig and that the
surfaces retired during the architecture cleanup do NOT reappear:

* all 9 configs/*.yaml parse without error;
* decoder/Qwen fields (model_name, quantization, LoRA, device_map) survive
  parsing;
* retired surfaces are absent from the parsed dataclass surfaces: 'fusion',
  standalone TRM decoder fields, removed Path-C fields;
* `from_dict` filters unknown/legacy keys silently (documented behavior) at
  every nesting level (decoder, encoders, top level);
* `device_map` and trainer fields relied on by the training script parse
  correctly;
* the "at least one encoder or decoder" validation still fires on empty input.

No network, no model loading — pure YAML/dataclass logic.
"""

import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
for p in {str(REPO_ROOT), str(REPO_ROOT / "src"), str(REPO_ROOT.parent)}:
    if p not in sys.path:
        sys.path.insert(0, p)

from config import (  # noqa: E402
    ExperimentConfig, DecoderConfig, EncoderConfig, load_config,
)

RETIRED_TOP = ("fusion", "fused", "path_c", "trm_decoder")
RETIRED_DECODER = ("trm_config", "num_trm_layers", "use_trm_standalone", "fusion_dim")
RETIRED_ENCODER = ("fusion_dim",)


def _retired_probe(cls, names):
    return {n: n in cls.__dataclass_fields__ for n in names}


def test_all_nine_configs_parse():
    files = sorted((REPO_ROOT / "configs").glob("*.yaml"))
    assert len(files) == 9, f"expected 9 retained configs, found {len(files)}: {files}"
    for f in files:
        cfg = load_config(str(f))
        assert isinstance(cfg, ExperimentConfig), f
        assert cfg.name, f"config {f.name} lost its name"


def test_decoder_fields_survive_parsing():
    cfg = load_config(str(REPO_ROOT / "configs" / "trm_vlm_qa_qwen2.5-3b.yaml"))
    dec = cfg.decoder
    assert dec is not None, "decoder block dropped by from_dict"
    assert dec.model_name, "decoder.model_name missing"
    assert dec.type == "qwen"
    # currently relied-on device placement field parses with correct type
    assert isinstance(dec.device_map, (str, type(None)))
    # LoRA fields
    assert isinstance(dec.use_lora, bool)
    assert dec.lora_r > 0 and dec.lora_alpha > 0
    assert isinstance(dec.lora_target_modules, list) and dec.lora_target_modules


def test_retired_surfaces_absent_from_dataclasses():
    top = _retired_probe(ExperimentConfig, RETIRED_TOP)
    dec = _retired_probe(DecoderConfig, RETIRED_DECODER)
    enc = _retired_probe(EncoderConfig, RETIRED_ENCODER)
    assert not any(top.values()), f"retired top-level fields present: {top}"
    assert not any(dec.values()), f"retired decoder fields present: {dec}"
    assert not any(enc.values()), f"retired encoder fields present: {enc}"


def test_retired_keys_in_yaml_do_not_crash_and_are_dropped():
    """Unknown legacy keys (e.g. from retired configs) are silently filtered —
    the documented from_dict behavior — at decoder, encoder, and top level."""
    cfg = ExperimentConfig.from_dict({
        "name": "legacy",
        "fusion": {"strategy": "concat"},            # retired top-level
        "path_c": {"llm_projector_dim": 28672},      # retired Path-C field
        "decoder": {
            "model_name": "Qwen/Qwen2.5-3B-Instruct",
            "device_map": None,
            "trm_config": {"hidden_dim": 8},         # retired TRM decoder field
            "nonexistent_future_key": 42,            # unknown key
        },
        "vision_encoder": {
            "model_name": "openai/clip-vit-large",
            "fusion_dim": 1024,                      # retired encoder field
        },
    })
    assert not hasattr(cfg, "fusion") or "fusion" not in ExperimentConfig.__dataclass_fields__
    assert cfg.decoder is not None
    assert cfg.decoder.model_name == "Qwen/Qwen2.5-3B-Instruct"
    assert cfg.decoder.device_map is None
    assert not hasattr(cfg.decoder, "trm_config")
    assert cfg.vision_encoder is not None
    assert cfg.vision_encoder.model_name == "openai/clip-vit-large"
    assert not hasattr(cfg.vision_encoder, "fusion_dim")


def test_device_map_and_trainer_fields_parse():
    cfg = ExperimentConfig.from_dict({
        "name": "devicemap",
        "decoder": {"model_name": "Qwen/Qwen2.5-3B-Instruct", "device_map": "auto"},
        "trainer": {"output_dir": "./out", "epochs": 2, "use_wandb": False},
        "training": {"output_dir": "./out"},
    })
    assert cfg.decoder.device_map == "auto"
    assert cfg.trainer.output_dir == "./out"
    assert cfg.trainer.epochs == 2
    assert cfg.trainer.use_wandb is False
    # cross-population from a single 'training' block still works
    cfg2 = ExperimentConfig.from_dict({
        "decoder": {"model_name": "Qwen/Qwen2.5-3B-Instruct"},
        "training": {"output_dir": "./x", "epochs": 4},
    })
    assert cfg2.trainer.output_dir == "./x"
    assert cfg2.trainer.epochs == 4


def test_empty_config_rejected():
    try:
        ExperimentConfig.from_dict({})
        raise AssertionError("expected ValueError for config with no encoders/decoder")
    except ValueError as e:
        assert "encoder or decoder" in str(e)


def test_no_decoder_config_defaults_to_none_not_qwen():
    """Alignment-only configs must not silently materialize a decoder."""
    cfg = ExperimentConfig.from_dict({
        "vision_encoder": {"model_name": "openai/clip-vit-large"},
        "text_encoder": {"model_name": "sentence-transformers/all-mpnet-base-v2"},
    })
    assert cfg.decoder is None


def test_yaml_roundtrip_save_load():
    cfg = ExperimentConfig.from_dict({
        "name": "roundtrip",
        "decoder": {"model_name": "Qwen/Qwen2.5-3B-Instruct", "device_map": "auto"},
    })
    import tempfile
    p = Path(tempfile.mkdtemp()) / "rt.yaml"
    cfg.save(str(p))
    raw = yaml.safe_load(open(p))
    # saved dict must not resurrect retired surfaces either
    assert "fusion" not in raw
    loaded = ExperimentConfig.from_yaml(str(p))
    assert loaded.decoder.model_name == "Qwen/Qwen2.5-3B-Instruct"
    assert loaded.decoder.device_map == "auto"
    assert loaded.name == "roundtrip"


if __name__ == "__main__":
    test_all_nine_configs_parse()
    print("PASS all_nine_configs_parse")
    test_decoder_fields_survive_parsing()
    print("PASS decoder_fields_survive_parsing")
    test_retired_surfaces_absent_from_dataclasses()
    print("PASS retired_surfaces_absent_from_dataclasses")
    test_retired_keys_in_yaml_do_not_crash_and_are_dropped()
    print("PASS retired_keys_in_yaml_do_not_crash_and_are_dropped")
    test_device_map_and_trainer_fields_parse()
    print("PASS device_map_and_trainer_fields_parse")
    test_empty_config_rejected()
    print("PASS empty_config_rejected")
    test_no_decoder_config_defaults_to_none_not_qwen()
    print("PASS no_decoder_config_defaults_to_none_not_qwen")
    test_yaml_roundtrip_save_load()
    print("PASS yaml_roundtrip_save_load")
    print("ALL CONFIG LOADING TESTS PASSED")
