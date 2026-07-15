"""Smoke test for B1 MCP protocol repairs.

Checks:
  - MCP pretrain and finetune both feed the encoder 8 tokens.
  - No pretrain-only calendar projection remains in the main iGT model.
  - The learnable MCP mask token receives gradient.
  - A configured but missing pretrained_path fails loudly.
"""

import logging
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from physformer.models.physformer.igt_model import PhysFormeriGT
from physformer.loss import PretrainLoss
from physformer.train.physformer_exp import PhysFormerExperiment


def test_igt_mcp_uses_eight_tokens():
    model = PhysFormeriGT(
        enc_in=6,
        seq_len=16,
        pred_len=8,
        d_model=64,
        n_heads=4,
        e_layers=2,
        d_ff=128,
        time_feat_dim=10,
    )
    assert not hasattr(model, "calendar_proj")

    token_shapes = []

    def capture_encoder_input(_module, inputs):
        token_shapes.append(tuple(inputs[0].shape))

    handle = model.encoder.register_forward_pre_hook(capture_encoder_input)
    x_comp = torch.randn(2, 16, 5)
    x_weather = torch.randn(2, 8, 3)
    y_mark = torch.randn(2, 8, 10)

    out = model(x_component_hist=x_comp, x_weather_future=x_weather, y_mark=y_mark)
    assert out["pred_net"].shape == (2, 8, 1)

    mask = torch.tensor([[True, False, False, False], [False, True, False, True]])
    out_masked = model(
        x_component_hist=x_comp,
        x_weather_future=x_weather,
        y_mark=y_mark,
        mask_indices=mask,
    )
    assert out_masked["pred_net"].shape == (2, 8, 1)
    assert token_shapes == [(2, 8, 64), (2, 8, 64)]

    criterion = PretrainLoss(lambda_net=1.0)
    y_target = torch.randn(2, 8, 1)
    y_aux = torch.randn(2, 8, 5)
    loss, debug, terms = criterion(out_masked, y_target, y_aux, mask)
    assert "comp_loss" in debug and "net_loss" in debug
    assert "comp_loss" in terms and "net_loss" in terms
    loss.backward()
    assert model.mask_token.grad is not None
    assert torch.isfinite(model.mask_token.grad).all()
    handle.remove()


def test_missing_pretrained_path_fails():
    exp = PhysFormerExperiment.__new__(PhysFormerExperiment)
    exp.args = SimpleNamespace(
        pretrained_path="tmp/__missing_b1_checkpoint__.pth",
        load_pretrained_scaler_buffers=False,
    )
    exp.logger = logging.getLogger("smoke_test_l1_l5")

    try:
        exp._load_pretrained_weights()
    except FileNotFoundError as exc:
        assert "Configured pretrained_path does not exist" in str(exc)
    else:
        raise AssertionError("missing pretrained_path did not raise FileNotFoundError")

    exp.args.pretrained_path = None
    assert exp._load_pretrained_weights() is False


def test_pretrained_loader_skips_scaler_buffers():
    source = PhysFormeriGT(
        enc_in=6,
        seq_len=16,
        pred_len=8,
        d_model=64,
        n_heads=4,
        e_layers=2,
        d_ff=128,
    )
    target = PhysFormeriGT(
        enc_in=6,
        seq_len=16,
        pred_len=8,
        d_model=64,
        n_heads=4,
        e_layers=2,
        d_ff=128,
    )
    original_target_mean = target.target_mean.detach().clone()
    state = {key: value.detach().clone() for key, value in source.state_dict().items()}
    for key in ("target_mean", "target_std", "aux_mean", "aux_std"):
        state[key] = torch.full_like(state[key], 123.0)

    exp = PhysFormerExperiment.__new__(PhysFormerExperiment)
    exp.args = SimpleNamespace(load_pretrained_scaler_buffers=False)
    exp.device = torch.device("cpu")
    exp.model = target
    exp.scaler_params = {
        "target_mean": target.target_mean.detach().cpu().numpy(),
        "target_std": target.target_std.detach().cpu().numpy(),
        "aux_mean": target.aux_mean.detach().cpu().numpy(),
        "aux_std": target.aux_std.detach().cpu().numpy(),
    }
    exp.logger = logging.getLogger("smoke_test_l1_l5")
    exp._last_scaler_buffer_report = {}

    with tempfile.TemporaryDirectory() as tmpdir:
        checkpoint = Path(tmpdir) / "pretrained_checkpoint.pth"
        torch.save(state, checkpoint)
        report = exp._load_model_state(
            checkpoint,
            "smoke pretrained checkpoint",
            filter_scaler_buffers=True,
            strict=False,
        )

    assert set(report["skipped_scaler_buffers"]) == {
        "target_mean",
        "target_std",
        "aux_mean",
        "aux_std",
    }
    assert torch.allclose(target.target_mean, original_target_mean)


if __name__ == "__main__":
    test_igt_mcp_uses_eight_tokens()
    test_missing_pretrained_path_fails()
    test_pretrained_loader_skips_scaler_buffers()
    print("ALL B1 PROTOCOL SMOKE TESTS PASSED")
