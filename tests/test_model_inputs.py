# 학습과 추론에서 사용하는 비트 컨텍스트 입력을 검증하는 테스트
import importlib
import importlib.util
import unittest

import numpy as np
import torch
import torch.nn as nn

from src.agents.anomaly_detector_agent import AnomalyDetectorAgent


class RecordingModel(nn.Module):
    def __init__(self, bit_len: int, recon_dim: int):
        super().__init__()
        self.bit_enc = nn.Module()
        self.bit_enc.fc = nn.Sequential(nn.Linear(bit_len, 1))
        self.recon_dim = recon_dim
        self.last_bits = None

    def forward(self, batch):
        self.last_bits = batch["bits"].detach().cpu().clone()
        return {
            "recon": torch.zeros(batch["high"].shape[0], self.recon_dim, device=batch["high"].device)
        }


class ModelInputTest(unittest.TestCase):
    def _module(self):
        spec = importlib.util.find_spec("src.utils.model_inputs")
        self.assertIsNotNone(spec, "src.utils.model_inputs 모듈이 필요합니다.")
        return importlib.import_module("src.utils.model_inputs")

    def test_explicit_context_is_padded_without_reordering(self):
        module = self._module()

        context = module.build_bit_context([], bit_len=5, context_bits=np.array([1.0, 0.0, 0.5]))

        np.testing.assert_array_equal(context, np.array([1.0, 0.0, 0.5, 0.0, 0.0], dtype=np.float32))

    def test_discrete_and_bitfield_sensors_build_context(self):
        module = self._module()
        sensor_windows = [
            {"type": "discrete", "raw_signal": np.array([0.0, 1.0])},
            {"type": "bits", "raw_signal": np.array([5])},
        ]

        context = module.build_bit_context(sensor_windows, bit_len=4)

        np.testing.assert_array_equal(context, np.array([1.0, 1.0, 0.0, 1.0], dtype=np.float32))

    def test_agent_passes_explicit_context_to_model(self):
        agent = AnomalyDetectorAgent(
            "test",
            {
                "reference_rate": 1,
                "window_sec": 4,
                "dataset_mode": "nasa",
                "model_config": {
                    "high_ch": 1,
                    "mid_ch": 1,
                    "low_dim": 1,
                    "bit_len": 3,
                    "enc_dim": 8,
                    "proj_dim": 4,
                    "recon_dim": 4,
                    "pred_steps": 1,
                    "time_tokens": 2,
                },
            },
        )
        recorder = RecordingModel(bit_len=3, recon_dim=4)
        agent.model = recorder
        agent.model_ready = True
        sensor_windows = [
            {
                "id": 0,
                "raw_signal": np.array([0.1, 0.2, 0.3, 0.4]),
                "raw_rate": 1,
                "type": "1word",
            }
        ]

        agent.process(sensor_windows, context_bits=np.array([1.0, 0.0, 1.0]))

        np.testing.assert_array_equal(recorder.last_bits.numpy(), np.array([[1.0, 0.0, 1.0]], dtype=np.float32))


if __name__ == "__main__":
    unittest.main()
