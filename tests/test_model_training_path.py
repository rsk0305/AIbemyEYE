# 이상 탐지 모델의 시간 토큰과 학습 목적 경로를 검증하는 테스트
import unittest

import torch
from torch.utils.data import DataLoader

import main_anomaly_contrastive as anomaly_model


class ModelTrainingPathTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_gnn_without_adjacency_preserves_distinct_tokens(self):
        gnn = anomaly_model.SimpleGNN(in_dim=2, hidden=2, out_dim=2)
        with torch.no_grad():
            gnn.lin.weight.copy_(torch.eye(2))
            gnn.lin.bias.zero_()
            gnn.lin2.weight.copy_(torch.eye(2))
            gnn.lin2.bias.zero_()

        tokens = torch.tensor([[[1.0, 0.0], [0.0, 1.0], [2.0, 2.0]]])
        output = gnn(tokens)

        self.assertGreater(float(output.var(dim=1, unbiased=False).sum().detach()), 0.0)

    def test_gnn_accepts_shared_two_dimensional_adjacency(self):
        gnn = anomaly_model.SimpleGNN(in_dim=2, hidden=3, out_dim=2)
        tokens = torch.randn(2, 4, 2)
        adjacency = torch.eye(4)

        try:
            output = gnn(tokens, adjacency)
        except RuntimeError as error:
            self.fail(f"공유 인접행렬 처리에 실패했습니다. {error}")

        self.assertEqual(output.shape, (2, 4, 2))

    def test_fusion_uses_token_order(self):
        torch.manual_seed(11)
        fusion = anomaly_model.FusionTransformer(
            input_dim=4,
            nhead=2,
            num_layers=1,
            dim_feedforward=8,
        )
        fusion.eval()
        tokens = torch.tensor(
            [[[1.0, 0.0, 0.0, 0.0], [0.0, 2.0, 0.0, 0.0], [0.0, 0.0, 3.0, 0.0]]]
        )

        forward = fusion(tokens)
        reversed_order = fusion(tokens.flip(1))

        self.assertFalse(torch.allclose(forward, reversed_order, atol=1e-6))

    def test_pretraining_loss_reaches_decoder(self):
        objective = getattr(anomaly_model, "multimodal_pretraining_loss", None)
        self.assertTrue(callable(objective), "multimodal_pretraining_loss가 필요합니다.")

        torch.manual_seed(17)
        model = anomaly_model.MultiModalAnomalyModel(
            {
                "high_ch": 1,
                "mid_ch": 1,
                "low_dim": 1,
                "bit_len": 3,
                "enc_dim": 8,
                "proj_dim": 4,
                "recon_dim": 8,
                "pred_steps": 1,
                "time_tokens": 2,
            }
        )
        batch_a = {
            "high": torch.randn(3, 8, 1),
            "mid": torch.randn(3, 8, 1),
            "low": torch.randn(3, 1),
            "bits": torch.randn(3, 3),
        }
        batch_b = {key: value + 0.001 * torch.randn_like(value) for key, value in batch_a.items()}

        outputs_a = model(batch_a)
        outputs_b = model(batch_b)
        loss, parts = objective(outputs_a, outputs_b, batch_a, batch_b)
        loss.backward()

        decoder_gradient = sum(
            float(parameter.grad.abs().sum())
            for parameter in model.decoder.parameters()
            if parameter.grad is not None
        )
        self.assertGreater(decoder_gradient, 0.0)
        self.assertIn("contrastive", parts)
        self.assertIn("reconstruction", parts)

    def test_trainer_pretraining_updates_decoder(self):
        torch.manual_seed(23)
        config = {
            "high_ch": 1,
            "mid_ch": 1,
            "low_dim": 1,
            "bit_len": 3,
            "enc_dim": 8,
            "proj_dim": 4,
            "recon_dim": 8,
            "pred_steps": 1,
            "time_tokens": 2,
            "lr": 1e-2,
            "lambda_recon": 1.0,
        }
        model = anomaly_model.MultiModalAnomalyModel(config)
        trainer = anomaly_model.Trainer(model, torch.device("cpu"), config)
        samples = [
            {
                "high": torch.randn(8, 1),
                "mid": torch.randn(8, 1),
                "low": torch.randn(1),
                "bits": torch.randn(3),
            }
            for _ in range(3)
        ]
        loader = DataLoader(samples, batch_size=3, shuffle=False)
        before = [parameter.detach().clone() for parameter in model.decoder.parameters()]

        trainer.pretrain_contrastive(loader, epochs=1)

        changed = sum(
            float((old - new.detach()).abs().sum())
            for old, new in zip(before, model.decoder.parameters())
        )
        self.assertGreater(changed, 0.0)


if __name__ == "__main__":
    unittest.main()
