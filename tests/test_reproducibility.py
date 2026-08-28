# 벤치마크 난수 시드가 반복 실행에서 동일한 값을 만드는지 검증하는 테스트
import importlib
import importlib.util
import random
import unittest

import numpy as np
import torch


class ReproducibilityTest(unittest.TestCase):
    def test_seed_everything_repeats_python_numpy_and_torch_sequences(self):
        spec = importlib.util.find_spec("src.utils.reproducibility")
        self.assertIsNotNone(spec, "src.utils.reproducibility 모듈이 필요합니다.")
        module = importlib.import_module("src.utils.reproducibility")

        module.seed_everything(42)
        first = (random.random(), np.random.random(), torch.rand(3))
        module.seed_everything(42)
        second = (random.random(), np.random.random(), torch.rand(3))

        self.assertEqual(first[0], second[0])
        self.assertEqual(first[1], second[1])
        self.assertTrue(torch.equal(first[2], second[2]))


if __name__ == "__main__":
    unittest.main()
