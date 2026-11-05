"""Check action semantics, resume continuity, and incomplete/corrupt checkpoint recovery."""
import itertools
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from dataset import action_statistics, normalize, unnormalize, encode_actions, decode_actions, ResumableSampler, sha256
from train import latest_checkpoint, verify_checkpoint


class TrainingContractTests(unittest.TestCase):
    def test_joint_and_gripper_normalization_and_quantization(self):
        rng = np.random.default_rng(42)
        actions = rng.uniform([-3., -2., 1., -2.5, -1.8, -.6, 0.], [3.5, -1., 2.5, -1.5, -1., .3, 1.], size=(2000, 7))
        actions[:, 6] = rng.integers(0, 2, 2000)
        stats = action_statistics(actions)
        clipped = np.clip(actions, stats['q01'], stats['q99'])
        np.testing.assert_allclose(unnormalize(normalize(actions, stats), stats), clipped, atol=1e-12)
        recovered = unnormalize(decode_actions(encode_actions(normalize(actions, stats), 32000), 32000), stats)
        tolerance = (np.asarray(stats['q99']) - stats['q01']) / 255
        self.assertTrue(np.all(np.abs(recovered - clipped) <= tolerance + 1e-12))
        np.testing.assert_array_equal(recovered[:, 6] >= .5, actions[:, 6] == 1)

    def test_nonfinite_and_degenerate_action_ranges_block_training(self):
        with self.assertRaises(ValueError):
            action_statistics(np.ones((20, 7)))
        values = np.arange(140, dtype=float).reshape(20, 7)
        values[0, 0] = np.nan
        with self.assertRaises(ValueError):
            action_statistics(values)

    def test_sample_resume_crosses_epoch_boundaries_without_replay(self):
        reference = list(itertools.islice(iter(ResumableSampler(13)), 130))
        for consumed in [0, 1, 12, 13, 14, 48, 109]:
            resumed = list(itertools.islice(iter(ResumableSampler(13, consumed=consumed)), 20))
            self.assertEqual(resumed, reference[consumed:consumed+20])

    def test_checkpoint_partial_ignored_and_corruption_not_silently_skipped(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / 'checkpoints/000250'
            path.mkdir(parents=True)
            (path / 'training_state.pt').write_bytes(b'checked optimizer state')
            (path / 'COMPLETE.json').write_text(json.dumps({'step': 250, 'sha256': {
                'training_state.pt': sha256(path / 'training_state.pt')}}))
            partial = path.with_name('000500.incomplete-1')
            partial.mkdir()
            self.assertEqual(latest_checkpoint(root), path)
            self.assertEqual(verify_checkpoint(path)['step'], 250)
            (path / 'training_state.pt').write_bytes(b'corrupted')
            with self.assertRaisesRegex(ValueError, 'Corrupt checkpoint'):
                latest_checkpoint(root)


if __name__ == '__main__':
    unittest.main()
