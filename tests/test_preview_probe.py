import json
from pathlib import Path
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch
import trimesh

from scripts.preview_probe import (PreviewProbe, ProbeConfig, UnsafePreviewError,
                                   random_fingerprints, require_mesh, sample_surface)
from hy3dgen.shapegen.pipelines import Hunyuan3DDiTPipeline
from hy3dgen.shapegen.models.autoencoders.surface_extractors import Latent2MeshOutput


def tetrahedron():
    return trimesh.Trimesh(vertices=[[10, 0, 0], [11, 0, 0], [10, 1, 0], [10, 0, 1]],
                           faces=[[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]], process=False)


class FakeVae:
    scale_factor = 2

    def __call__(self, latents):
        self.received = latents
        return latents

    def latents2mesh(self, latents, **kwargs):
        mesh = tetrahedron()
        return [Latent2MeshOutput(mesh.vertices.copy(), mesh.faces.copy())]


class FakePipeline:
    def __init__(self):
        self.scheduler = SimpleNamespace(order=1)
        self.vae = FakeVae()
        self.snapshots = []
        self.callback_thread = None

    def _export(self, **kwargs):
        self.snapshots.append(kwargs)
        self.callback_thread = threading.get_ident()
        return Hunyuan3DDiTPipeline._export(self, **kwargs)

    def __call__(self, **kwargs):
        assert kwargs['callback_steps'] == 1
        for index in range(kwargs['num_inference_steps']):
            self.latent = torch.full((1, 2, 3), float(index + 1))
            kwargs['callback'](index, torch.tensor(index), SimpleNamespace(prev_sample=self.latent))
        return [tetrahedron()]


class PreviewProbeTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)
        self.config = ProbeConfig(device='cpu')

    def test_step24_uses_full_export_scaling_local_rng_and_same_thread(self):
        pipeline = FakePipeline()
        before = random_fingerprints('cpu')
        PreviewProbe(self.config, self.directory).run(pipeline, None)
        self.assertEqual(before, random_fingerprints('cpu'))
        self.assertEqual(len(pipeline.snapshots), 1)
        self.assertEqual(pipeline.callback_thread, threading.get_ident())
        self.assertTrue(torch.equal(pipeline.snapshots[0]['latents'], torch.full((1, 2, 3), 24.0)))
        self.assertTrue(torch.equal(pipeline.vae.received, torch.full((1, 2, 3), 12.0)))
        self.assertEqual(pipeline.snapshots[0]['octree_resolution'], 128)
        self.assertEqual(pipeline.snapshots[0]['mc_algo'], 'mc')
        self.assertTrue((self.directory / 'final.glb').is_file())
        points = json.loads((self.directory / 'preview_step_024.json').read_text())['positions']
        self.assertEqual(len(points), 5000)
        self.assertGreaterEqual(np.asarray(points)[:, 0].min(), 10 - 1e-12)  # original coordinates
        events = [json.loads(line) for line in (self.directory / 'events.jsonl').read_text().splitlines()]
        self.assertEqual([event['sequence'] for event in events], list(range(1, len(events) + 1)))
        self.assertEqual([event['completedSteps'] for event in events if event['type'] == 'sampling_step'], list(range(1, 31)))
        saved_index = next(index for index, event in enumerate(events) if event['type'] == 'preview_saved')
        self.assertTrue(all(event['previewVersion'] == 0 for event in events[:saved_index]))
        self.assertTrue(all(event['previewVersion'] == 1 for event in events[saved_index:]))

    def test_regular_preview_failure_records_failure_and_finishes_final(self):
        pipeline = FakePipeline()
        with patch.object(pipeline, '_export', side_effect=ValueError('ordinary decoder error')):
            PreviewProbe(self.config, self.directory).run(pipeline, None)
        summary = json.loads((self.directory / 'summary.json').read_text())
        self.assertEqual(summary['preview']['status'], 'failed')
        self.assertEqual(summary['finalStatus'], 'succeeded')
        self.assertFalse((self.directory / 'preview_step_024.json').exists())
        events = [json.loads(line) for line in (self.directory / 'events.jsonl').read_text().splitlines()]
        self.assertEqual(sum(event['type'] == 'preview_failed' for event in events), 1)
        self.assertTrue(all(event['previewVersion'] == 0 for event in events))

    def test_device_failure_aborts_and_does_not_publish_final(self):
        pipeline = FakePipeline()
        with patch.object(pipeline, '_export', side_effect=torch.OutOfMemoryError('test OOM')):
            with self.assertRaises(UnsafePreviewError):
                PreviewProbe(self.config, self.directory).run(pipeline, None)
        summary = json.loads((self.directory / 'summary.json').read_text())
        self.assertEqual(summary['finalStatus'], 'failed')
        self.assertFalse((self.directory / 'final.glb').exists())

    def test_actual_preview_rng_consumption_still_aborts_with_generator_name(self):
        pipeline = FakePipeline()
        original_export = pipeline._export
        previous = np.random.get_state()
        self.addCleanup(np.random.set_state, previous)

        def consuming_export(**kwargs):
            result = original_export(**kwargs)
            np.random.random()  # Deliberate violation, injected only by this regression test.
            return result

        with patch.object(pipeline, '_export', side_effect=consuming_export):
            with self.assertRaises(UnsafePreviewError):
                PreviewProbe(self.config, self.directory).run(pipeline, None)
        summary = json.loads((self.directory / 'summary.json').read_text())
        self.assertEqual(summary['preview']['changedRandomGenerators'], ['numpy'])
        self.assertFalse(summary['preview']['randomStateUnchanged'])
        self.assertFalse((self.directory / 'final.glb').exists())

    def test_disabled_preview_never_decodes_and_degenerate_surface_is_rejected(self):
        pipeline = FakePipeline()
        PreviewProbe(self.config, self.directory, enabled=False).run(pipeline, None)
        self.assertEqual(pipeline.snapshots, [])
        self.assertEqual(list(self.directory.glob('preview_step*')), [])
        degenerate = trimesh.Trimesh(vertices=[[0, 0, 0]] * 3, faces=[[0, 1, 2]], process=False)
        with self.assertRaises(ValueError):
            sample_surface(degenerate, 5000, np.random.default_rng(1234))

    def test_surface_sampling_is_area_weighted_and_leaves_global_rng_untouched(self):
        mesh = trimesh.Trimesh(vertices=[[0, 0, 0], [1, 0, 0], [0, 1, 0],
                                       [10, 0, 0], [13, 0, 0], [10, 3, 0]],
                               faces=[[0, 1, 2], [3, 4, 5]], process=False)
        before = random_fingerprints('cpu')
        points = sample_surface(mesh, 5000, np.random.default_rng(1234))
        self.assertEqual(before, random_fingerprints('cpu'))
        self.assertAlmostEqual(float(np.mean(points[:, 0] >= 10)), 0.9, delta=0.02)

    def test_invalid_meshes_are_rejected_before_sampling(self):
        invalid_mesh = tetrahedron()
        invalid_mesh.vertices[0, 0] = np.nan
        for outputs in ([], [None], [trimesh.Trimesh()], [invalid_mesh]):
            with self.subTest(outputs=type(outputs[0]).__name__ if outputs else 'empty'):
                with self.assertRaises(ValueError):
                    require_mesh(outputs)

    def test_failed_preview_save_does_not_increment_version_or_stop_sampling(self):
        with patch('hy3dgen.shapegen.preview.save_points', side_effect=OSError('disk unavailable')):
            PreviewProbe(self.config, self.directory).run(FakePipeline(), None)
        events = [json.loads(line) for line in (self.directory / 'events.jsonl').read_text().splitlines()]
        self.assertTrue(all(event['previewVersion'] == 0 for event in events))
        self.assertEqual(events[-1]['type'], 'task_completed')
        self.assertTrue(any(event['type'] == 'preview_failed' for event in events))


if __name__ == '__main__':
    unittest.main()
