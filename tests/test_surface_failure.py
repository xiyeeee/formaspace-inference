from contextlib import redirect_stderr
import io
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import gradio as gr
import numpy as np
from PIL import Image
import torch

import gradio_app
from hy3dgen.shapegen.models.autoencoders.surface_extractors import Latent2MeshOutput, MCSurfaceExtractor
from hy3dgen.shapegen.pipelines import export_to_trimesh


class SurfaceFailureTests(unittest.TestCase):
    def test_no_surface_preserves_original_error_and_records_field_range(self):
        extractor = MCSurfaceExtractor()
        with self.assertRaisesRegex(RuntimeError, r'level=0.0.*finite=512/512.*finite_min=0.0.*finite_max=0.0') as caught:
            extractor.run(torch.zeros((8, 8, 8)), mc_level=0.0, bounds=1.01, octree_resolution=7)
        self.assertIn('No surface found', str(caught.exception.__cause__))

    def test_nonfinite_field_reports_missing_finite_range_without_fabrication(self):
        with self.assertRaisesRegex(RuntimeError, r'finite=0/512.*finite_min=None.*finite_max=None'):
            MCSurfaceExtractor().run(torch.full((8, 8, 8), float('nan')),
                                     mc_level=0.0, bounds=1.01, octree_resolution=7)

    def test_successful_surface_coordinates_are_unchanged(self):
        volume = torch.arange(8, dtype=torch.float32)[:, None, None].expand(8, 8, 8).contiguous() - 3.5
        vertices, faces = MCSurfaceExtractor().run(volume, mc_level=0.0, bounds=1.01, octree_resolution=7)
        self.assertGreater(len(faces), 0)
        self.assertTrue(np.isfinite(vertices).all())
        self.assertTrue(np.allclose(vertices[:, 0], 3.5 / 8 * 2.02 - 1.01))

    def test_batch_contract_still_returns_none_for_failed_item_and_logs_diagnostics(self):
        diagnostics = io.StringIO()
        with redirect_stderr(diagnostics):
            outputs = MCSurfaceExtractor()(torch.zeros((1, 8, 8, 8)),
                                           mc_level=0.0, bounds=1.01, octree_resolution=7)
        self.assertEqual(outputs, [None])
        self.assertIn('finite_min=0.0', diagnostics.getvalue())

    def run_gradio(self, outputs):
        settings = {
            'args': SimpleNamespace(model_path='test', subfolder='mini', texgen_model_path='unused'),
            'MV_MODE': False, 'i23d_worker': Mock(return_value=outputs),
            'export_to_trimesh': export_to_trimesh,
        }
        with patch.multiple(gradio_app, create=True, **settings), patch.object(gradio_app, 'gen_save_folder', return_value='unused'):
            return gradio_app._gen_shape(image=Image.new('RGBA', (16, 16)), seed=1234)

    def test_gradio_empty_decoder_result_becomes_business_error_instead_of_attribute_error(self):
        for outputs in ([], [None], None):
            with self.subTest(outputs=outputs), self.assertRaises(gr.Error) as caught:
                self.run_gradio(outputs)
            self.assertIn('未能提取有效模型表面', str(caught.exception))

    def test_gradio_valid_mesh_still_returns_geometry_and_stats(self):
        output = Latent2MeshOutput(
            mesh_v=np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32),
            mesh_f=np.array([[0, 1, 2]]),
        )
        mesh, image, directory, stats, seed = self.run_gradio([output])
        self.assertEqual(stats['number_of_faces'], 1)
        self.assertEqual(stats['number_of_vertices'], 3)
        self.assertEqual(seed, 1234)


if __name__ == '__main__':
    unittest.main()
