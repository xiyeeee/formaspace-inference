import unittest
from types import SimpleNamespace
from unittest.mock import patch

import gradio_app


class GradioConcurrencyTests(unittest.TestCase):
    def test_generation_endpoints_share_one_pipeline_slot(self):
        # Build only the UI; importing gradio_app does not load model weights.
        settings = {
            'args': SimpleNamespace(model_path='tencent/Hunyuan3D-2mini',
                                    subfolder='hunyuan3d-dit-v2-mini', enable_t23d=False),
            'MV_MODE': False,
            'TURBO_MODE': False,
            'HAS_TEXTUREGEN': False,
            'HAS_T2I': False,
            'HTML_OUTPUT_PLACEHOLDER': '',
            'SUPPORTED_FORMATS': ['glb'],
            'example_is': [],
            'example_ts': [],
            'example_mvs': [],
        }
        with patch('gradio.utils.colab_check', return_value=False), patch.multiple(gradio_app, create=True, **settings):
            app = gradio_app.build_app()
        endpoints = {function.api_name: function for function in app.fns.values()}
        shape = endpoints['shape_generation']
        textured = endpoints['generation_all']
        self.assertEqual(shape.concurrency_id, textured.concurrency_id)
        self.assertEqual(shape.concurrency_limit, 1)
        self.assertEqual(textured.concurrency_limit, 1)


if __name__ == '__main__':
    unittest.main()
