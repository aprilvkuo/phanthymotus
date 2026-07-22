"""Build-time end-to-end smoke test for local PP-OCRv6 inference."""

from __future__ import annotations

import json
import logging
import os
import pathlib
import sys
import unittest


PERCEPTION_ROOT = pathlib.Path(__file__).parents[1]
FIXTURE_PATH = pathlib.Path(__file__).parent / "fixtures" / "ocr_sample.png"
sys.path.insert(0, str(PERCEPTION_ROOT))

from plugins.ocr_local import PPOCRv6ONNXAdapter  # noqa: E402


EXPECTED_TOKENS = (
    "离线OCR测试",
    "PhanthyMotus",
    "机器人视觉2026",
    "HelloWorld",
    "OCR20260722",
)


def normalize_ocr_text(text: str) -> str:
    """Ignore case, whitespace, and punctuation in comparisons."""
    return "".join(
        character for character in text.casefold() if character.isalnum()
    )


class LocalOCREndToEndTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(name)s] %(levelname)s %(message)s",
            stream=sys.stdout,
            force=True,
        )

    def test_generated_card_recognizes_expected_text(self):
        adapter = PPOCRv6ONNXAdapter(
            {
                "model_dir": os.environ.get(
                    "OCR_TEST_MODEL_DIR", "/models/ppocr-v6-tiny"
                ),
                "auto_download": True,
                "num_threads": 2,
                "det_limit_side_len": 1280,
                "det_thresh": 0.2,
                "det_box_thresh": 0.4,
                "det_unclip_ratio": 1.4,
                "rec_score_thresh": 0.3,
                "max_candidates": 1000,
            }
        )

        results = adapter.recognize(FIXTURE_PATH.read_bytes(), language="zh")
        print(
            "========== OCR E2E RESULT ==========\n"
            + json.dumps(results, ensure_ascii=False, indent=2),
            flush=True,
        )

        recognized_text = "\n".join(item["text"] for item in results)
        normalized_text = normalize_ocr_text(recognized_text)
        print(f"OCR normalized text: {normalized_text}", flush=True)

        self.assertTrue(results, "OCR returned no text for the generated fixture")
        for expected in EXPECTED_TOKENS:
            with self.subTest(expected=expected):
                self.assertIn(normalize_ocr_text(expected), normalized_text)


if __name__ == "__main__":
    unittest.main(verbosity=2)
