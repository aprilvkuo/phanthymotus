# OCR Build Smoke Test Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run real PP-OCRv6 Tiny inference by default in both perception image builds, print the recognized result, and fail when the fixture is not recognized.

**Architecture:** A dedicated unittest owns model initialization, image inference, logging, normalization, and assertions. Both Dockerfiles copy the same test assets and execute the test through a default-on `RUN_OCR_E2E_TEST=1` build argument.

**Tech Stack:** Python 3 unittest, PP-OCRv6 Tiny ONNX, OpenCV, ONNX Runtime, Docker

## Global Constraints

- Fixture path: `perception/tests/fixtures/ocr_sample.png`.
- Default build argument: `RUN_OCR_E2E_TEST=1`; value `0` skips the test.
- Default model directory: `/models/ppocr-v6-tiny`, overridable with `OCR_TEST_MODEL_DIR`.
- Missing models use the existing `OCR_MODEL_BASE_URL` download flow.
- Print UTF-8 JSON results and normalized recognized text before assertions.
- Ignore case, whitespace, and punctuation while requiring all stable expected tokens.
- Do not change runtime OCR inference behavior or model formats.

---

### Task 1: Add the End-to-End OCR Test

**Files:**
- Modify: `.gitignore`
- Create: `perception/tests/test_local_ocr.py`
- Add: `perception/tests/fixtures/ocr_sample.png`

**Interfaces:**
- Consumes: `plugins.ocr_local.PPOCRv6ONNXAdapter`, `OCR_MODEL_BASE_URL`, and optional `OCR_TEST_MODEL_DIR`.
- Produces: `LocalOCREndToEndTest.test_generated_card_recognizes_expected_text`.

- [ ] **Step 1: Verify the fixture is ignored**

Run `git check-ignore -v perception/tests/fixtures/ocr_sample.png`.

Expected: the repository-wide `*.png` rule is reported.

- [ ] **Step 2: Track only OCR fixture PNGs**

Add this immediately after `*.png` in `.gitignore`:

```gitignore
!perception/tests/fixtures/*.png
```

- [ ] **Step 3: Create `perception/tests/test_local_ocr.py`**

```python
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
    return "".join(character for character in text.casefold() if character.isalnum())


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
```

- [ ] **Step 4: Verify normalization and existing unit tests**

Run:

```bash
python3 -c "import sys; sys.path.insert(0, 'perception/tests'); from test_local_ocr import normalize_ocr_text; assert normalize_ocr_text('Hello, World!') == 'helloworld'; assert normalize_ocr_text('机器人视觉 2026') == '机器人视觉2026'"
cd perception && python3 -m unittest discover -s tests -p 'test_ocr_local.py' -v
```

Expected: normalization assertions succeed and six existing OCR helper tests pass. If importing the integration test fails because local OCR binary dependencies are absent, run the normalization function from its source inside the target image and record the local limitation.

### Task 2: Integrate the Test Into Both Docker Builds

**Files:**
- Modify: `perception/Dockerfile`
- Modify: `perception/Dockerfile.jetson`

**Interfaces:**
- Consumes: the test and fixture from Task 1.
- Produces: default-on build-time OCR inference in CPU and Jetson images.

- [ ] **Step 1: Verify hooks are absent**

Run `! rg -q 'RUN_OCR_E2E_TEST' perception/Dockerfile perception/Dockerfile.jetson`.

Expected: exit 0.

- [ ] **Step 2: Add the CPU hook**

Declare `ARG RUN_OCR_E2E_TEST=1` after `FROM`. After application files are copied, add:

```dockerfile
COPY tests/test_local_ocr.py /work/tests/test_local_ocr.py
COPY tests/fixtures/ocr_sample.png /work/tests/fixtures/ocr_sample.png
RUN if [ "${RUN_OCR_E2E_TEST}" = "1" ]; then \
        python3 -u -m unittest discover -s /work/tests -p "test_local_ocr.py" -v; \
    else \
        echo "Skipping OCR end-to-end build test (RUN_OCR_E2E_TEST=${RUN_OCR_E2E_TEST})"; \
    fi
```

- [ ] **Step 3: Add the Jetson hook**

Declare `ARG RUN_OCR_E2E_TEST=1` after `FROM`. After application files and the final OCR dependency check, add:

```dockerfile
COPY perception/tests/test_local_ocr.py /work/tests/test_local_ocr.py
COPY perception/tests/fixtures/ocr_sample.png /work/tests/fixtures/ocr_sample.png
RUN if [ "${RUN_OCR_E2E_TEST}" = "1" ]; then \
        python3 -u -m unittest discover -s /work/tests -p "test_local_ocr.py" -v; \
    else \
        echo "Skipping OCR end-to-end build test (RUN_OCR_E2E_TEST=${RUN_OCR_E2E_TEST})"; \
    fi
```

- [ ] **Step 4: Verify configuration and patch quality**

Run:

```bash
rg -n 'RUN_OCR_E2E_TEST|test_local_ocr|ocr_sample' perception/Dockerfile perception/Dockerfile.jetson
git diff --check
```

Expected: each Dockerfile declares the default-on argument, copies both assets, and runs unittest; no whitespace errors.

- [ ] **Step 5: Run a target build when available**

Run:

```bash
BUILDKIT_PROGRESS=plain ./deploy/build_perception.sh --variant jetson --mirror tuna
```

Expected: model logs, `OCR E2E RESULT` JSON, normalized text, and one passing unittest. If Docker or target architecture is unavailable, record the build as not run.

- [ ] **Step 6: Commit implementation files only**

```bash
git add .gitignore perception/tests/test_local_ocr.py \
    perception/tests/fixtures/ocr_sample.png \
    perception/Dockerfile perception/Dockerfile.jetson
git commit -m "test(perception): run OCR inference during image builds"
```
