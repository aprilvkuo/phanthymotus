# OCR Build Smoke Test Design

## Goal

Run a real PP-OCRv6 Tiny inference during both CPU and Jetson image builds, print the recognized result in the build log, and fail the build when the fixture is not recognized adequately.

## Test Fixture

Use `perception/tests/fixtures/ocr_sample.png`, a generated photograph of a printed card containing:

```text
离线OCR测试
Phanthy Motus
机器人视觉 2026
Hello, World!
SN: OCR-20260722
```

The fixture is committed with the test so the input remains deterministic. Because
the repository globally ignores `*.png`, `.gitignore` will add the narrow exception
`!perception/tests/fixtures/*.png`; other PNG files remain ignored.

## End-to-End Test

Create `perception/tests/test_local_ocr.py`. The test will:

1. Configure visible INFO logging.
2. Construct `PPOCRv6ONNXAdapter` with `/models/ppocr-v6-tiny` and automatic model download enabled.
3. Read the committed PNG fixture as bytes.
4. Run `recognize()` through OpenCV preprocessing and ONNX Runtime detection/recognition.
5. Print the full OCR result as formatted UTF-8 JSON with immediate flushing.
6. Normalize whitespace and punctuation before checking stable text tokens, so harmless line segmentation does not fail the build.
7. Require the normalized output to contain the Chinese phrases, English words, and serial-number digits present on the card.

Missing dependencies, model-download failures, ONNX initialization failures, empty recognition, or missing expected tokens will fail the test.

## Docker Integration

Both `perception/Dockerfile` and `perception/Dockerfile.jetson` will declare:

```dockerfile
ARG RUN_OCR_E2E_TEST=1
```

The default value `1` runs the test on every image build. Setting the argument to `0` explicitly skips it for emergency or offline builds.

Each Dockerfile will copy the test and fixture after application code is present, then run:

```text
python3 -m unittest discover -s /work/tests -p test_local_ocr.py -v
```

The CPU Dockerfile copies from its `perception/` build context. The Jetson Dockerfile copies from the repository-root build context.

## Logging

The build output will include:

- model download and initialization messages;
- unittest test name and status;
- the full recognized JSON result;
- a normalized text line used for assertions;
- the standard unittest summary.

Users can display uncached, plain BuildKit output with `BUILDKIT_PROGRESS=plain` and a no-cache build when required.

## Scope

This change adds one fixture, one narrow `.gitignore` exception, one end-to-end test,
and build hooks in the two perception Dockerfiles. It does not change OCR inference
behavior, model files, or runtime configuration.
