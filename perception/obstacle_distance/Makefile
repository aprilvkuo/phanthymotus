PY ?= python3
MODULE := .

.PHONY: help test train export docker push

help:
	@echo "make test   - 跑 pytest (tests/)"
	@echo "make train  - 训练 (需数据)"
	@echo "make export - 导出 onnx"
	@echo "make docker - docker compose 跑测试"
	@echo "make push   - git push 到远端分支"

test:
	cd $(MODULE) && $(PY) -m pytest tests/ -q -rs

train:
	cd $(MODULE) && $(PY) src/train.py --config configs/default.yaml

export:
	cd $(MODULE) && $(PY) src/export_onnx.py

docker:
	docker compose up --build test

push:
	git push -u origin feat/obstacle-distance-baseline
