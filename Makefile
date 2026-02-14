.PHONY: install install-web demo train-codec train-model ensure-checkpoints generate serve-ui test

MANIFEST ?= data/demo/manifest.txt
PRESET ?= tiny

install:
	pip install -e ".[dev]"

install-web:
	pip install -e ".[dev,web]"

demo:
	python scripts/create_demo_dataset.py --output-dir data/demo --num-samples 20

train-codec:
	python scripts/train_codec.py --manifest $(MANIFEST) --output checkpoints/codec --preset $(or $(PRESET),small)

train-model:
	python scripts/train_model.py --manifest $(MANIFEST) --codec checkpoints/codec/codec_final.pt --output checkpoints/model --preset $(or $(PRESET),small)

ensure-checkpoints: checkpoints/codec/codec_final.pt checkpoints/model/model_final.pt

data/demo/manifest.txt:
	$(MAKE) demo

checkpoints/codec/codec_final.pt: $(MANIFEST)
	python scripts/train_codec.py --manifest $(MANIFEST) --output checkpoints/codec --preset $(PRESET) --epochs 5 --batch-size 4

checkpoints/model/model_final.pt: checkpoints/codec/codec_final.pt $(MANIFEST)
	python scripts/train_model.py --manifest $(MANIFEST) --codec checkpoints/codec/codec_final.pt --output checkpoints/model --preset $(PRESET) --epochs 3 --batch-size 4 --max-frames 80

generate:
	python scripts/generate.py --prompt "$(PROMPT)" --codec $(CODEC) --model $(MODEL) --output output.wav --preset $(or $(PRESET),small)

serve-ui:
	python scripts/serve_ui.py

test:
	PYTHONPATH=src pytest tests/ -v
