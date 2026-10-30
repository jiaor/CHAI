.PHONY: native test lint wheel wheel-check clean web-models web-vendor web web-test
native:                 ## build the C/C++/Fortran reference implementations, where their sources are present
	@if [ -d native ]; then $(MAKE) -C native; else echo "native/ is not in this tree (the reference sources are not redistributed); the tests that need them skip"; fi
test: native            ## run the test suite
	uv run pytest -q
lint:
	uv run ruff check src tests
wheel:                  ## build dist/chai_thermo-<version>-py3-none-any.whl (and the sdist)
	uv build
	uvx twine check dist/chai_thermo-$$(uv version --short)*
PY ?= 3.12              # make wheel-check PY=3.11 checks the other Python the package claims
wheel-check: wheel      ## install the wheel into a fresh venv outside the checkout, and run inference there
	set -e; d=$$(mktemp -d); trap 'rm -rf $$d' EXIT; \
	uv venv -q -p $(PY) $$d/venv; \
	uv pip install -q -p $$d/venv dist/chai_thermo-$$(uv version --short)-py3-none-any.whl; \
	cd $$d && venv/bin/python $(CURDIR)/scripts/wheel_smoke.py

# --- CHAI in the browser (web/, docs/browser.md) ---------------------------------------
# Node is needed for the vendored runtime and the tests, not by the page.  This machine's is a conda
# env (`conda create -n chai-web -c conda-forge nodejs=22`); WEB_PATH puts it first on the PATH.
WEB_PATH ?= $(HOME)/miniconda3/envs/chai-web/bin
ORT_VERSION := $(shell [ -f web/package.json ] && sed -n 's/.*"onnxruntime-web": "\(.*\)".*/\1/p' web/package.json)
ORT_DIST := web/node_modules/onnxruntime-web/dist
web-models web-vendor web web-test: export PATH := $(WEB_PATH):$(PATH)
# WEB_FAMILY picks the family to export; empty is the one api.load() ships.  For example
#   make web-models WEB_FAMILY="--members 100=results/<run_100> 250=results/<run_250> --chooser results/<chooser>"
WEB_FAMILY ?=
web-models:             ## export the family to ONNX (web/models/) and write the port's fixtures
	uv run --group web python scripts/export_onnx.py $(WEB_FAMILY) --fixtures web/test/fixtures.json
web-vendor:             ## ONNX Runtime Web, Archivo and Courier Prime, pinned in web/package.json, into web/vendor/
	cd web && npm ci --no-audit --no-fund
	grep -q "vendor/ort-$(ORT_VERSION)/" web/js/worker.js || \
		{ echo "web/js/worker.js does not import ONNX Runtime Web $(ORT_VERSION)"; exit 1; }
	rm -rf web/vendor && mkdir -p web/vendor/ort-$(ORT_VERSION)
	cp $(ORT_DIST)/ort.wasm.min.mjs $(ORT_DIST)/ort-wasm-simd-threaded.mjs $(ORT_DIST)/ort-wasm-simd-threaded.wasm \
		web/vendor/ort-$(ORT_VERSION)/
	echo "onnxruntime-web $(ORT_VERSION), MIT licence, Copyright (c) Microsoft Corporation." \
		"https://github.com/microsoft/onnxruntime" > web/vendor/ort-$(ORT_VERSION)/NOTICE
	mkdir -p web/vendor/archivo-5.3.0
	cp web/node_modules/@fontsource-variable/archivo/files/archivo-latin-wght-normal.woff2 \
		web/node_modules/@fontsource-variable/archivo/LICENSE web/vendor/archivo-5.3.0/
	mkdir -p web/vendor/courier-prime-5.3.0
	cp web/node_modules/@fontsource/courier-prime/files/courier-prime-latin-700-normal.woff2 \
		web/node_modules/@fontsource/courier-prime/LICENSE web/vendor/courier-prime-5.3.0/
web: web-vendor         ## assemble dist/web/chai/ and dist/web/chai-stats/, the folders to upload beside wp-content
	test -f web/models/meta.json || { echo "no web/models/meta.json: run make web-models first"; exit 1; }
	rm -rf dist/web/chai && mkdir -p dist/web/chai/models
	cp -r web/index.html web/style.css web/js web/vendor web/run dist/web/chai/
	cp web/models/*.onnx web/models/meta.json web/models/example.json dist/web/chai/models/
	cp web/htaccess dist/web/chai/.htaccess
	cp LICENSE dist/web/chai/LICENSE.txt
	rm -rf dist/web/chai-stats && mkdir -p dist/web/chai-stats
	cp web/stats/hit.php dist/web/chai-stats/
	cp web/stats/htaccess dist/web/chai-stats/.htaccess
	touch dist/web/chai-stats/index.html
	cp LICENSE dist/web/chai-stats/LICENSE.txt
	du -sh dist/web/chai
web-test: web           ## the port against Python (Node), then the page in headless Chromium and Firefox
	cd web && npm test
	cd web && node test/browser.mjs ../dist/web/chai
clean:
	@[ -d native ] && $(MAKE) -C native clean || true
	rm -rf dist
	find . -name __pycache__ -type d -exec rm -rf {} +
