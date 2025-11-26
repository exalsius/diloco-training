.DEFAULT_GOAL := build

# Docker image configuration
IMAGE_NAME ?= diloco-training

SHELL = /bin/bash
.SHELLFLAGS += -e -o pipefail
.ONESHELL:
$(V).SILENT:

format:
	echo "*"
	echo "* Running formatter..."
	echo "*"
	black .
	isort --profile black .
.PHONY: format

lint:
	echo "*"
	echo "* Running linter..."
	echo "*"
	ruff check ./ --ignore=E501,F403,F405
.PHONY: lint

test: format lint
	echo "*"
	echo "* Running tests..."
	echo "*"
	pytest -v
.PHONY: test

lock-cuda:
	echo "*"
	echo "* Generating CUDA lock file (uv.lock)..."
	echo "*"
	uv lock --index https://download.pytorch.org/whl/cu126 --index-strategy unsafe-best-match
.PHONY: lock-cuda

lock-rocm:
	echo "*"
	echo "* Generating ROCm lock file (uv.rocm.lock)..."
	echo "*"
	mv uv.lock uv.cuda.lock.tmp
	uv lock --index https://download.pytorch.org/whl/rocm6.3 --index-strategy unsafe-best-match
	mv uv.lock uv.rocm.lock
	mv uv.cuda.lock.tmp uv.lock
	echo "* Created uv.rocm.lock and restored uv.lock"
.PHONY: lock-rocm

lock-all: lock-cuda lock-rocm
	echo "*"
	echo "* Generated both CUDA and ROCm lock files"
	echo "*"
.PHONY: lock-all

setup-rocm:
	echo "*"
	echo "* Installing custom PyTorch Geometric wheels for ROCm"
	echo "* Run this after 'uv sync --extra rocm'"
	echo "*"
	./scripts/setup_rocm.sh
.PHONY: setup-rocm

build:
	echo "*"
	echo "* Building CUDA docker image (default)"
	echo "*"
	docker build -t $(IMAGE_NAME):cuda -f ./Dockerfile.cuda .
.PHONY: build

build-cuda:
	echo "*"
	echo "* Building CUDA docker image"
	echo "*"
	docker build -t $(IMAGE_NAME):cuda -f ./Dockerfile.cuda .
.PHONY: build-cuda

build-rocm:
	echo "*"
	echo "* Building ROCm docker image"
	echo "*"
	docker build -t $(IMAGE_NAME):rocm -f ./Dockerfile.rocm .
.PHONY: build-rocm

build-all: build-cuda build-rocm
	echo "*"
	echo "* Built both CUDA and ROCm images"
	echo "*"
.PHONY: build-all

