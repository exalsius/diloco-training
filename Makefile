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

build:
	echo "*"
	echo "* Building exalsius docker image"
	echo "*"
	docker build -t $(IMAGE_NAME) -f ./Dockerfile .
.PHONY: build

