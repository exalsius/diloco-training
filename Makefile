# Docker image configuration
IMAGE_NAME ?= diloco-training

# Build the Docker image
.PHONY: build
build:
	docker build -t $(IMAGE_NAME) .
