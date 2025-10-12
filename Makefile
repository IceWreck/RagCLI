#!make
-include .env
export $(shell sed 's/=.*//' .env)
SHELL := /bin/bash


.PHONY: *

develop:
	uv venv ./.venv
	uv sync
	uv pip install -e .
	touch .env

format:
	uv run ruff format ./src

lint:
	uv run ruff check --fix src/
	uv run ruff check

mypy:
	uv run mypy ./src

qdrant-start:
	podman run -d --name qdrant -p 6333:6333 -p 6334:6334 qdrant/qdrant:latest

qdrant-stop:
	podman stop qdrant
	podman rm qdrant

qdrant-logs:
	podman logs -f qdrant
