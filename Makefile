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
	uv tool run ruff format ./src

lint:
	uv tool run ruff check --fix src/
	uv tool run ruff check

mypy:
	uv run mypy ./src
