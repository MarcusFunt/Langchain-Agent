COMPOSE ?= docker compose
IMAGE ?= langchain-agent:latest

.PHONY: build run dev

build:
$(COMPOSE) -f docker-compose.yml build

run:
$(COMPOSE) -f docker-compose.yml up

dev:
$(COMPOSE) -f docker-compose.yml up --build
