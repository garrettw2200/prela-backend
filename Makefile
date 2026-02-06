.PHONY: help up down logs clean test

help:
	@echo "Prela Backend - Available commands:"
	@echo "  make up       - Start all services"
	@echo "  make down     - Stop all services"
	@echo "  make logs     - View logs"
	@echo "  make clean    - Remove containers and volumes"
	@echo "  make test     - Run tests"

up:
	docker-compose up -d

down:
	docker-compose down

logs:
	docker-compose logs -f

clean:
	docker-compose down -v
	rm -rf __pycache__ .pytest_cache

test:
	pytest services/
