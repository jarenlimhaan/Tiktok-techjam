install:
	@echo "Installing dependencies..."
	@pip install pipx
	@pipx install poetry --force
	@pipx ensurepath
	@echo "If it fails just reopen the terminal and rerun make install"
	@cd backend && poetry install
	@cd frontend && npm install
	@echo "Dependencies installed successfully!✅"


# Run BE
run-be:
	@echo "Starting FastAPI Server..."
	@cd backend && poetry run python main.py


# Run FE
run-fe:
	@echo "Starting NextJS Application..."
	@cd frontend && npm run dev

# Run both fe and be
run:
	@echo "Starting FastAPI Server and NextJS Application..."
	@$(MAKE) run-be &
	@$(MAKE) run-fe &