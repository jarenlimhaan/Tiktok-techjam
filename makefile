# Run BE
run-be:
	@echo "Starting FastAPI Server..."
	@cd backend && python main.py


# Run FE
run-fe:
	@echo "Starting NextJS Application..."
	@cd frontend && npm run dev

# Run both fe and be
run:
	@echo "Starting FastAPI Server and NextJS Application..."
	@$(MAKE) run-be &
	@$(MAKE) run-fe &