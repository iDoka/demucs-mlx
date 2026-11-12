INSTALL_DIR := $(HOME)/.local/bin
REPO_DIR    := $(shell cd "$(dir $(lastword $(MAKEFILE_LIST)))" && pwd)
# Use the repo-local .venv if present, otherwise the system python3.
PYTHON      ?= $(if $(wildcard $(REPO_DIR)/.venv/bin/python),$(REPO_DIR)/.venv/bin/python,python3)
COMMAND     := demucs-mlx

.PHONY: install uninstall deps help

help: ## Show this help
	@echo ""
	@echo "  demucs-mlx installer"
	@echo "  ────────────────────"
	@echo ""
	@echo "  make install     Install demucs-mlx command into your PATH"
	@echo "  make uninstall   Remove demucs-mlx from your PATH"
	@echo "  make deps        Install demucs-mlx package (editable) + dependencies"
	@echo ""

deps: ## Install demucs-mlx package (editable) and its dependencies
	@echo "Installing demucs-mlx and dependencies..."
	@$(PYTHON) -m pip install -e "$(REPO_DIR)" -q
	@echo "Done."

install: deps ## Install demucs-mlx into ~/.local/bin
	@mkdir -p "$(INSTALL_DIR)"
	@echo '#!/bin/bash' > "$(INSTALL_DIR)/$(COMMAND)"
	@echo '# demucs-mlx — Music source separation on Apple Silicon' >> "$(INSTALL_DIR)/$(COMMAND)"
	@echo 'exec "$(PYTHON)" -m demucs_mlx "$$@"' >> "$(INSTALL_DIR)/$(COMMAND)"
	@chmod +x "$(INSTALL_DIR)/$(COMMAND)"
	@echo ""
	@echo "  ✓ Installed $(COMMAND) → $(INSTALL_DIR)/$(COMMAND)"
	@echo ""
	@if echo "$$PATH" | tr ':' '\n' | grep -qx "$(INSTALL_DIR)"; then \
		echo "  Ready! Try: $(COMMAND) --help"; \
	else \
		echo "  ⚠ $(INSTALL_DIR) is not in your PATH."; \
		echo ""; \
		echo "  Add this line to your shell profile (~/.zshrc or ~/.bashrc):"; \
		echo ""; \
		echo "    export PATH=\"$(INSTALL_DIR):\$$PATH\""; \
		echo ""; \
		echo "  Then restart your terminal or run: source ~/.zshrc"; \
	fi
	@echo ""

uninstall: ## Remove demucs-mlx from PATH
	@rm -f "$(INSTALL_DIR)/$(COMMAND)"
	@echo "  ✓ Removed $(COMMAND) from $(INSTALL_DIR)"
