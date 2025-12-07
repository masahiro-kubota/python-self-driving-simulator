# Minimal makefile for Sphinx documentation
#

# You can set these variables from the command line, and also
# from the environment for the first two.
SPHINXOPTS    ?=
SPHINXBUILD   ?= sphinx-build
SOURCEDIR     = source
BUILDDIR      = build

# Source code paths for API documentation
PACKAGES = ../core/src/core \
           ../simulators/core/src/simulator_core \
           ../simulators/simulator_kinematic/src/simulator_kinematic \
           ../experiment/runner/src/experiment_runner \
           ../experiment/training/src/experiment_training \
           ../ad_components/core/src/ad_component_core \
           ../ad_components/planning/pure_pursuit/src/pure_pursuit \
           ../ad_components/planning/planning_utils/src/planning_utils \
           ../ad_components/control/pid_controller/src/pid_controller \
           ../ad_components/control/neural_controller/src/neural_controller \
           ../dashboard/src/dashboard

# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

# Automatically generate API docs before building HTML
html:
	@echo "Generating API documentation from source..."
	uv run sphinx-apidoc -f -o "$(SOURCEDIR)" $(PACKAGES)
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

.PHONY: help Makefile html

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
%: Makefile
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
