#!/usr/bin/env bash

# Required
export PROSAFEPRUNE_MODEL_ID="MODEL_ID"
export PROSAFEPRUNE_MODEL_PATH="/path/to/model"

# Required for evaluation
export PROSAFEPRUNE_WILDGUARD_PATH="/path/to/allenai-wildguard"

# Optional: custom output directory
# export PROSAFEPRUNE_ARTIFACTS_DIR="/path/to/artifacts"

# Optional: collect attention projections only; prune with --modules att
# export PROSAFEPRUNE_TARGET_MODULES="q_proj,k_proj,v_proj,o_proj"

# Optional: for chat templates that support enable_thinking
# export PROSAFEPRUNE_DISABLE_THINKING="true"
