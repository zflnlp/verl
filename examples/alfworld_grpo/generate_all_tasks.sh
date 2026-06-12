#!/usr/bin/env bash
# Generate real ALFWorld data for all 6 task types
#
# Usage:
#   bash examples/alfworld_grpo/generate_all_tasks.sh
#
# This script generates training data for all 6 ALFWorld task types
# and combines them into a single dataset.

set -xeuo pipefail

########################### user-adjustable ###########################
# ALFWorld data directory
ALFWORLD_DATA_DIR=${ALFWORLD_DATA_DIR:-/workspace/data/alf_data}

# Output directory
OUTPUT_DIR=${OUTPUT_DIR:-/workspace/data/alfworld_real_all}

# Number of games per task type
NUM_GAMES=${NUM_GAMES:-100}

# Random seed
SEED=${SEED:-42}
########################### end user-adjustable ###########################

# Task types
TASK_TYPES=(
    "pick_and_place"
    "pick_clean_then_place"
    "pick_heat_then_place"
    "pick_cool_then_place"
    "look_at_obj_in_light"
    "pick_two_obj"
)

echo "=========================================="
echo "ALFWorld Data Generation (All Task Types)"
echo "=========================================="
echo "ALFWorld data: ${ALFWORLD_DATA_DIR}"
echo "Output dir: ${OUTPUT_DIR}"
echo "Games per task: ${NUM_GAMES}"
echo "Task types: ${TASK_TYPES[*]}"
echo "=========================================="

# Create output directory
mkdir -p "${OUTPUT_DIR}"

# Generate data for each task type
for task_type in "${TASK_TYPES[@]}"; do
    echo ""
    echo "------------------------------------------"
    echo "Generating: ${task_type}"
    echo "------------------------------------------"

    python examples/alfworld_grpo/data_preprocess.py \
        --local_save_dir "${OUTPUT_DIR}/${task_type}" \
        --use_real_env \
        --task_type "${task_type}" \
        --num_games "${NUM_GAMES}" \
        --seed "${SEED}" \
        --game_files_dir "${ALFWORLD_DATA_DIR}"
done

# Combine all task types into a single dataset
echo ""
echo "=========================================="
echo "Combining all task types..."
echo "=========================================="

python3 << 'EOF'
import pandas as pd
import os
import json

output_dir = os.environ.get("OUTPUT_DIR", "/workspace/data/alfworld_real_all")
task_types = [
    "pick_and_place",
    "pick_clean_then_place",
    "pick_heat_then_place",
    "pick_cool_then_place",
    "look_at_obj_in_light",
    "pick_two_obj",
]

all_train = []
all_test = []
total_games = 0

for task_type in task_types:
    task_dir = os.path.join(output_dir, task_type)
    train_path = os.path.join(task_dir, "train.parquet")
    test_path = os.path.join(task_dir, "test.parquet")

    if os.path.exists(train_path):
        train_df = pd.read_parquet(train_path)
        all_train.append(train_df)
        print(f"  {task_type}: {len(train_df)} train tasks")

    if os.path.exists(test_path):
        test_df = pd.read_parquet(test_path)
        all_test.append(test_df)
        print(f"  {task_type}: {len(test_df)} test tasks")

# Combine all data
if all_train:
    combined_train = pd.concat(all_train, ignore_index=True)
    combined_train.to_parquet(os.path.join(output_dir, "train.parquet"), index=False)
    print(f"\nTotal train tasks: {len(combined_train)}")

if all_test:
    combined_test = pd.concat(all_test, ignore_index=True)
    combined_test.to_parquet(os.path.join(output_dir, "test.parquet"), index=False)
    print(f"Total test tasks: {len(combined_test)}")

# Save metadata
metadata = {
    "mode": "real",
    "task_types": task_types,
    "num_games_per_task": int(os.environ.get("NUM_GAMES", "100")),
    "seed": int(os.environ.get("SEED", "42")),
    "train_size": len(combined_train) if all_train else 0,
    "test_size": len(combined_test) if all_test else 0,
}

with open(os.path.join(output_dir, "task_metadata.json"), "w") as f:
    json.dump(metadata, f, indent=2)

print(f"\nSaved to: {output_dir}")
print(f"  - train.parquet")
print(f"  - test.parquet")
print(f"  - task_metadata.json")
EOF

echo ""
echo "=========================================="
echo "Done! All task types generated."
echo "=========================================="
