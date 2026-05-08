#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd "$script_dir/.." && pwd)
dataset_rel="test_training_set/LegoGears_v2"
dataset_dir="$repo_root/$dataset_rel"

train_list="$dataset_dir/LegoGears_train.local.txt"
valid_list="$dataset_dir/LegoGears_valid.local.txt"
data_file="$dataset_dir/LegoGears.local.data"
backup_dir="$dataset_dir/backup"

if [[ ! -d "$dataset_dir" ]]; then
	echo "Missing dataset directory: $dataset_dir" >&2
	exit 1
fi

cd "$repo_root"
mkdir -p "$backup_dir"

find "$dataset_rel/set_01" "$dataset_rel/set_02_empty" -maxdepth 1 -type f -name '*.jpg' -print | sort > "$train_list"
find "$dataset_rel/set_03" -maxdepth 1 -type f -name '*.jpg' -print | sort > "$valid_list"

{
	echo "classes = 5"
	echo "train = $dataset_rel/LegoGears_train.local.txt"
	echo "valid = $dataset_rel/LegoGears_valid.local.txt"
	echo "names = $dataset_rel/LegoGears.names"
	echo "backup = $dataset_rel/backup"
} > "$data_file"

printf 'Wrote %s (%s images)\n' "$train_list" "$(wc -l < "$train_list")"
printf 'Wrote %s (%s images)\n' "$valid_list" "$(wc -l < "$valid_list")"
printf 'Wrote %s\n' "$data_file"
