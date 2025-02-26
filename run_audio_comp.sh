#!/bin/bash

# Load environment variables from .env
export $(xargs < .env)

# Ensure logs directory exists
mkdir -p logs

# Function to run the script and log output
run_script_clotho() {
    cap_num=$1
    method=$2
    log_file="logs/audiogen_audio_comparison_run_cap_${cap_num}_method_${method}_$(date '+%Y-%m-%d_%H-%M-%S').log"

    echo "Running script with cap_num=$cap_num and method=$method"
    echo "Logging output to $log_file"

    # For EzAudio
    python random_selection_based_preprocessing/parallel_clotho_compare_audios.py --cap_num "$cap_num" --method "$method" >> "$log_file" 2>&1

    echo "Completed script with cap_num=$cap_num and method=$method"
}

run_script_ez() {
    cap_num=$1
    method=$2
    log_file="logs/ezaudio_audio_comparison_run_cap_${cap_num}_method_${method}_$(date '+%Y-%m-%d_%H-%M-%S').log"

    echo "Running script with cap_num=$cap_num and method=$method"
    echo "Logging output to $log_file"

    # For EzAudio
    python random_selection_based_preprocessing/parallel_ez_compare_audios.py --cap_num "$cap_num" --method "$method" >> "$log_file" 2>&1

    echo "Completed script with cap_num=$cap_num and method=$method"
}
# Execute commands sequentially

run_script_clotho 1 dtw
sleep 30
run_script_clotho 1 wcc
sleep 30

run_script_clotho 2 dtw
sleep 30
run_script_clotho 2 wcc
sleep 30

run_script_clotho 3 dtw
sleep 30
run_script_clotho 3 wcc
sleep 30

run_script_clotho 4 dtw
sleep 30
run_script_clotho 4 wcc
sleep 30

run_script_clotho 5 dtw
sleep 30
run_script_clotho 5 wcc
sleep 30

run_script_clotho 1 model
sleep 30
run_script_clotho 2 model
sleep 30
run_script_clotho 3 model
sleep 30
run_script_clotho 4 model
sleep 30
run_script_clotho 5 model


run_script_ez 1 dtw
sleep 30
run_script_ez 1 wcc
sleep 30

run_script_ez 2 dtw
sleep 30
run_script_ez 2 wcc
sleep 30

run_script_ez 3 dtw
sleep 30
run_script_ez 3 wcc
sleep 30

run_script_ez 4 dtw
sleep 30
run_script_ez 4 wcc
sleep 30

run_script_ez 5 dtw
sleep 30
run_script_ez 5 wcc
sleep 30

run_script_ez 1 model
sleep 30
run_script_ez 2 model
sleep 30
run_script_ez 3 model
sleep 30
run_script_ez 4 model
sleep 30
run_script_ez 5 model

echo "All processes completed."
