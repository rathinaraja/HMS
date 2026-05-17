#!/bin/bash

# Define the list of script paths
SCRIPTS=(
    "/home/rajaj/HMS/Geometrically_Perturbation/automation_model_parallel.sh"
    "/home/rajaj/HMS/Harmonic_Mean_Scalar/automation_model_parallel.sh"
    "/home/rajaj/HMS/Heavy_Ball_Friction/automation_model_parallel.sh"
    "/home/rajaj/HMS/Lookahead_Optimizer/automation_model_parallel.sh"
    "/home/rajaj/HMS/Slow_Momentum/automation_model_parallel.sh"
    "/home/rajaj/HMS/Stochastic_Polyak_Step_Size/automation_model_parallel.sh"
)

echo "Starting sequential execution of ${#SCRIPTS[@]} scripts..."

for SCRIPT in "${SCRIPTS[@]}"; do
    if [ -f "$SCRIPT" ]; then
        echo "========================================================"
        echo "EXECUTING: $SCRIPT"
        echo "START TIME: $(date)"
        echo "========================================================"
        
        # Execute the script and wait for it to finish
        # Using 'bash' explicitly to ensure it runs even if +x permission is missing
        bash "$SCRIPT"
        
        # Optional: Check if the script failed
        if [ $? -ne 0 ]; then
            echo "ERROR: Script $SCRIPT failed with exit code $?. Continuing to next..."
        fi
        
        echo "FINISHED: $SCRIPT at $(date)"
    else
        echo "SKIP: File not found -> $SCRIPT"
    fi
done

echo "========================================================"
echo "All tasks completed."
echo "========================================================"