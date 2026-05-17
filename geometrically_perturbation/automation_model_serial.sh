#!/bin/bash

# ======================= CONDA INITIALIZATION =======================
if [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/anaconda3/etc/profile.d/conda.sh"
elif [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
fi

# ======================= GLOBAL PARAMETERS =======================
# SGD, Momentum, NAG, AdaGrad, RMSprop, AdaDelta, Adam, AdaMax, NAdam, AdamW, RAdam, ASGD, Rprop
OPTIMIZER="All"
EPOCHS=1000
RUNS=5
DEVICES=(0 1 2 3 4 5 6 7)

# Ordered list of datasets (as per your request)
DATASETS=("boston" "california" "diabetes" "concrete" "energy" "kin8nm" "naval" "protein"  "year_prediction_msd" 
          "ct_slices" "bike_sharing" "wine_quality"  "appliances_energy" "load_appliances_energy" "power_plant")

# Models to iterate through for each dataset
MODELS=("linear" "mlp" "lstm" "gru")

# ======================= EXECUTION FUNCTION =======================
run_model_serial() {
    local gpu_idx=0
    local num_gpus=${#DEVICES[@]}

    for DATASET in "${DATASETS[@]}"; do
        DEVICE="${DEVICES[$gpu_idx]}"
        
        # We use ( ... ) & to run the DATASET's model-sequence in the background.
        # This allows the script to move to the next GPU immediately.
        (
            for MODEL in "${MODELS[@]}"; do
                echo "[GPU ${DEVICE}] STARTING: ${DATASET} | MODEL: ${MODEL}"
                
                # No '&' here: script waits for this model to finish 
                # before starting the next model in the MODELS array.
                python GAP.py \
                    --optimizer "$OPTIMIZER" \
                    --epochs "$EPOCHS" \
                    --runs "$RUNS" \
                    --dataset "$DATASET" \
                    --model "$MODEL" \
                    --gpu "$DEVICE"
                
                echo "[GPU ${DEVICE}] FINISHED: ${DATASET} | MODEL: ${MODEL}"
            done
        ) &

        # Increment GPU index
        ((gpu_idx++))

        # If all 8 GPUs have been assigned a dataset, wait for them to finish
        if [ $gpu_idx -eq $num_gpus ]; then
            echo ">>> BATCH FULL: All 8 GPUs are busy. Waiting for datasets to finish..."
            wait
            gpu_idx=0
        fi
    done
    
    wait # Final wait for the last batch of datasets
}
# ======================= EXECUTION FLOW =======================
echo "Activating: hms" # Adjust environment name if different
conda activate hms

run_model_serial

echo "======================================================="
echo "All HBF experiments completed."
echo "======================================================="