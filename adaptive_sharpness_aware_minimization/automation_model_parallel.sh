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
run_model_parallel() {
    local gpu_idx=0
    local num_gpus=${#DEVICES[@]}

    for DATASET in "${DATASETS[@]}"; do
        DEVICE="${DEVICES[$gpu_idx]}"
        
        # We wrap the model loop in a background block ( ) &
        # This makes the DATASET run on a specific GPU, 
        # but the MODELS within that dataset run one after another.
        (
            for MODEL in "${MODELS[@]}"; do
                echo "GPU ${DEVICE} | Starting: ${DATASET} - ${MODEL}"
                
                python ASAM.py \
                    --optimizer "$OPTIMIZER" \
                    --epochs "$EPOCHS" \
                    --runs "$RUNS" \
                    --dataset "$DATASET" \
                    --model "$MODEL" \
                    --gpu "$DEVICE"
                
                echo "GPU ${DEVICE} | Finished: ${DATASET} - ${MODEL}"
            done
        ) &

        ((gpu_idx++))

        # Wait if all 8 GPUs are currently busy with a dataset
        if [ $gpu_idx -eq $num_gpus ]; then
            echo ">>> All 8 GPUs are busy. Waiting for this batch of datasets to finish..."
            wait
            gpu_idx=0
        fi
    done
    wait 
}

# ======================= EXECUTION FLOW =======================
echo "Activating: hms" # Adjust environment name if different
conda activate hms

run_model_parallel

echo "======================================================="
echo "All HBF experiments completed."
echo "======================================================="