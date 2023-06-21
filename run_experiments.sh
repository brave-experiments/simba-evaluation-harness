#!/bin/bash

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/x86_64-linux-gnu/

for MODEL in "OpenAssistant/falcon-40b-sft-mix-1226" "tiiuae/falcon-40b-instruct" ;do
    for TASK in "simba"; do

        MODEL_REPO=$(echo $MODEL | cut -d '/' -f 1)
        MODEL_NAME=$(echo $MODEL | cut -d '/' -f 2)

        echo "Evaluating ${MODEL_REPO}/${MODEL_NAME} on ${TASK}"
        python main.py \
            --model hf-causal-experimental \
            --model_args pretrained=${MODEL},trust_remote_code=True,dtype="bfloat16",use_accelerate="True" \
            --tasks ${TASK} \
            --device cuda \
            --output_path results/${MODEL_NAME}-${TASK} |& tee ${MODEL_NAME}-${TASK}.log
    done
done

for MODEL in "OpenAssistant/oasst-sft-4-pythia-12b-epoch-3.5" "databricks/dolly-v2-12b" ;do
    for TASK in "simba"; do

        MODEL_REPO=$(echo $MODEL | cut -d '/' -f 1)
        MODEL_NAME=$(echo $MODEL | cut -d '/' -f 2)

        echo "Evaluating ${MODEL_REPO}/${MODEL_NAME} on ${TASK}"
        python main.py \
            --model hf-causal-experimental \
            --model_args pretrained=${MODEL},trust_remote_code=True \
            --tasks ${TASK} \
            --device cuda:0 \
            --output_path results/${MODEL_NAME}-${TASK} |& tee ${MODEL_NAME}-${TASK}.log
    done
done
