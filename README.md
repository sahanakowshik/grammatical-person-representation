# Grammatical Person Representation in Large Language Models

In this work, we study how LLMs internally represent grammatical person (the distinction between “I” and “you”) and how this representation relates to the personas they adopt during generation.

## Overview
Many interpretability analyses of language models struggle to differentiate between user- and model-oriented perspectives because natural language often switches between first-person (“I”) and second-person (“you”). Our project investigates whether large language models encode grammatical person linearly in their latent representations and how this encoding influences output behavior when we intervene along these directions.

Our approach uses contrastive steering vectors derived from first- and second-person sentence pairs. We demonstrate that:

- Grammatical person corresponds to a consistent linear direction in the model representation space.

- Steering along this direction induces distinct persona shifts in model generation.

These effects are most pronounced in instruction-tuned models compared to base models.

## Repository Structure

```
├── data/                 # Data preperation
├── eval/                 # LLM-as-a-judge evaluation
├── human_eval/           # Human annotation code and plots
├── interp/               # Steering & interpretability code
├── README.md     
└── environment.yml       # Conda environment dependencies

```

## Setup

### Clone the Repository

```bash
git clone https://github.com/sahanakowshik/grammatical-person-representation.git
cd grammatical-person-representation
```

### Environment Setup
```bash
conda env create -f environment.yml
conda activate grammatical-person
```

## Dataset creation

Update the `data/seed_examples.json` and the `data/config.yml` files

```bash
cd data/

bash generate_data.sh
or
qsub generate_data.sh (submit a btach job)
```

## Generate Steering vectors

Update `interp/get_steering_vectors/config.yml`

```bash
cd interp/get_steering_vectors/

bash main.sh
or
qsub main.sh (submit a btach job)
```

## Steer the vectors
### Layer steering
Update the steering coefficient, model id and folder name in `interp/layer_steer.py`

```bash
cd interp/
python layer_steer.py
```

### Coefficient sensitivity
Update the layer number, model id and folder name in `interp/coeff_sensitivity_analysis.py`

```bash
cd interp/
python coeff_sensitivity_analysis.py
```

## LLM-as-a-judge experiment
Update `eval/config.yml`

```bash
cd eval/
python llm_as_a_judge.py
```
