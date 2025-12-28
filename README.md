# Budget-Constrained-RLRS
This repository contains the code to reproduce the experiments of the paper "Time-Constrained Recommendations: Reinforcement Learning Strategies for E-Commerce".

## Installation

To create a virtual environment before installing, you can use the command:
```bash
conda create -n rl_env python=3.11
conda activate rl_env
pip install -r requirements.txt
```

## Dataset
Download the datasets using the instructions given in this [link](https://github.com/rank2rec/rerank#readme) and copy the dataset files to the `data` directory. 

## Experiments

### Training
For running the experiments, navigate to the src directory.
```bash
cd src
```

To run the training and evaluation of baseline Personalized Re-Ranking model on Alibaba's Re-Ranking dataset, use the following command:
```bash
python main.py
```
Make sure to set the correct hyper-parameters in the `Config` class in `main.py` file.

### Simulation
Reinforcement Learning based simulations (SARSA and Q-Learning) can be done by using the following command:
```bash
python simulation.py
```
Make sure to set the correct hyper-parameters and initial parameter ranges for the experiments in `simulation.py` file.

## Citation

If you use this codebase in academic work, please cite:

```
@misc{Budget-Constrained-RLRS,
  title   = {Budget-Constrained-RLRS},
  author  = {Anonymous},
  year    = {2025},
  howpublished = {\url{https://github.com/anonymous/Budget-Constrained-RLRS}}
}
```

---

## License

Read the [LICENSE](LICENSE) file.
