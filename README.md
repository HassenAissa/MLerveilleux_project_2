# GPTime: Time dependent MoE implementation




This project explores time dependent LLMs by implementing a time dependent Mixture of Experts on GPT2. <br>
`main.py` allows to train GPT2, standard gating MoE and Time Dependent MoE in order to be able to compare their performance. 
In order to chose which models to run, you need to specify the `moe_routings` as follows:
| Model type | moe_routing    |
| :---:   | :---: |
| GPT2 | None   | 
| MoE    | "standard_gating" |
| Time dependent    | "masked" |

The training of the Time Dependent on Fineweb10BT dataset with 8M sequences which is approximately 5.5B tokens takes about 20 hours on an A100 GPU and we get the following training curve.
We use a batch size of 64 with gradient accumulation of 2 and a cosine scheduler with a maximum learning rate of 1e-3 that is reduced to 1e-4. (CUDA is necessary for the training)<br>


![Time dependent training curve](assets/time_dependent.png)

The final perplexities we get are the following
| Model type | Perplexity    |
| :---:   | :---: |
| GPT2 | 41.86  | 
| MoE    | 32.09 |
| Time dependent    | 33.11 |

For more details, please check the wandb run here: : https://wandb.ai/hassen-aissa1/time_dependant_llm/reports/GPTime-Time-Dependent-LLM-training--VmlldzoxMDY0ODE3Mg<br>
If you want to implement other types of masking, this can be done in `moe.py` as we provide a general class `MaskedMoE` that is used to implement `TimeDependentMoE` and could be used for other types of masking (example: Age masking)
# Interact with the models

## Installation
You need to have python 3.10.0 installed in order to run the project.
You can install the requirements with the following command:
```
pip install -r requirements.txt 
```
WARNING: Pip might raise an error if your version of python is not python 3.10.0!
## Query the Models
In order to have a hand on experience on our models, we provide the script `interact.py` which allows you to ask questions and requests to our model and compare their behaviors directly. 

The script will prompt the user for the following inputs:
- Path to the saved weights of the model
- The type of the model: GPT-2 Baseline or the MoE Baseline or the Time Dependant MoE
- The Temperature for temperature sampling

After this the user will be prompted with "~>" and can type its request there.

The file prompt_frame.txt contains a series of question and answer in the following format:
```
Q: Who wrote Romeo and Juliet?  
A: Romeo and Juliet was written by William Shakespeare.

Q: Who painted the Mona Lisa?  
A: The Mona Lisa was painted by Leonardo da Vinci.

Q: What is the name of the fictional detective created by Arthur Conan Doyle?  
A: The fictional detective is Sherlock Holmes.
```
The content of this file will be concatenated with the users query. This helps to give to the model a context and an idea of the task it has to solve.
Therefore, if the user wants to have the best performances it should query the model using this format: 

```Q: [Your question here]```

---
***NOTE 1:***
If your model is the Time Dependent MoE, the script will also ask for the year of inference to set.<br>
---
Example of questions : 
```
[User Input]: Q: What are the most common
first symptoms of COVID-19? <br>

[Masked MoE - 2020 selected]: A: The
first symptoms of COVID-19 are: fever,
fever, diarrhea, fever, episodes of
fever, and shortness of breath.
```


---
***NOTE 2:***
As the training is very long, we provide pretrained models on this drive:
---
https://drive.google.com/drive/u/1/folders/1ODVqddfcViq9rMAOkrkZNvce29Tn0h8q

# Refences:
The `aux_losses.py`, `moe.py` and `gpt.py` were originally provided from https://github.com/epfml/llm-baselines and then impreoved for Time Depenndent implementation and better performance during training.


