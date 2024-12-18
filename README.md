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


TODOOO:add plots

The final perplexities we get are the following
| Model type | Perplexity    |
| :---:   | :---: |
| GPT2 | 41.86  | 
| MoE    |  |
| Time dependent    | 33.11 |

For more details, please check the wandb run here: : https://wandb.ai/hassen-aissa1/time_dependant_llm/reports/GPTime-Time-Dependent-LLM-training--VmlldzoxMDY0ODE3Mg<br>
If you want to implement other types of masking, this can be done in `moe.py` as we provide a general class `MaskedMoE` that is used to implement `TimeDependentMoE` and could be used for other types of masking (example: Age masking)
# Interact with the models
In order to have a hand on experience on our models, we provide the script `interact.py` which allows you to ask questions and requests to our model and compare their behaviors directly. 
The script wil ask you to set provide the path to the model you want to test, asks about the type of the model, the temperature you want to set and finally the request you want to send to the model. 
If your model is a the Time Dependent, the script will also ask for the year of inference you want to set.<br>
Example question: <br>
[User Input]: Q: What are the most common
first symptoms of COVID-19? <br>
[Masked MoE - 2020 selected]: A: The
first symptoms of COVID-19 are: fever,
fever, diarrhea, fever, episodes of
fever, and shortness of breath.



# Refences:
The `aux_losses.py`, `moe.py` and `gpt.py` were originally provided from https://github.com/epfml/llm-baselines and then impreoved for Time Depenndent implementation and better performance during training.


