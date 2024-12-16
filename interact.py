from gpt import GPTBase
import torch
from config import Config
import tiktoken
import torch.nn.functional as F

MOE_ROUTINGS = [ None, "standard_gating", "masked"]
MAX_DATE = 2024
MIN_DATE = 2013
STOP_QUERY = "STOP"
INPUT_SYMBOL = "~>"
#PROMPT_FRAME = "You are an AI language model based on the GPT-2 architecture. Your purpose is to generate human-like text based on the input you receive. Your task is to understand the context of the conversation or query and respond coherently and appropriately. You have knowledge on a wide range of topics, but your responses should be based on the data you were trained on, which has a cutoff date of [insert date]. When responding, ensure your answers are clear, relevant, and informative, adhering to the tone of the input provided."
PROMPT_FRAME_PATH="prompt_frame.txt"

def temperature_sampling(logits, temperature=1.0):
    logits = logits / temperature
    probs = F.softmax(logits, dim=-1)
    sampled_token = torch.multinomial(probs, 1)
    return sampled_token

def choices(clist, text="Please choose one of the following options: "):
    print(text)
    for i, c in enumerate(clist):
        print(str(i) + ") " + str(c))
    while c== None or c.isnumeric() == False or int(c) >= len(clist):
        c = input(">")
    return clist[int(c)]
def get_prompt_frame(path):
    with open(path, "r") as file:
        text = file.read()
    return text

def run_query(query,date, model, tokenizer, config, prompt_frame,max_answer_len = 100, device="cuda", temperature=1.0):
    ctx_window_size = config.sequence_length
    query_tokens = tokenizer.encode_ordinary(query)
    prompt_frame_pad = tokenizer.encode_ordinary(prompt_frame)
    query_len = len(query_tokens)
    query_tokens = tokenizer.encode(" ")*(ctx_window_size - query_len - len(prompt_frame_pad)) + prompt_frame_pad + query_tokens
    #print("len frame",len(prompt_frame_pad))
    query_tokens = query_tokens[-ctx_window_size:]
    query_tokens = torch.tensor(query_tokens, dtype=torch.int64)
    query_tokens = query_tokens.reshape((1,-1))

    answer_count = 0
    new_token = None
    print("")
    while new_token == None or (answer_count < max_answer_len and new_token.item() != tokenizer.eot_token):
        query_tokens = query_tokens.to(device)
        output = model(query_tokens, date, get_logits=True, moe=config.moe)
        logits = output["logits"].reshape((-1,))
        new_token = temperature_sampling(logits, temperature=temperature).reshape((1,1))
        #print(query_tokens.shape)
        #print(new_token.shape)
        query_tokens = torch.cat([query_tokens[:,1:], new_token], dim=-1)
        answer_count += 1
        print(tokenizer.decode([new_token]), end="", flush=True)
    print("")










if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("*** Interact with the trained models programm ***")
    print("Copyright Hassen, Lysandre and Pierre")

    model_path = input("Enter your model path here : ")
    routing = choices(MOE_ROUTINGS)
    config = Config(**{
        "moe_num_experts": (MAX_DATE - MIN_DATE + 1) // 2,
        "moe_softmax_order": "softmax_topk",
        "batch_size": 64,
        "n_embd": 768,
        # "date_list": date_list,
        "moe_routing": routing,
        "moe": routing is not None
    })
    temperature = float(input("Enter Temperature of model here : "))
    print("Loading model ...")
    model = GPTBase(config)
    state_dict = torch.load(model_path)["model"]
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    print("done")


    max_len = config.sequence_length
    prompt_frame = get_prompt_frame(PROMPT_FRAME_PATH)
    print("Loading tokenizer...")
    tokenizer = tiktoken.get_encoding("gpt2")
    print("done")
    print("*** Mixture of Experts Chat ***")
    print("routing type : ", routing)
    print("The model joined the chat !")

    query = input(INPUT_SYMBOL)
    while query != STOP_QUERY:
        run_query(query, None, model, tokenizer, config, prompt_frame, temperature=temperature)
        query = input(INPUT_SYMBOL)



