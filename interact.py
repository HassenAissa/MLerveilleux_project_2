from gpt import GPTBase
import torch
from config import Config
import tiktoken

MOE_ROUTINGS = [ None, "standard_gating", "masked"]
MAX_DATE = 2024
MIN_DATE = 2013
STOP_QUERY = "STOP"
INPUT_SYMBOL = "~>"

def choices(clist, text="Please choose one of the following options: "):
    print(text)
    for i, c in enumerate(clist):
        print(str(i) + ") " + str(c))
    while c== None or c.isnumeric() == False or int(c) >= len(clist):
        c = input(">")
    return clist[int(c)]

def run_query(query,date, model, tokenizer, config, max_answer_len = 100, device="cuda"):
    ctx_window_size = config.sequence_length
    query_tokens = tokenizer.encode_ordinary(query)
    query_len = len(query_tokens)
    query_tokens = tokenizer.encode(" ")*(ctx_window_size - query_len) + query_tokens
    query_tokens = torch.tensor(query_tokens, dtype=torch.int64)
    query_tokens = query_tokens.reshape((1,-1))

    answer_count = 0
    new_token = None
    print("")
    while new_token == None or (answer_count < max_answer_len and new_token.item() != tokenizer.eot_token):
        query_tokens = query_tokens.to(device)
        output = model(query_tokens, date, get_logits=True, moe=config.moe)
        logits = output["logits"].reshape((-1,))
        new_token = torch.argmax(logits).reshape((1,1))
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
    print("Loading model ...")
    model = GPTBase(config)
    state_dict = torch.load(model_path)["model"]
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    print("done")


    max_len = config.sequence_length
    print("Loading tokenizer...")
    tokenizer = tiktoken.get_encoding("gpt2")
    print("done")
    print("*** Mixture of Experts Chat ***")
    print("routing type : ", routing)
    print("The model joined the chat !")

    query = input(INPUT_SYMBOL)
    while query != STOP_QUERY:
        run_query(query, None, model, tokenizer, config)
        query = input(INPUT_SYMBOL)



