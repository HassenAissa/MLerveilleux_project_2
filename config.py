import torch

class Config:

    def __init__(self, **kwargs):
        self.moe_num_experts = kwargs["moe_num_experts"]
        self.moe_softmax_order = kwargs["moe_softmax_order"]
        self.batch_size = kwargs["batch_size"]
        self.n_embd = kwargs["n_embd"]
        self.dropout = 0.0
        self.n_head = 4
        self.n_layer = 4
        self.sequence_length = 1024
        self.dtype = torch.bfloat16
        self.bias = False
        self.vocab_size = 50304
        self.moe = True
        self.moe_routing = "masked"
        self.moe_num_experts_per_tok = 2
        self.moe_router_loss = "load_balancing_z_loss"
        self.moe_aux_loss_factor = 0.1
        self.moe_z_loss_factor = 1
        self.routing = None
        self.mlp_dim_exp_factor = 4
        