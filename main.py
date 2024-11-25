import numpy as np
import datasets
from gpt import GPTBase
from moe import TimeDependantMoE2
import torch

with open("data/dataset/fineweb/train.bin", "rb") as f:
    train_data = np.frombuffer(f.read(), dtype=np.uint16)

with open("data/dataset/fineweb/test.bin", "rb") as f:
    test_data = np.frombuffer(f.read(), dtype=np.uint16)

train_data = datasets.Dataset.from_dict({"tokens": train_data})
test_data = datasets.Dataset.from_dict({"tokens": test_data})
config = {
    "moe_num_experts": 4,
    "moe_softmax_order": "softmax_topk",
    "batch_size": 4,
    "n_embd": 256,
}
mlp = GPTBase
moe = TimeDependantMoE2(config, mlp)

optimizer = torch.optim.Adam(moe.parameters(), lr=1e-4)
criterion = torch.nn.MSELoss()   

for epoch in range(1):  # Number of epochs
    for batch in train_data:
        inputs = torch.tensor(batch["tokens"], dtype=torch.long)
        labels = torch.tensor(batch["tokens"], dtype=torch.long)

        optimizer.zero_grad()
        outputs = moe(inputs)
        loss = criterion(outputs.view(-1, outputs.size(-1)), labels.view(-1))
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch + 1}, Loss: {loss.item()}")

