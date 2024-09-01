import argparse
import abc
import random
from itertools import permutations
from typing import Set, List, Tuple
import os
import json
import numpy as np
from einops import rearrange, repeat
import torch
from torch.utils.data import IterableDataset
from torch import nn, Tensor

# Abstract base class for datasets
class AbstractDataset(abc.ABC):
    def __init__(self, group_elements1: Set, group_elements2: Set, frac_train: float):
        self.frac_train = frac_train
        self.group_elements1 = group_elements1
        self.group_elements2 = group_elements2
        self.ordered_group_elements1 = list(self.group_elements1)
        self.ordered_group_elements2 = list(self.group_elements2)
        self.idx2vocab = ["o", "="] + list(group_elements1.union(group_elements2))
        self.vocab2idx = {vocab: idx for idx, vocab in enumerate(self.idx2vocab)}
        self.n_vocab = len(self.idx2vocab)
        self.n_out = len(group_elements1.union(group_elements2))
        
        # Shuffle and split the dataset
        idxs = list(range(len(self.group_elements1) * len(self.group_elements2)))
        random.shuffle(idxs)
        split_point = int(len(idxs) * frac_train)
        self.train_pairs, self.val_pairs = idxs[:split_point], idxs[split_point:]

    @abc.abstractmethod
    def fetch_output(self, a, b):
        """Abstract method to compute the output of the operation"""
        pass

    def encode(self, sequence: List) -> List[int]:
        """Convert a sequence of symbols to their corresponding indices"""
        return [self.vocab2idx[item] for item in sequence]

    def decode(self, sequence: List[int]) -> List:
        """Convert a sequence of indices back to their corresponding symbols"""
        return [self.idx2vocab[item] for item in sequence]

    def form_equation(self, a, b, c) -> List:
        """Form an equation string from operands and result"""
        return [a, "o", b, "=", c]

    def fetch_example(self, idx: int) -> Tuple[List[int], int, List]:
        """Fetch a single example from the dataset"""
        a = self.ordered_group_elements1[idx // len(self.group_elements2)]
        b = self.ordered_group_elements2[idx % len(self.group_elements2)]
        c = self.fetch_output(a, b)
        equation = self.form_equation(a, b, c)
        return self.encode(equation[:-1]), (self.vocab2idx[c] - 2), equation

    def fetch_train_example(self) -> Tuple[List[int], int, List]:
        """Fetch a random training example"""
        idx = random.choice(self.train_pairs)
        return self.fetch_example(idx)

    def fetch_val_example(self) -> Tuple[List[int], int, List]:
        """Fetch a random validation example"""
        idx = random.choice(self.val_pairs)
        return self.fetch_example(idx)

    def reverse_operands(self, a, b) -> Tuple:
        """Reverse the order of operands"""
        return b, a

# Modular sum dataset
class ModSumDataset(AbstractDataset):
    def __init__(self, p: int, frac_train: float):
        super(ModSumDataset, self).__init__(set(range(p)), set(range(p)), frac_train)
        self.p = p

    def fetch_output(self, a: int, b: int) -> int:
        """Compute (a + b) mod p"""
        return (a + b) % self.p

    def fetch_example(self, idx: int) -> Tuple[List[int], int, List]:
        """Fetch an example with possible operand reversal and negation"""
        a = self.ordered_group_elements1[idx // len(self.group_elements2)]
        b = self.ordered_group_elements2[idx % len(self.group_elements2)]
        if random.random() < 0.2:
            a, b = self.reverse_operands(a, b)
        if random.random() < 0.2:
            a, b = self.negate_operands(a, b)
        c = self.fetch_output(a, b)
        equation = self.form_equation(a, b, c)
        return self.encode(equation[:-1]), (self.vocab2idx[c] - 2), equation

    def negate_operands(self, a: int, b: int) -> Tuple[int, int]:
        """Negate both operands in modular arithmetic"""
        return (self.p - a) % self.p, (self.p - b) % self.p

# Modular subtraction dataset
class ModSubtractDataset(AbstractDataset):
    def __init__(self, p: int, frac_train: float):
        super(ModSubtractDataset, self).__init__(set(range(p)), set(range(p)), frac_train)
        self.p = p

    def fetch_output(self, a: int, b: int) -> int:
        """Compute (a - b) mod p"""
        return (a - b) % self.p

    def fetch_example(self, idx: int) -> Tuple[List[int], int, List]:
        """Fetch an example with possible operand reversal and negation"""
        a = self.ordered_group_elements1[idx // len(self.group_elements2)]
        b = self.ordered_group_elements2[idx % len(self.group_elements2)]
        rand = random.random()
        if rand < 0.2:
            a, b = self.reverse_operands(a, b)
        elif rand < 0.4:
            a, b = self.negate_operands(a, b)
        c = self.fetch_output(a, b)
        equation = self.form_equation(a, b, c)
        return self.encode(equation[:-1]), (self.vocab2idx[c] - 2), equation

    def negate_operands(self, a: int, b: int) -> Tuple[int, int]:
        """Negate both operands in modular arithmetic"""
        return (self.p - a) % self.p, (self.p - b) % self.p

# Modular division dataset
class ModDivisionDataset(AbstractDataset):
    def __init__(self, p: int, frac_train: float):
        super(ModDivisionDataset, self).__init__(set(range(p)), set(range(1, p)), frac_train)
        self.p = p

    def fetch_output(self, a: int, b: int) -> int:
        """Compute (a / b) mod p using Fermat's Little Theorem"""
        return (a * pow(b, self.p - 2, self.p)) % self.p

    def fetch_example(self, idx: int) -> Tuple[List[int], int, List]:
        """Fetch an example with possible dividend negation"""
        a = self.ordered_group_elements1[idx // len(self.group_elements2)]
        b = self.ordered_group_elements2[idx % len(self.group_elements2)]
        if random.random() < 0.2:
            a, b = self.negate_operands(a, b)
        c = self.fetch_output(a, b)
        equation = self.form_equation(a, b, c)
        return self.encode(equation[:-1]), (self.vocab2idx[c] - 2), equation

    def negate_operands(self, a: int, b: int) -> Tuple[int, int]:
        """Negate only the dividend in modular arithmetic"""
        return (self.p - a) % self.p, b

# Permutation group dataset
class PermutationGroup(AbstractDataset):
    def __init__(self, k: int, frac_train: float):
        perms = set(map(tuple, permutations(list(range(k)))))
        super(PermutationGroup, self).__init__(perms, perms, frac_train)
        self.k = k

    def fetch_output(self, a: Tuple[int, ...], b: Tuple[int, ...]) -> Tuple[int, ...]:
        """Compute the composition of two permutations"""
        return tuple([a[b[i]] for i in range(len(b))])

# Iterable dataset for PyTorch
class GroupDataset(IterableDataset):
    def __init__(self, dataset: AbstractDataset, split: str):
        super(GroupDataset, self).__init__()
        assert split in {"train", "val"}
        self.dataset = dataset
        self.split = split
        self.fetch_f = self.dataset.fetch_train_example if split == "train" else self.dataset.fetch_val_example

    def __iter__(self):
        return self

    def __next__(self):
        x, y, _ = self.fetch_f()
        return torch.tensor(x), torch.tensor(y)

# Factory function to create the appropriate dataset
def operation_mod_p_data(operation: str, p: int, frac_train: float) -> AbstractDataset:
    """
    Create a dataset for the specified operation in modulo p arithmetic
    """
    if operation == "x_plus_y":
        return ModSumDataset(p=p, frac_train=frac_train)
    elif operation == "x_minus_y":
        return ModSubtractDataset(p=p, frac_train=frac_train)
    elif operation == "x_div_y":
        return ModDivisionDataset(p=p, frac_train=frac_train)
    elif operation == "permutation":
        return PermutationGroup(k=5, frac_train=frac_train)
    else:
        raise ValueError(f"Unknown operation: {operation}")

# Function to get data loaders and dataset information
def get_data(operation: str, prime: int, training_fraction: float, batch_size: int):
    dataset = operation_mod_p_data(operation, prime, training_fraction)
    train_dataset = GroupDataset(dataset, "train")
    val_dataset = GroupDataset(dataset, "val")
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size)
    return train_loader, val_loader, train_dataset.dataset.n_vocab, train_dataset.dataset.n_out

# Transformer decoder block
class DecoderBlock(torch.nn.Module):
    def __init__(self, dim_model: int, n_heads: int):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(dim_model, n_heads)
        self.self_attn_norm = nn.LayerNorm(dim_model)
        self.ffn = nn.Sequential(
            nn.Linear(dim_model, dim_model * 4),
            nn.GELU(),
            nn.Linear(dim_model * 4, dim_model),
        )
        self.ffn_norm = nn.LayerNorm(dim_model)

    def forward(self, x: Tensor) -> Tensor:
        # Create causal attention mask
        attn_mask = torch.full((len(x), len(x)), -float("Inf"), device=x.device, dtype=x.dtype)
        attn_mask = torch.triu(attn_mask, diagonal=1)

        # Self-attention and layer normalization
        a1, _ = self.self_attn(x, x, x, attn_mask=attn_mask)
        a1 = self.self_attn_norm(x + a1)

        # Feed-forward network and layer normalization
        a2 = self.ffn(a1)
        a2 = self.ffn_norm(a1 + a2)

        return a2

# Transformer model
class Transformer(torch.nn.Module):
    def __init__(self, num_layers: int, dim_model: int, num_heads: int, vocab_size: int, output_size: int, seq_len: int):
        super().__init__()
        self.token_embeddings = nn.Embedding(vocab_size, dim_model)
        self.position_embeddings = nn.Embedding(seq_len, dim_model)
        self.model = nn.Sequential(
            *[DecoderBlock(dim_model, num_heads) for _ in range(num_layers)],
            nn.LayerNorm(dim_model),
            nn.Linear(dim_model, output_size),
        )

    def forward(self, inputs: Tensor) -> Tensor:
        batch_size, context_len = inputs.shape

        # Compute token and position embeddings
        token_embedding = self.token_embeddings(inputs)
        positions = repeat(torch.arange(context_len, device=inputs.device), "p -> b p", b=batch_size)
        position_embedding = self.position_embeddings(positions)

        # Combine embeddings and reshape for transformer input
        embedding = token_embedding + position_embedding
        embedding = rearrange(embedding, "b s d -> s b d")

        return self.model(embedding)

# Training function
def train(model, train_loader, val_loader, optimizer, scheduler, device, num_train_batches, num_eval_batches):
    model.train()
    criterion = torch.nn.CrossEntropyLoss()
    loss_total, correct, total = 0.0, 0.0, 0
    step_val_acc_95, prev_val_acc, max_acc_increase_rate = None, 0, 0

    for count, batch in enumerate(train_loader, 1):
        inputs, labels = [t.to(device) for t in batch]

        optimizer.zero_grad()
        output = model(inputs)[-1, :, :]
        loss = criterion(output, labels)
        correct += (torch.argmax(output, dim=1) == labels).sum()
        loss_total += loss * len(labels)
        total += len(labels)

        loss.backward()
        optimizer.step()
        scheduler.step()

        if count % 100 == 0:
            val_metrics = evaluate(model, val_loader, device, num_eval_batches)
            val_acc = val_metrics["val_accuracy"]
            
            if step_val_acc_95 is None and val_acc >= 0.95:
                step_val_acc_95 = count * num_train_batches

            acc_increase_rate = (val_acc - prev_val_acc) / 100
            max_acc_increase_rate = max(max_acc_increase_rate, acc_increase_rate)
            prev_val_acc = val_acc

        if count >= num_train_batches:
            break

    metrics = {
        "train_accuracy": float(correct / total),
        "train_loss": float(loss_total / total),
        "step_val_acc_95": step_val_acc_95,
        "max_acc_increase_rate": max_acc_increase_rate,
    }
    return metrics

# Evaluation function
# Evaluation function
def evaluate(model, val_loader, device, num_eval_batches):
    model.eval()
    criterion = torch.nn.CrossEntropyLoss()
    correct, loss, total = 0, 0.0, 0

    with torch.no_grad():
        for count, batch in enumerate(val_loader, 1):
            inputs, labels = [t.to(device) for t in batch]
            output = model(inputs)[-1, :, :]
            correct += (torch.argmax(output, dim=1) == labels).sum()
            loss += criterion(output, labels) * len(labels)
            total += len(labels)

            if count >= num_eval_batches:
                break

    metrics = {
        "val_accuracy": float(correct / total),
        "val_loss": float(loss / total)
    }
    return metrics

# Main experiment function
def run(out_dir, dataset, seed_offset):
    os.makedirs(out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(1337 + seed_offset)

    # Setup data loaders
    train_loader, val_loader, n_vocab, n_output = get_data(
        operation=dataset,
        prime=97,
        training_fraction=0.5,
        batch_size=512,
    )

    # Initialize model
    model = Transformer(
        num_layers=2,
        dim_model=128,
        num_heads=4,
        vocab_size=n_vocab,
        output_size=n_output,
        seq_len=5,
    ).to(device)

    # Setup optimizer and scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-3,
        betas=(0.9, 0.98),
        weight_decay=0.5,
    )
    num_train_batches = 10
    num_eval_batches = 8
    num_total_updates = 7500
    warmup_steps = 50
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=lambda s: min(s / warmup_steps, 1)
    )

    # Training loop
    final_info, train_log_info, val_log_info = [], [], []
    step_val_acc_99 = num_total_updates
    for ep in range(num_total_updates // num_train_batches):
        train_metrics = train(
            model,
            train_loader,
            val_loader,
            optimizer,
            scheduler,
            device,
            num_train_batches,
            num_eval_batches,
        )
        val_metrics = evaluate(
            model,
            val_loader,
            device,
            num_eval_batches,
        )
        train_metrics["step"] = (ep + 1) * num_train_batches
        val_metrics["step"] = (ep + 1) * num_train_batches

        if step_val_acc_99 == num_total_updates and val_metrics["val_accuracy"] > 0.99:
            step_val_acc_99 = val_metrics["step"]
        train_log_info.append(train_metrics)
        val_log_info.append(val_metrics)

    # Collect final metrics
    final_info = {
        "final_train_loss": train_metrics["train_loss"],
        "final_val_loss": val_metrics["val_loss"],
        "final_train_acc": train_metrics["train_accuracy"],
        "final_val_acc": val_metrics["val_accuracy"],
        "step_val_acc_99": step_val_acc_99 if step_val_acc_99 != num_total_updates else None,
        "step_val_acc_95": train_metrics["step_val_acc_95"],
        "max_acc_increase_rate": train_metrics["max_acc_increase_rate"],
    }
    print(final_info)
    
    # Save results
    with open(os.path.join(out_dir, f"final_info_{dataset}_{seed_offset}.json"), "w") as f:
        json.dump(final_info, f)
    
    return final_info, train_log_info, val_log_info

# Command-line argument parser
parser = argparse.ArgumentParser(description="Run experiment")
parser.add_argument("--out_dir", type=str, default="run_0", help="Output directory")
args = parser.parse_args()

# Main execution
if __name__ == "__main__":
    num_seeds = {
        "x_div_y": 3,
        "x_plus_y": 3,
        "x_minus_y": 3,
        "permutation": 3,
    }

    out_dir = args.out_dir
    all_results = {}
    final_infos = {}
    
    # Run experiments for each dataset and seed
    for dataset in ["x_div_y", "x_minus_y", "x_plus_y", "permutation"]:
        final_info_list = []
        for seed_offset in range(num_seeds[dataset]):
            print(f"Running {dataset} with seed offset {seed_offset}")
            final_info, train_info, val_info = run(args.out_dir, dataset, seed_offset)
            all_results[f"{dataset}_{seed_offset}_final_info"] = final_info
            all_results[f"{dataset}_{seed_offset}_train_info"] = train_info
            all_results[f"{dataset}_{seed_offset}_val_info"] = val_info
            final_info_list.append(final_info)
        
        # Compute statistics across seeds
        final_info_dict = {
            k: [d[k] for d in final_info_list if d[k] is not None] for k in final_info_list[0].keys()
        }
        means = {f"{k}_mean": np.mean(v) if v else None for k, v in final_info_dict.items()}
        stderrs = {
            f"{k}_stderr": np.std(v) / np.sqrt(len(v)) if v else None for k, v in final_info_dict.items()
        }
        final_infos[dataset] = {
            "means": means,
            "stderrs": stderrs,
            "final_info_dict": final_info_dict,
        }

    # Save aggregated results
    with open(os.path.join(out_dir, "final_info.json"), "w") as f:
        json.dump(final_infos, f)

    with open(os.path.join(out_dir, "all_results.npy"), "wb") as f:
        np.save(f, all_results)
