from datasets import load_dataset
from datasets.distributed import split_dataset_by_node
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, DataCollatorForLanguageModeling


def get_c4_pile(world_size, local_rank, per_device_train_batch_size, split="train"):
    """Loads C4/The Pile dataset for GPT-Neo training."""

    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-1.3B", fast=True)
    tokenizer.pad_token = "</s>"  # Ensure pad token is set for models that need it

    dataset = load_dataset("PrimeIntellect/c4-tiny", "en", ignore_verifications=True)

    def tokenize_fn(batch):
        return tokenizer(
            batch["text"], truncation=True, max_length=1024, padding="max_length"
        )

    tokenized_datasets = dataset.map(
        tokenize_fn, batched=True, remove_columns=["text", "timestamp", "url"]
    )
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    dataset = split_dataset_by_node(
        tokenized_datasets[split], world_size=world_size, rank=local_rank
    )
    dataloader = DataLoader(
        dataset, collate_fn=data_collator, batch_size=per_device_train_batch_size
    )

    return dataset, dataloader


if __name__ == "__main__":
    dataset, dataloader = get_c4_pile(
        world_size=1, local_rank=0, per_device_train_batch_size=32
    )
    print("Dataset length:", len(dataset))
    print("Dataloader length:", len(dataloader))
    print("Dataloader batch size:", dataloader.batch_size)

    for step, batch in enumerate(dataloader):
        print(batch["input_ids"].shape)
