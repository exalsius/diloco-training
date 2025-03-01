import torchaudio
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import Wav2Vec2Processor
import torch
from torch.nn.utils.rnn import pad_sequence
from datasets.distributed import split_dataset_by_node


def get_librispeech(world_size, local_rank, per_device_train_batch_size, split="train.clean.100"):
    """Loads LibriSpeech dataset for Wav2Vec2 training."""
    if split == 'train':
        split="train.clean.100"
    # Load Wav2Vec2 model and processor
    model_name = "facebook/wav2vec2-base-960h"
    processor = Wav2Vec2Processor.from_pretrained(model_name)
        # Function to preprocess audio
    def preprocess(batch):
        audio = batch["audio"]["array"][:32000]
        input_values = processor(audio, sampling_rate=16000, return_tensors="pt").input_values[0]
        batch["input_values"] = input_values
        return batch

    # Apply preprocessing
    dataset = load_dataset("librispeech_asr", cache_dir="/workspace/diloco_training/datasets", split=split)
    dataset = dataset.select(range(10))  # For testing purposes
    dataset = dataset.map(preprocess, remove_columns=["file", "audio"])
    def collate_fn(batch):
        input_values = [torch.tensor(sample["input_values"]) for sample in batch]
        labels = [torch.tensor(processor.tokenizer(sample["text"]).input_ids) for sample in batch]
        
        input_values = pad_sequence(input_values, batch_first=True, padding_value=0.0)
        labels = pad_sequence(labels, batch_first=True, padding_value=-100)  # -100 is ignored in loss computation
        
        return {"input_values": input_values, "labels": labels}
    dataset = split_dataset_by_node(
        dataset=dataset, world_size=world_size, rank=local_rank
    )
    # Dataloader
    dataloader = DataLoader(dataset, batch_size=per_device_train_batch_size, collate_fn=collate_fn, shuffle=True)

    return dataset, dataloader



def main():
    # Test the get_librispeech function with a small number of samples
    split = "train.clean.100"


    dataset, dataloader = get_librispeech(1,0,4)

    print(f"Loaded {len(dataset)} samples from the {split} split.")
    for i, batch in enumerate(dataloader):
        print(f"Batch {i+1}: {batch['input_values'].shape, batch['labels'].shape}")
        if i >= 2:  # Print only the first 3 batches
            break

if __name__ == "__main__":
    main()