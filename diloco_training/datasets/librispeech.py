import torchaudio
from datasets import load_dataset
from torch.utils.data import DataLoader


def get_librispeech(split="train.clean.100", batch_size=32):
    """Loads LibriSpeech dataset for Wav2Vec2 training."""

    dataset = load_dataset("librispeech_asr", split=split)

    def transform_fn(batch):
        waveform, _ = torchaudio.load(batch["file"])
        return waveform

    dataset.set_transform(transform_fn)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    return dataset, dataloader
