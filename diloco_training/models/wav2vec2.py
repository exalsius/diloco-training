from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor


def get_wav2vec2(model_name="facebook/wav2vec2-large-960h"):
    """Return Wav2Vec2 model for speech recognition"""
    model = Wav2Vec2ForCTC.from_pretrained(model_name)
    processor = Wav2Vec2Processor.from_pretrained(model_name)
    return model, processor


# Example usage
if __name__ == "__main__":
    model, processor = get_wav2vec2()
    print(model)
