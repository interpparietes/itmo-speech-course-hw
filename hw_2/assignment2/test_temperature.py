import torch
from wav2vec2decoder import Wav2Vec2Decoder
from run_experiments import DatasetLoader
import numpy as np

# Загружаем декодер
decoder = Wav2Vec2Decoder(
    model_name="facebook/wav2vec2-base-100h",
    temperature=1.0,
    device="cpu"
)

# Загружаем один sample
loader = DatasetLoader("data/librispeech_test_other", use_manifest=True)
waveform, reference = loader[0]

print(f"Reference: {reference}")

# Подготавливаем вход
inputs = decoder.processor(waveform, sampling_rate=16000, return_tensors="pt")
input_values = inputs.input_values

with torch.no_grad():
    logits = decoder.model(input_values).logits
    print(f"Logits shape: {logits.shape}")
    
    # Проверяем logits распределение
    print(f"Logits before temperature - min: {logits.min():.3f}, max: {logits.max():.3f}")
    
    # Применяем разные температуры
    temperatures = [0.5, 1.0, 2.0]
    
    for T in temperatures:
        print(f"\n=== Temperature: {T} ===")
        
        # Применяем температуру
        scaled_logits = logits / T
        probs = torch.softmax(scaled_logits, dim=-1)
        
        # Получаем предсказания
        pred_tokens = torch.argmax(probs, dim=-1)[0]
        print(f"First 20 tokens: {pred_tokens[:20].tolist()}")
        
        # Декодируем
        merged = []
        prev = None
        for token in pred_tokens.tolist():
            if token != 0:
                if token != prev:
                    merged.append(token)
                prev = token
            else:
                prev = None
        
        text = decoder._tokens_to_text(merged)
        print(f"Decoded: {text}")
        
        # Проверяем энтропию распределения
        entropy = -torch.sum(probs * torch.log(probs + 1e-10), dim=-1).mean()
        print(f"Avg entropy: {entropy:.3f}")