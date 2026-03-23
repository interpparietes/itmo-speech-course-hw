"""
Извлечение примеров для качественного анализа
"""

import torch
import pandas as pd
from wav2vec2decoder import Wav2Vec2Decoder
from run_experiments import DatasetLoader, evaluate_dataset

def extract_examples():
    """Извлечение предсказаний для всех методов на небольшой выборке"""
    
    print("Loading decoder...")
    decoder = Wav2Vec2Decoder(
        model_name="facebook/wav2vec2-base-100h",
        temperature=1.0,
        lm_model_path="lm/3-gram.pruned.1e-7.arpa.gz",
        device="cpu"
    )
    
    print("Loading dataset...")
    dataset = DatasetLoader("data/librispeech_test_other", use_manifest=True)
    
    # Берем только первые 100 файлов для анализа, чтобы не прогонять все еще раз
    dataset.data = dataset.data[:100]
    print(f"Using {len(dataset)} samples for analysis")
    
    # Параметры из лучших конфигураций
    best_alpha = 0.1
    best_beta = 1.5
    beam_width = 10
    
    results = {}
    
    # Greedy
    print("\nRunning greedy...")
    df_greedy, _, _ = evaluate_dataset_with_save(decoder, dataset, method='greedy')
    results['greedy'] = df_greedy
    
    # Beam (без LM)
    print("\nRunning beam search (no LM)...")
    df_beam, _, _ = evaluate_dataset_with_save(decoder, dataset, method='beam', beam_width=beam_width)
    results['beam'] = df_beam
    
    # Shallow fusion
    print("\nRunning shallow fusion...")
    df_shallow, _, _ = evaluate_dataset_with_save(decoder, dataset, method='shallow', 
                                                  beam_width=beam_width, alpha=best_alpha, beta=best_beta)
    results['shallow'] = df_shallow
    
    # Rescoring
    print("\nRunning rescoring...")
    df_rescore, _, _ = evaluate_dataset_with_save(decoder, dataset, method='rescore', 
                                                  beam_width=100, alpha=best_alpha, beta=best_beta)
    results['rescore'] = df_rescore
    
    # Объединяем все предсказания
    combined = results['greedy'].copy()
    combined['beam_pred'] = results['beam']['prediction']
    combined['shallow_pred'] = results['shallow']['prediction']
    combined['rescore_pred'] = results['rescore']['prediction']
    
    # Находим примеры, где LM что-то изменила
    combined['beam_vs_shallow_diff'] = combined['beam_pred'] != combined['shallow_pred']
    combined['beam_vs_rescore_diff'] = combined['beam_pred'] != combined['rescore_pred']
    combined['any_diff'] = combined['beam_vs_shallow_diff'] | combined['beam_vs_rescore_diff']
    
    # Сохраняем все результаты
    combined.to_csv('all_predictions.csv', index=False)
    
    # Выводим примеры, где есть изменения
    diff_examples = combined[combined['any_diff'] == True]
    print(f"\nFound {len(diff_examples)} examples where LM changed prediction")
    
    # Сохраняем только измененные примеры
    diff_examples.to_csv('lm_changed_examples.csv', index=False)
    
    # Выводим первые 10 примеров для отчета
    print("\n" + "="*80)
    print("EXAMPLES WHERE LM CHANGED PREDICTION (first 10)")
    print("="*80)
    
    for i, row in diff_examples.head(10).iterrows():
        print(f"\n{i}. File: {row['file_id']}")
        print(f"   REF:  {row['reference']}")
        print(f"   BEAM: {row['beam_pred']}")
        if row['beam_vs_shallow_diff']:
            print(f"   SF:   {row['shallow_pred']}")
        else:
            print(f"   SF:   {row['shallow_pred']} (unchanged)")
        if row['beam_vs_rescore_diff']:
            print(f"   RS:   {row['rescore_pred']}")
        else:
            print(f"   RS:   {row['rescore_pred']} (unchanged)")
        print("-"*80)
    
    return combined, diff_examples

def evaluate_dataset_with_save(decoder, dataset, method='greedy', **kwargs):
    """Вспомогательная функция для evaluate с сохранением"""
    predictions = []
    references = []
    file_ids = []
    
    from tqdm import tqdm
    import torch
    
    for i, (waveform, reference) in enumerate(tqdm(dataset, desc=f"Evaluating {method}")):
        inputs = decoder.processor(waveform, sampling_rate=16000, return_tensors="pt")
        input_values = inputs.input_values.to(decoder.device)
        
        with torch.no_grad():
            logits = decoder.model(input_values).logits
        
        if method == 'greedy':
            pred = decoder.greedy_decode(logits)
        elif method == 'beam':
            pred = decoder.beam_search_decode(logits, **kwargs)
        elif method == 'shallow':
            pred = decoder.beam_search_with_lm(logits, **kwargs)
        elif method == 'rescore':
            pred = decoder.lm_rescore(logits, **kwargs)
        
        predictions.append(pred[0].lower())
        references.append(reference.lower())
        file_ids.append(dataset.data[i][0])
    
    import pandas as pd
    df = pd.DataFrame({
        'file_id': file_ids,
        'reference': references,
        'prediction': predictions
    })
    
    return df, None, None

if __name__ == "__main__":
    combined, diff_examples = extract_examples()
    print("\n✓ Results saved to all_predictions.csv and lm_changed_examples.csv")