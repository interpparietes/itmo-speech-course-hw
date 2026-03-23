import os
import torch
import torchaudio
import librosa
from wav2vec2decoder import Wav2Vec2Decoder
import jiwer
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
from typing import List, Tuple, Optional
import numpy as np


class DatasetLoader:
    """Load audio files and transcripts for evaluation"""
    
    def __init__(self, audio_dir: str, trans_file: str = None, use_manifest: bool = True):
        """
        Initialize dataset loader.
        
        Args:
            audio_dir: Directory containing .wav files
            trans_file: Optional path to transcripts file. If None, will try to find it.
            use_manifest: If True, look for manifest.csv file
        """
        self.audio_dir = audio_dir
        self.data = []
        
        # Try to load from manifest.csv first
        if use_manifest:
            manifest_path = os.path.join(audio_dir, 'manifest.csv')
            if os.path.exists(manifest_path):
                print(f"Loading transcripts from manifest.csv: {manifest_path}")
                self._load_from_manifest(manifest_path)
                return
        
        # If trans_file provided or try to find it
        if trans_file is None:
            trans_file = self._find_transcript_file(audio_dir)
        
        if trans_file and os.path.exists(trans_file):
            self._load_transcripts(trans_file)
        else:
            # Try to load from individual transcript files
            self._load_individual_transcripts()
    
    def _load_from_manifest(self, manifest_path: str):
        """Load transcripts from manifest.csv file with format: path,text"""
        try:
            # Read CSV file
            with open(manifest_path, 'r') as f:
                lines = f.readlines()
            
            # Skip header if it exists
            start_idx = 0
            if lines and ('path' in lines[0].lower() or 'text' in lines[0].lower()):
                start_idx = 1
            
            for line in lines[start_idx:]:
                line = line.strip()
                if not line:
                    continue
                
                # Split by comma - careful: text might contain commas
                # Find the first comma separating path and text
                parts = line.split(',', 1)
                
                if len(parts) == 2:
                    file_path = parts[0].strip()
                    transcript = parts[1].strip()
                    
                    # Remove quotes if present
                    if transcript.startswith('"') and transcript.endswith('"'):
                        transcript = transcript[1:-1]
                    
                    # Extract file ID from path
                    file_id = os.path.splitext(os.path.basename(file_path))[0]
                    
                    self.data.append((file_id, transcript.lower()))
                else:
                    print(f"Warning: Invalid line in manifest: {line}")
            
            print(f"Loaded {len(self.data)} samples from manifest")
            
            # Show first few samples for verification
            for i in range(min(3, len(self.data))):
                print(f"  Sample {i}: {self.data[i][0]} -> {self.data[i][1][:50]}...")
                
        except Exception as e:
            print(f"Error reading manifest.csv: {e}")
            self._load_transcripts_manual(manifest_path)
    
    def _load_transcripts_manual(self, manifest_path: str):
        """Manually parse manifest.csv"""
        with open(manifest_path, 'r') as f:
            lines = f.readlines()
            
        # Skip header if exists
        start_idx = 0
        if lines and ('file' in lines[0].lower() or 'wav' in lines[0].lower()):
            start_idx = 1
        
        for line in lines[start_idx:]:
            line = line.strip()
            if not line:
                continue
            
            # Try to split by comma
            parts = line.split(',')
            if len(parts) >= 2:
                # First part is filename, rest is transcript
                file_name = parts[0].strip()
                transcript = ','.join(parts[1:]).strip()
                file_id = os.path.splitext(os.path.basename(file_name))[0]
                self.data.append((file_id, transcript.lower()))
            elif len(parts) == 1:
                # Maybe it's just a filename without transcript
                file_id = os.path.splitext(parts[0])[0]
                self.data.append((file_id, ""))
    
    def _find_transcript_file(self, audio_dir: str) -> Optional[str]:
        """Try to find transcript file in audio directory"""
        possible_names = ['trans.txt', 'transcripts.txt', 'test.txt', 'metadata.csv']
        for name in possible_names:
            path = os.path.join(audio_dir, name)
            if os.path.exists(path):
                print(f"Found transcript file: {path}")
                return path
        return None
    
    def _load_transcripts(self, trans_file: str):
        """Load transcripts from file"""
        with open(trans_file, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                
                # Try different possible formats
                parts = line.split(maxsplit=1)
                if len(parts) == 2:
                    file_id, transcript = parts
                    self.data.append((file_id, transcript.lower()))
                elif len(parts) == 1:
                    # Maybe it's just the transcript with filename from audio file name
                    file_id = os.path.splitext(os.path.basename(parts[0]))[0]
                    transcript = ""
                    self.data.append((file_id, transcript))
                else:
                    print(f"Warning: Invalid line in {trans_file}: {line}")
    
    def _load_individual_transcripts(self):
        """Load transcripts from individual .txt files for each audio"""
        wav_files = [f for f in os.listdir(self.audio_dir) if f.endswith('.wav')]
        
        for wav_file in wav_files:
            file_id = os.path.splitext(wav_file)[0]
            txt_file = os.path.join(self.audio_dir, f"{file_id}.txt")
            transcript = ""
            
            if os.path.exists(txt_file):
                with open(txt_file, 'r') as f:
                    transcript = f.read().strip().lower()
            
            self.data.append((file_id, transcript))
        
        if not self.data and wav_files:
            # If no transcript files, just add all wav files with empty transcripts
            for wav_file in wav_files:
                file_id = os.path.splitext(wav_file)[0]
                self.data.append((file_id, ""))
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        """Return (audio_tensor, transcript) for sample idx"""
        file_id, transcript = self.data[idx]
        
        # Try to find audio file
        audio_path = None
        
        # Try with file_id directly
        for ext in ['.wav', '.flac', '.mp3']:
            test_path = os.path.join(self.audio_dir, f"{file_id}{ext}")
            if os.path.exists(test_path):
                audio_path = test_path
                break
        
        # If not found, try with sample_ prefix (for examples)
        if audio_path is None:
            for ext in ['.wav', '.flac', '.mp3']:
                test_path = os.path.join(self.audio_dir, f"sample_{file_id}{ext}")
                if os.path.exists(test_path):
                    audio_path = test_path
                    break
        
        if audio_path is None:
            raise FileNotFoundError(f"Audio file not found for {file_id} in {self.audio_dir}")
        
        # Load audio with librosa (avoids torchcodec issues)
        try:
            waveform, sample_rate = librosa.load(audio_path, sr=16000, mono=True)
            waveform = torch.from_numpy(waveform).float()
            return waveform, transcript
        except Exception as e:
            print(f"Error loading {audio_path}: {e}")
            return torch.zeros(16000), transcript
    
    def __iter__(self):
        """Iterate over dataset"""
        for i in range(len(self)):
            yield self[i]


def evaluate_dataset(decoder: Wav2Vec2Decoder,
                     dataset: DatasetLoader,
                     method: str = 'greedy',
                     **kwargs) -> Tuple[float, float]:
    """
    Evaluate decoder on dataset.
    """
    predictions = []
    references = []
    
    decoder.model.eval()
    
    for waveform, reference in tqdm(dataset, desc=f"Evaluating {method}"):
        # waveform is 1D [T] from librosa
        
        # Use processor to prepare input (handles resampling and normalization)
        inputs = decoder.processor(waveform, sampling_rate=16000, return_tensors="pt")
        input_values = inputs.input_values.to(decoder.device)  # [1, T]
        
        # Get logits from model
        with torch.no_grad():
            logits = decoder.model(input_values).logits
        
        # Decode
        if method == 'greedy':
            pred = decoder.greedy_decode(logits)
        elif method == 'beam':
            pred = decoder.beam_search_decode(logits, **kwargs)
        elif method == 'shallow':
            pred = decoder.beam_search_with_lm(logits, **kwargs)
        elif method == 'rescore':
            pred = decoder.lm_rescore(logits, **kwargs)
        else:
            raise ValueError(f"Unknown method: {method}")
        
        predictions.append(pred[0].lower())
        references.append(reference.lower())
    
    # Calculate metrics
    wer = jiwer.wer(references, predictions)
    cer = jiwer.cer(references, predictions)
    
    return wer, cer


def task1_greedy(decoder: Wav2Vec2Decoder, dataset: DatasetLoader) -> Tuple[float, float]:
    """Task 1: Greedy decoding"""
    print("\n" + "="*60)
    print("Task 1: Greedy Decoding")
    print("="*60)
    
    wer, cer = evaluate_dataset(decoder, dataset, method='greedy')
    
    print(f"WER: {wer:.2%}")
    print(f"CER: {cer:.2%}")
    
    return wer, cer


def task2_beam_search(decoder: Wav2Vec2Decoder, dataset: DatasetLoader) -> List[dict]:
    """Task 2: Beam search with different beam widths"""
    print("\n" + "="*60)
    print("Task 2: Beam Search - Varying Beam Width")
    print("="*60)
    
    beam_widths = [1, 3, 10, 50]
    results = []
    
    for bw in beam_widths:
        print(f"\nBeam width: {bw}")
        wer, cer = evaluate_dataset(decoder, dataset, method='beam', beam_width=bw)
        print(f"WER: {wer:.2%}, CER: {cer:.2%}")
        results.append({'beam_width': bw, 'wer': wer, 'cer': cer})
    
    # Plot results
    df = pd.DataFrame(results)
    
    plt.figure(figsize=(10, 6))
    plt.plot(df['beam_width'], df['wer'] * 100, marker='o', label='WER', linewidth=2)
    plt.plot(df['beam_width'], df['cer'] * 100, marker='s', label='CER', linewidth=2)
    plt.xlabel('Beam Width', fontsize=12)
    plt.ylabel('Error Rate (%)', fontsize=12)
    plt.title('Beam Search: Quality vs Beam Width', fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.xscale('log')
    plt.tight_layout()
    plt.savefig('task2_beam_search_quality.png', dpi=150)
    plt.show()
    
    df.to_csv('task2_beam_search_results.csv', index=False)
    
    return results


def task3_temperature_sweep(decoder: Wav2Vec2Decoder, dataset: DatasetLoader) -> List[dict]:
    """Task 3: Temperature sweep with greedy decoding"""
    print("\n" + "="*60)
    print("Task 3: Temperature Sweep (Greedy Decoding)")
    print("="*60)
    
    temperatures = [0.5, 0.8, 1.0, 1.2, 1.5, 2.0]
    results = []
    
    original_temp = decoder.temperature
    
    for T in temperatures:
        print(f"\nTemperature: {T}")
        decoder.temperature = T
        wer, cer = evaluate_dataset(decoder, dataset, method='greedy')
        print(f"WER: {wer:.2%}, CER: {cer:.2%}")
        results.append({'temperature': T, 'wer': wer, 'cer': cer})
    
    # Restore temperature
    decoder.temperature = original_temp
    
    # Plot results
    df = pd.DataFrame(results)
    
    plt.figure(figsize=(10, 6))
    plt.plot(df['temperature'], df['wer'] * 100, marker='o', label='WER', linewidth=2)
    plt.plot(df['temperature'], df['cer'] * 100, marker='s', label='CER', linewidth=2)
    plt.xlabel('Temperature', fontsize=12)
    plt.ylabel('Error Rate (%)', fontsize=12)
    plt.title('Temperature Scaling Effect on Greedy Decoding', fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('task3_temperature_sweep.png', dpi=150)
    plt.show()
    
    df.to_csv('task3_temperature_sweep_results.csv', index=False)
    
    return results


def task4_shallow_fusion_grid(decoder: Wav2Vec2Decoder, 
                              dataset: DatasetLoader,
                              beam_width: int = 10) -> Tuple[dict, List[dict]]:
    """Task 4: Grid search for shallow fusion parameters"""
    print("\n" + "="*60)
    print("Task 4: Shallow Fusion - Grid Search")
    print("="*60)
    
    alphas = [0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0]
    betas = [0.0, 0.5, 1.0, 1.5]
    
    results = []
    
    for alpha in alphas:
        for beta in betas:
            print(f"\nAlpha: {alpha}, Beta: {beta}")
            wer, cer = evaluate_dataset(decoder, dataset, method='shallow',
                                       beam_width=beam_width, alpha=alpha, beta=beta)
            print(f"WER: {wer:.2%}, CER: {cer:.2%}")
            results.append({'alpha': alpha, 'beta': beta, 'wer': wer, 'cer': cer})
    
    # Save results
    df = pd.DataFrame(results)
    df.to_csv('task4_shallow_fusion_grid.csv', index=False)
    
    # Create heatmaps
    pivot_wer = df.pivot(index='alpha', columns='beta', values='wer') * 100
    pivot_cer = df.pivot(index='alpha', columns='beta', values='cer') * 100
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    sns.heatmap(pivot_wer, annot=True, fmt='.1f', cmap='YlOrRd', ax=ax1)
    ax1.set_title('WER (%) for Shallow Fusion', fontsize=14)
    ax1.set_xlabel('Beta', fontsize=12)
    ax1.set_ylabel('Alpha', fontsize=12)
    
    sns.heatmap(pivot_cer, annot=True, fmt='.1f', cmap='YlOrRd', ax=ax2)
    ax2.set_title('CER (%) for Shallow Fusion', fontsize=14)
    ax2.set_xlabel('Beta', fontsize=12)
    ax2.set_ylabel('Alpha', fontsize=12)
    
    plt.tight_layout()
    plt.savefig('task4_shallow_fusion_heatmap.png', dpi=150)
    plt.show()
    
    # Find best parameters
    best = df.loc[df['wer'].idxmin()]
    print(f"\nBest parameters: alpha={best['alpha']}, beta={best['beta']}")
    print(f"Best WER: {best['wer']:.2%}")
    
    return best.to_dict(), results


def task5_4gram_lm(decoder: Wav2Vec2Decoder,
                   dataset: DatasetLoader,
                   best_params: dict,
                   beam_width: int = 10) -> Tuple[float, float]:
    """Task 5: Evaluate with 4-gram LM"""
    print("\n" + "="*60)
    print("Task 5: 4-gram Language Model")
    print("="*60)
    
    wer, cer = evaluate_dataset(decoder, dataset, method='shallow',
                               beam_width=beam_width,
                               alpha=best_params['alpha'],
                               beta=best_params['beta'])
    
    print(f"4-gram LM - WER: {wer:.2%}, CER: {cer:.2%}")
    
    return wer, cer


def task6_lm_rescoring(decoder: Wav2Vec2Decoder,
                       dataset: DatasetLoader,
                       beam_width: int = 100) -> List[dict]:
    """Task 6: LM rescoring grid search"""
    print("\n" + "="*60)
    print("Task 6: LM Rescoring - Grid Search")
    print("="*60)
    
    alphas = [0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0]
    betas = [0.0, 0.5, 1.0, 1.5]
    
    results = []
    
    for alpha in alphas:
        for beta in betas:
            print(f"\nAlpha: {alpha}, Beta: {beta}")
            wer, cer = evaluate_dataset(decoder, dataset, method='rescore',
                                       beam_width=beam_width, alpha=alpha, beta=beta)
            print(f"WER: {wer:.2%}, CER: {cer:.2%}")
            results.append({'alpha': alpha, 'beta': beta, 'wer': wer, 'cer': cer})
    
    # Save results
    df = pd.DataFrame(results)
    df.to_csv('task6_lm_rescoring_grid.csv', index=False)
    
    # Create heatmaps
    pivot_wer = df.pivot(index='alpha', columns='beta', values='wer') * 100
    pivot_cer = df.pivot(index='alpha', columns='beta', values='cer') * 100
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    sns.heatmap(pivot_wer, annot=True, fmt='.1f', cmap='YlOrRd', ax=ax1)
    ax1.set_title('WER (%) for LM Rescoring', fontsize=14)
    ax1.set_xlabel('Beta', fontsize=12)
    ax1.set_ylabel('Alpha', fontsize=12)
    
    sns.heatmap(pivot_cer, annot=True, fmt='.1f', cmap='YlOrRd', ax=ax2)
    ax2.set_title('CER (%) for LM Rescoring', fontsize=14)
    ax2.set_xlabel('Beta', fontsize=12)
    ax2.set_ylabel('Alpha', fontsize=12)
    
    plt.tight_layout()
    plt.savefig('task6_lm_rescoring_heatmap.png', dpi=150)
    plt.show()
    
    best = df.loc[df['wer'].idxmin()]
    print(f"\nBest parameters: alpha={best['alpha']}, beta={best['beta']}")
    print(f"Best WER: {best['wer']:.2%}")
    
    return results


def task7_ood_evaluation(decoder: Wav2Vec2Decoder,
                         librispeech: DatasetLoader,
                         earnings22: DatasetLoader,
                         best_shallow: dict,
                         best_rescore: dict) -> dict:
    """Task 7: Out-of-domain evaluation on Earnings22"""
    print("\n" + "="*60)
    print("Task 7: Out-of-Domain Evaluation")
    print("="*60)
    
    results = {}
    
    # Greedy on both
    print("\nGreedy Decoding:")
    wer_ls, cer_ls = evaluate_dataset(decoder, librispeech, method='greedy')
    wer_earn, cer_earn = evaluate_dataset(decoder, earnings22, method='greedy')
    print(f"  LibriSpeech: WER={wer_ls:.2%}, CER={cer_ls:.2%}")
    print(f"  Earnings22:  WER={wer_earn:.2%}, CER={cer_earn:.2%}")
    results['greedy'] = {'librispeech': (wer_ls, cer_ls), 'earnings22': (wer_earn, cer_earn)}
    
    # Beam search
    print("\nBeam Search (bw=10):")
    wer_ls, cer_ls = evaluate_dataset(decoder, librispeech, method='beam', beam_width=10)
    wer_earn, cer_earn = evaluate_dataset(decoder, earnings22, method='beam', beam_width=10)
    print(f"  LibriSpeech: WER={wer_ls:.2%}, CER={cer_ls:.2%}")
    print(f"  Earnings22:  WER={wer_earn:.2%}, CER={cer_earn:.2%}")
    results['beam'] = {'librispeech': (wer_ls, cer_ls), 'earnings22': (wer_earn, cer_earn)}
    
    # Shallow fusion with best parameters
    print("\nShallow Fusion (best params from Task 4):")
    wer_ls, cer_ls = evaluate_dataset(decoder, librispeech, method='shallow',
                                      beam_width=10, alpha=best_shallow['alpha'], beta=best_shallow['beta'])
    wer_earn, cer_earn = evaluate_dataset(decoder, earnings22, method='shallow',
                                          beam_width=10, alpha=best_shallow['alpha'], beta=best_shallow['beta'])
    print(f"  LibriSpeech: WER={wer_ls:.2%}, CER={cer_ls:.2%}")
    print(f"  Earnings22:  WER={wer_earn:.2%}, CER={cer_earn:.2%}")
    results['shallow'] = {'librispeech': (wer_ls, cer_ls), 'earnings22': (wer_earn, cer_earn)}
    
    # Rescoring with best parameters
    print("\nLM Rescoring (best params from Task 6):")
    wer_ls, cer_ls = evaluate_dataset(decoder, librispeech, method='rescore',
                                      beam_width=100, alpha=best_rescore['alpha'], beta=best_rescore['beta'])
    wer_earn, cer_earn = evaluate_dataset(decoder, earnings22, method='rescore',
                                          beam_width=100, alpha=best_rescore['alpha'], beta=best_rescore['beta'])
    print(f"  LibriSpeech: WER={wer_ls:.2%}, CER={cer_ls:.2%}")
    print(f"  Earnings22:  WER={wer_earn:.2%}, CER={cer_earn:.2%}")
    results['rescore'] = {'librispeech': (wer_ls, cer_ls), 'earnings22': (wer_earn, cer_earn)}
    
    return results


def task7b_temperature_ood(decoder: Wav2Vec2Decoder,
                          dataset: DatasetLoader,
                          best_shallow: dict) -> List[dict]:
    """Task 7b: Temperature sweep on out-of-domain data with shallow fusion"""
    print("\n" + "="*60)
    print("Task 7b: Temperature Sweep on Earnings22 (with Shallow Fusion)")
    print("="*60)
    
    temperatures = [0.5, 1.0, 1.5, 2.0]
    results = []
    
    original_temp = decoder.temperature
    
    for T in temperatures:
        print(f"\nTemperature: {T}")
        decoder.temperature = T
        wer, cer = evaluate_dataset(decoder, dataset, method='shallow',
                                    beam_width=10, alpha=best_shallow['alpha'], beta=best_shallow['beta'])
        print(f"WER: {wer:.2%}, CER: {cer:.2%}")
        results.append({'temperature': T, 'wer': wer, 'cer': cer})
    
    decoder.temperature = original_temp
    
    # Plot results
    df = pd.DataFrame(results)
    
    plt.figure(figsize=(10, 6))
    plt.plot(df['temperature'], df['wer'] * 100, marker='o', label='WER (with LM)', linewidth=2)
    plt.plot(df['temperature'], df['cer'] * 100, marker='s', label='CER (with LM)', linewidth=2)
    plt.xlabel('Temperature', fontsize=12)
    plt.ylabel('Error Rate (%)', fontsize=12)
    plt.title('Temperature Effect on Out-of-Domain Data (Shallow Fusion)', fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('task7b_temperature_ood.png', dpi=150)
    plt.show()
    
    df.to_csv('task7b_temperature_ood_results.csv', index=False)
    
    return results


def main():
    """Main function to run all experiments"""
    
    print("="*60)
    print("ASR Decoding Assignment - Running Experiments")
    print("="*60)
    
    # Initialize decoder
    print("\nLoading decoder...")
    
    # Check if LM file exists
    lm_path = "lm/3-gram.pruned.1e-7.arpa.gz"
    if not os.path.exists(lm_path):
        print(f"Warning: LM file not found at {lm_path}")
        print("Will run without language model")
        lm_path = None
    
    decoder = Wav2Vec2Decoder(
        model_name="facebook/wav2vec2-base-100h",
        temperature=1.0,
        lm_model_path=lm_path,
        device="cpu"  # Change to "cuda" if GPU available
    )
    
    # Load datasets
    print("\nLoading datasets...")
    
    # Load LibriSpeech test-other
    librispeech_dir = "data/librispeech_test_other"
    librispeech = None
    
    if os.path.exists(librispeech_dir):
        try:
            librispeech = DatasetLoader(
                audio_dir=librispeech_dir,
                use_manifest=True
            )
            print(f"✓ LibriSpeech test-other: {len(librispeech)} samples")
        except Exception as e:
            print(f"✗ Error loading LibriSpeech dataset: {e}")
    else:
        print(f"✗ LibriSpeech directory not found: {librispeech_dir}")
    
    # Load Earnings22 test
    earnings_dir = "data/earnings22_test"
    earnings22 = None
    
    if os.path.exists(earnings_dir):
        try:
            earnings22 = DatasetLoader(
                audio_dir=earnings_dir,
                use_manifest=True
            )
            print(f"✓ Earnings22 test: {len(earnings22)} samples")
        except Exception as e:
            print(f"✗ Error loading Earnings22 dataset: {e}")
    else:
        print(f"✗ Earnings22 directory not found: {earnings_dir}")
    
    # If no datasets found, use examples
    if librispeech is None and earnings22 is None:
        print("\n⚠ No datasets found, using examples directory...")
        examples_dir = "examples"
        if os.path.exists(examples_dir):
            try:
                # For examples, we need to create a simple trans.txt
                examples_trans = os.path.join(examples_dir, "trans.txt")
                if not os.path.exists(examples_trans):
                    # Create a simple trans.txt for examples
                    example_transcripts = {
                        "sample1": "he had taken the wrong road entirely",
                        "sample2": "the committee met on thursday",
                        "sample3": "this is a test sample three",
                        "sample4": "sample four audio recording",
                        "sample5": "sample five with some text",
                        "sample6": "sample six example",
                        "sample7": "sample seven transcription",
                        "sample8": "sample eight final example"
                    }
                    with open(examples_trans, 'w') as f:
                        for name, text in example_transcripts.items():
                            f.write(f"{name} {text}\n")
                    print(f"Created {examples_trans}")
                
                librispeech = DatasetLoader(
                    audio_dir=examples_dir,
                    trans_file=examples_trans,
                    use_manifest=False
                )
                print(f"✓ Using examples: {len(librispeech)} samples")
            except Exception as e:
                print(f"✗ Error loading examples: {e}")
                return
        else:
            print("✗ Examples directory not found")
            return
    
    # Run experiments
    if librispeech:
        print("\n" + "="*60)
        print("Starting experiments on LibriSpeech/Examples")
        print("="*60)
        
        # Task 1
        task1_greedy(decoder, librispeech)
        
        # Task 2
        beam_results = task2_beam_search(decoder, librispeech)
        
        # Task 3
        temp_results = task3_temperature_sweep(decoder, librispeech)
        
        # Task 4 (only if LM is available)
        best_shallow = None
        if decoder.lm:
            best_shallow, shallow_results = task4_shallow_fusion_grid(decoder, librispeech)
        else:
            print("\n⚠ Skipping Task 4 (Shallow Fusion) - No LM loaded")
            best_shallow = {'alpha': 0.5, 'beta': 0.0}
        
        # Task 6 (only if LM is available)
        rescore_results = None
        best_rescore = None
        if decoder.lm:
            rescore_results = task6_lm_rescoring(decoder, librispeech)
            best_rescore = min(rescore_results, key=lambda x: x['wer'])
        else:
            print("\n⚠ Skipping Task 6 (LM Rescoring) - No LM loaded")
            best_rescore = {'alpha': 0.5, 'beta': 0.0}
        
        # Task 7 (if earnings22 available and LM available)
        if earnings22 and decoder.lm and best_shallow and best_rescore:
            print("\n" + "="*60)
            print("Starting experiments on Earnings22")
            print("="*60)
            ood_results = task7_ood_evaluation(decoder, librispeech, earnings22, best_shallow, best_rescore)
            task7b_temperature_ood(decoder, earnings22, best_shallow)
            
            # Create final summary table
            print("\n" + "="*60)
            print("FINAL SUMMARY TABLE")
            print("="*60)
            print("\nMethod\t\tLibriSpeech WER\tLibriSpeech CER\tEarnings22 WER\tEarnings22 CER")
            print("-" * 80)
            
            for method in ['greedy', 'beam', 'shallow', 'rescore']:
                if method in ood_results:
                    ls_wer, ls_cer = ood_results[method]['librispeech']
                    earn_wer, earn_cer = ood_results[method]['earnings22']
                    method_name = method.upper() if method != 'beam' else 'Beam (bw=10)'
                    if method == 'shallow':
                        method_name = 'Shallow Fusion'
                    elif method == 'rescore':
                        method_name = 'LM Rescoring'
                    print(f"{method_name:<15}\t{ls_wer:.2%}\t\t{ls_cer:.2%}\t\t{earn_wer:.2%}\t\t{earn_cer:.2%}")
            
            # Save summary to CSV
            summary_data = []
            for method in ['greedy', 'beam', 'shallow', 'rescore']:
                if method in ood_results:
                    ls_wer, ls_cer = ood_results[method]['librispeech']
                    earn_wer, earn_cer = ood_results[method]['earnings22']
                    summary_data.append({
                        'Method': method.upper(),
                        'LibriSpeech_WER': ls_wer,
                        'LibriSpeech_CER': ls_cer,
                        'Earnings22_WER': earn_wer,
                        'Earnings22_CER': earn_cer
                    })
            
            summary_df = pd.DataFrame(summary_data)
            summary_df.to_csv('final_summary_results.csv', index=False)
            print("\n✓ Summary saved to final_summary_results.csv")
    
    print("\nAll experiments completed!")
    print("Results saved to:")
    print("  - Various CSV files for each task")
    print("  - PNG plots for visualization")
    print("  - final_summary_results.csv (if all data available)")


if __name__ == '__main__':
    main()