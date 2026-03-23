import torch
import torch.nn.functional as F
import heapq
import kenlm
import math
from typing import List, Tuple, Optional, Dict
import numpy as np

def _log_add(a: float, b: float) -> float:
    """Numerically stable log(exp(a) + exp(b))."""
    if a == float('-inf'):
        return b
    if b == float('-inf'):
        return a
    if a > b:
        return a + math.log1p(math.exp(b - a))
    return b + math.log1p(math.exp(a - b))


class Wav2Vec2Decoder:
    """
    CTC Decoder for Wav2Vec2.0 acoustic model.
    Implements:
    - Greedy decoding
    - Beam search decoding
    - Beam search with language model (shallow fusion)
    - LM rescoring (second-pass)
    """
    
    def __init__(self, 
                 model_name: str = "facebook/wav2vec2-base-100h",
                 temperature: float = 1.0,
                 lm_model_path: Optional[str] = None,
                 device: str = "cpu"):
        """
        Initialize decoder with acoustic model and optional language model.
        
        Args:
            model_name: HuggingFace model name
            temperature: Temperature for scaling logits
            lm_model_path: Path to KenLM ARPA file (optional)
            device: Device to run model on
        """
        from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
        
        self.device = device
        self.temperature = temperature
        
        # Load model and processor
        self.processor = Wav2Vec2Processor.from_pretrained(model_name)
        self.model = Wav2Vec2ForCTC.from_pretrained(model_name)
        self.model.to(device)
        self.model.eval()
        
        # Vocabulary mapping
        self.vocab = self.processor.tokenizer.get_vocab()
        # Reverse mapping: index -> character
        self.idx2char = {v: k for k, v in self.vocab.items()}
        
        # Blank token is <pad> which is index 0
        self.blank_idx = 0
        
        # Load language model if provided
        self.lm = None
        if lm_model_path:
            self.lm = kenlm.Model(lm_model_path)
            # KenLM returns log10 probabilities, we need natural log
            self.ln10 = math.log(10.0)
    
    def _lm_score(self, sentence: str) -> float:
        """
        Get natural log probability of sentence from LM.
        
        Args:
            sentence: Text sentence
            
        Returns:
            Natural log probability
        """
        if self.lm is None:
            return 0.0
        # KenLM score returns log10 probability
        log10_prob = self.lm.score(sentence, bos=True, eos=False)
        # Convert to natural log
        return log10_prob * self.ln10
    
    def _tokens_to_text(self, tokens: List[int]) -> str:
        """
        Convert token indices to text.
        
        Args:
            tokens: List of token indices
            
        Returns:
            Text string
        """
        chars = []
        for idx in tokens:
            if idx != self.blank_idx and idx in self.idx2char:
                char = self.idx2char[idx]
                if char == '|':
                    chars.append(' ')
                else:
                    chars.append(char)
        return ''.join(chars).strip()
    
    # ==================== Task 1: Greedy Decoding ====================
    
    def greedy_decode(self, logits: torch.Tensor) -> List[str]:
        """
        Greedy CTC decoding: choose most probable token at each time step.
        
        Args:
            logits: Tensor of shape [batch_size, seq_len, vocab_size]
            
        Returns:
            List of decoded strings
        """
        # Apply temperature scaling and log_softmax
        logits = logits / self.temperature
        log_probs = torch.log_softmax(logits, dim=-1)  # [batch, T, vocab]
        
        # Get argmax at each time step
        predictions = torch.argmax(log_probs, dim=-1)  # [batch, T]
        
        results = []
        for batch_idx in range(predictions.shape[0]):
            seq = predictions[batch_idx]
            
            # Collapse repetitions and remove blanks
            decoded_tokens = []
            prev_token = None
            
            for token in seq:
                token = token.item()
                if token != self.blank_idx:  # Not blank
                    if token != prev_token:  # Different from previous
                        decoded_tokens.append(token)
                    prev_token = token
                else:
                    prev_token = None
            
            # Convert tokens to text
            text = self._tokens_to_text(decoded_tokens)
            results.append(text.lower())
        
        return results
    
    # ==================== Task 2: Beam Search Decoding ====================
    
    def beam_search_decode(self, 
                          logits: torch.Tensor, 
                          beam_width: int = 10) -> List[str]:
        """
        Beam search CTC decoding without language model.
        
        Args:
            logits: Tensor of shape [batch_size, seq_len, vocab_size]
            beam_width: Number of hypotheses to keep
            
        Returns:
            List of decoded strings
        """
        # Apply temperature and log_softmax
        logits = logits / self.temperature
        log_probs = torch.log_softmax(logits, dim=-1)  # [batch, T, vocab]
        
        batch_size, T, vocab_size = log_probs.shape
        results = []
        
        for batch_idx in range(batch_size):
            # Initialize hypotheses: (log_prob, token_sequence, last_was_blank)
            # We track last_was_blank to handle repeated tokens correctly
            hypotheses = [(0.0, [], True)]
            
            for t in range(T):
                new_hypotheses = []
                
                for log_prob, tokens, last_blank in hypotheses:
                    # Get log probs for current time step
                    probs = log_probs[batch_idx, t]  # [vocab]
                    
                    # Try all possible next tokens
                    for token_idx in range(vocab_size):
                        lp = probs[token_idx].item()
                        new_log_prob = log_prob + lp
                        
                        if token_idx == self.blank_idx:
                            # Blank token: don't add to sequence
                            new_tokens = tokens.copy()
                            new_hypotheses.append((new_log_prob, new_tokens, True))
                        else:
                            # Non-blank token
                            if tokens and tokens[-1] == token_idx and not last_blank:
                                # Same as last non-blank token and last wasn't blank
                                # Don't add duplicate
                                new_tokens = tokens.copy()
                            else:
                                # Add new token
                                new_tokens = tokens + [token_idx]
                            new_hypotheses.append((new_log_prob, new_tokens, False))
                
                # Keep only top beam_width hypotheses
                new_hypotheses.sort(key=lambda x: x[0], reverse=True)
                hypotheses = new_hypotheses[:beam_width]
            
            # Get best hypothesis
            best_tokens = hypotheses[0][1]
            text = self._tokens_to_text(best_tokens)
            results.append(text.lower())
        
        return results
    
    # ==================== Task 4: Shallow Fusion with LM ====================
    
    def beam_search_with_lm(self,
                           logits: torch.Tensor,
                           beam_width: int = 10,
                           alpha: float = 0.5,
                           beta: float = 0.0) -> List[str]:
        """
        Beam search with shallow fusion of language model.
        Score = log_p_acoustic + alpha * log_p_lm + beta * num_words
        
        Args:
            logits: Tensor of shape [batch_size, seq_len, vocab_size]
            beam_width: Number of hypotheses to keep
            alpha: LM weight
            beta: Word count bonus
            
        Returns:
            List of decoded strings
        """
        # Apply temperature and log_softmax
        logits = logits / self.temperature
        log_probs = torch.log_softmax(logits, dim=-1)
        
        batch_size, T, vocab_size = log_probs.shape
        results = []
        
        for batch_idx in range(batch_size):
            # Hypothesis structure:
            # (acoustic_score, token_sequence, text_so_far, num_words, last_blank)
            # We store text to avoid recomputing LM score from tokens
            hypotheses = [(0.0, [], "", 0, True)]
            
            for t in range(T):
                new_hypotheses = {}
                
                for aco_score, tokens, text, num_words, last_blank in hypotheses:
                    probs = log_probs[batch_idx, t]  # [vocab]
                    
                    for token_idx in range(vocab_size):
                        lp = probs[token_idx].item()
                        new_aco = aco_score + lp
                        
                        if token_idx == self.blank_idx:
                            # Blank token
                            key = (tuple(tokens), text, num_words)
                            if key not in new_hypotheses or new_aco > new_hypotheses[key][0]:
                                new_hypotheses[key] = (new_aco, tokens, text, num_words, True)
                        else:
                            # Non-blank token
                            char = self.idx2char[token_idx]
                            
                            if token_idx == 4:  # '|' - word separator
                                # End of word
                                new_tokens = tokens + [token_idx]
                                # Update text: add space if not first word
                                new_text = text + " "
                                new_num_words = num_words + 1
                                key = (tuple(new_tokens), new_text, new_num_words)
                                if key not in new_hypotheses or new_aco > new_hypotheses[key][0]:
                                    new_hypotheses[key] = (new_aco, new_tokens, new_text, new_num_words, False)
                            else:
                                # Regular character
                                new_tokens = tokens + [token_idx]
                                
                                # Check for duplicate consecutive chars
                                if tokens and tokens[-1] == token_idx and not last_blank:
                                    # This is a duplicate, don't add to text
                                    key = (tuple(tokens), text, num_words)
                                    if key not in new_hypotheses or new_aco > new_hypotheses[key][0]:
                                        new_hypotheses[key] = (new_aco, tokens, text, num_words, False)
                                else:
                                    # Add character to current word
                                    if char == '|':
                                        # Already handled above
                                        pass
                                    else:
                                        # Append character to last word
                                        if text and text[-1] == ' ':
                                            new_text = text + char
                                        else:
                                            new_text = text + char
                                        key = (tuple(new_tokens), new_text, num_words)
                                        if key not in new_hypotheses or new_aco > new_hypotheses[key][0]:
                                            new_hypotheses[key] = (new_aco, new_tokens, new_text, num_words, False)
                
                # Convert to list and compute total scores with LM
                scored_hypotheses = []
                for (tokens_tuple, text, num_words), (aco_score, tokens, text, num_words, last_blank) in new_hypotheses.items():
                    # Compute LM score for current text
                    if text.strip():
                        lm_score = self._lm_score(text.strip())
                    else:
                        lm_score = 0.0
                    
                    total_score = aco_score + alpha * lm_score + beta * num_words
                    scored_hypotheses.append((total_score, aco_score, list(tokens), text, num_words, last_blank))
                
                # Keep top beam_width
                scored_hypotheses.sort(reverse=True, key=lambda x: x[0])
                hypotheses = [(aco, tokens, text, words, last_blank) 
                             for _, aco, tokens, text, words, last_blank in scored_hypotheses[:beam_width]]
            
            # Get best hypothesis
            best = max(hypotheses, key=lambda x: x[0] + alpha * self._lm_score(x[2].strip()) + beta * x[3])
            text = best[2].strip()
            results.append(text.lower())
        
        return results
    
    # ==================== Task 6: LM Rescoring ====================
    
    def lm_rescore(self,
                  logits: torch.Tensor,
                  beam_width: int = 100,
                  alpha: float = 0.5,
                  beta: float = 0.0) -> List[str]:
        """
        Second-pass LM rescoring: first generate hypotheses with beam search,
        then rescore them with LM.
        
        Args:
            logits: Tensor of shape [batch_size, seq_len, vocab_size]
            beam_width: Number of hypotheses to generate
            alpha: LM weight
            beta: Word count bonus
            
        Returns:
            List of decoded strings
        """
        # First, generate hypotheses with beam search (no LM)
        logits_scaled = logits / self.temperature
        log_probs = torch.log_softmax(logits_scaled, dim=-1)
        
        batch_size, T, vocab_size = log_probs.shape
        results = []
        
        for batch_idx in range(batch_size):
            # Generate hypotheses with beam search
            hypotheses = [(0.0, [], True)]  # (acoustic_score, tokens, last_blank)
            
            for t in range(T):
                new_hypotheses = {}
                
                for aco_score, tokens, last_blank in hypotheses:
                    probs = log_probs[batch_idx, t]
                    
                    for token_idx in range(vocab_size):
                        lp = probs[token_idx].item()
                        new_aco = aco_score + lp
                        
                        if token_idx == self.blank_idx:
                            key = tuple(tokens)
                            if key not in new_hypotheses or new_aco > new_hypotheses[key][0]:
                                new_hypotheses[key] = (new_aco, tokens, True)
                        else:
                            # Non-blank token
                            if tokens and tokens[-1] == token_idx and not last_blank:
                                # Duplicate, don't add
                                key = tuple(tokens)
                                if key not in new_hypotheses or new_aco > new_hypotheses[key][0]:
                                    new_hypotheses[key] = (new_aco, tokens, False)
                            else:
                                new_tokens = tokens + [token_idx]
                                key = tuple(new_tokens)
                                if key not in new_hypotheses or new_aco > new_hypotheses[key][0]:
                                    new_hypotheses[key] = (new_aco, new_tokens, False)
                
                # Keep top beam_width
                sorted_hyp = sorted(new_hypotheses.items(), key=lambda x: x[1][0], reverse=True)[:beam_width]
                hypotheses = [(score, tokens, last_blank) for (_, (score, tokens, last_blank)) in sorted_hyp]
            
            # Now rescore each hypothesis with LM
            best_score = -float('inf')
            best_text = ""
            
            for aco_score, tokens, _ in hypotheses:
                text = self._tokens_to_text(tokens)
                if text:
                    words = text.split()
                    num_words = len(words)
                    lm_score = self._lm_score(text)
                    total_score = aco_score + alpha * lm_score + beta * num_words
                    
                    if total_score > best_score:
                        best_score = total_score
                        best_text = text
            
            results.append(best_text.lower())
        
        return results
    
    # ==================== Helper method for getting all hypotheses ====================
    
    def get_all_hypotheses(self, logits: torch.Tensor, beam_width: int = 100) -> List[Tuple[float, List[int]]]:
        """
        Generate all beam search hypotheses (without LM) for rescoring.
        
        Args:
            logits: Tensor of shape [batch_size, seq_len, vocab_size]
            beam_width: Number of hypotheses to keep
            
        Returns:
            List of (acoustic_score, token_sequence) for first batch item
        """
        logits = logits / self.temperature
        log_probs = torch.log_softmax(logits, dim=-1)
        
        T, vocab_size = log_probs.shape[1], log_probs.shape[2]
        hypotheses = [(0.0, [], True)]
        
        for t in range(T):
            new_hypotheses = {}
            
            for aco_score, tokens, last_blank in hypotheses:
                probs = log_probs[0, t]
                
                for token_idx in range(vocab_size):
                    lp = probs[token_idx].item()
                    new_aco = aco_score + lp
                    
                    if token_idx == self.blank_idx:
                        key = tuple(tokens)
                        if key not in new_hypotheses or new_aco > new_hypotheses[key][0]:
                            new_hypotheses[key] = (new_aco, tokens, True)
                    else:
                        if tokens and tokens[-1] == token_idx and not last_blank:
                            key = tuple(tokens)
                            if key not in new_hypotheses or new_aco > new_hypotheses[key][0]:
                                new_hypotheses[key] = (new_aco, tokens, False)
                        else:
                            new_tokens = tokens + [token_idx]
                            key = tuple(new_tokens)
                            if key not in new_hypotheses or new_aco > new_hypotheses[key][0]:
                                new_hypotheses[key] = (new_aco, new_tokens, False)
            
            sorted_hyp = sorted(new_hypotheses.items(), key=lambda x: x[1][0], reverse=True)[:beam_width]
            hypotheses = [(score, tokens, last_blank) for (_, (score, tokens, last_blank)) in sorted_hyp]
        
        return [(score, tokens) for score, tokens, _ in hypotheses]