from typing import Any

import torch

from rau.tools.torch.model_interface import ModelInterface
from rau.tasks.common.training_loop import (
    add_training_loop_arguments as common_add_training_loop_arguments,
    get_training_loop_kwargs,
    TrainingLoop
)
from .batching import group_into_batches
from .data import VocabularyContainer
from lrayuela.base.symbol import Sym


def add_training_loop_arguments(parser):
    common_add_training_loop_arguments(parser,
        max_tokens_per_batch_help=
        'The maximum number of tokens allowed per batch. This puts a limit on '
        'the number of elements included in a single batch tensor, including '
        'BOS, EOS, and padding tokens. If a single example exceeds the limit, '
        'it is not discarded, but included in a batch by itself.'
    )

Example = torch.Tensor
PreparedBatch = tuple[torch.Tensor, torch.Tensor]

class LanguageModelingTrainingLoop(TrainingLoop[
    Example,
    PreparedBatch,
    VocabularyContainer
]):

    def get_validation_metric_name(self):
        return 'cross_entropy_per_token'

    def get_validation_metric_mode(self):
        return 'min'

    def generate_batches(self, examples, max_tokens):
        return generate_batches(examples, max_tokens)

    def get_prepared_batch_info(self, prepared_batch):
        model_input, correct_target = prepared_batch
        return dict(
            input_size=tuple(model_input.size()),
            output_size=tuple(correct_target.size())
        )

    def log_failed_batch(self, vocabulary, batch, info, console_logger, event_logger):
        if info is not None:
            console_logger.info(f'  input size: {info.get("input_size")}')
            console_logger.info(f'  output size: {info.get("output_size")}')
        tokens = sum(map(len, batch))
        console_logger.info(f'  tokens: {tokens}')
        lengths = list(map(len, batch))
        console_logger.info(f'  sequence lengths: {lengths}')
        token_strs = [
            [vocabulary.input_vocab.to_string(w) for w in x]
            for x in batch
        ]
        sequences_str = '\n'.join(' '.join(x) for x in token_strs)
        console_logger.info(f'  sequences:\n{sequences_str}')
        return dict(
            **(info or {}),
            examples=token_strs
        )

    def get_loss(self, model, model_interface, prepared_batch):
        return get_cross_entropy_loss(self, model, model_interface, prepared_batch)

    def evaluate_batch(self, model, model_interface, prepared_batch):
        return evaluate_batch(model, model_interface, prepared_batch)


class LanguageModelingTrainingLoopKL(LanguageModelingTrainingLoop):
    def _get_dists_for_batch(self, model, model_interface, prepared_batch, vocabulary):
        """Get distributions from both the model and the automaton for KL calculation.
        
        Args:
            model: The neural language model
            model_interface: Interface for the model
            prepared_batch: Prepared batch of sequences
            vocabulary: The vocabulary container for converting between IDs and symbols
            
        Returns:
            tuple: (model_dists, automaton_dists, valid_positions_mask)
                - model_dists: Model's probability distributions [batch_size, seq_len, vocab_size]
                - automaton_dists: Automaton's probability distributions [batch_size, seq_len, vocab_size]
                - valid_positions_mask: Mask for positions with valid tokens [batch_size, seq_len]
        """
        model_input, correct_target = prepared_batch
        pad_index = model_interface.output_padding_index
        
        # Get logits from the model and convert to probabilities
        logits = model_interface.get_logits(model, model_input)  # [batch, seq_len, vocab_size]
        model_probs = torch.nn.functional.softmax(logits, dim=2)
        
        # Create mask for valid positions (non-padding)
        valid_positions = (correct_target != pad_index)

        # Get automaton distributions for each sequence in the batch
        batch_size, seq_len = model_input.size()[:2]
        vocab_size = model_probs.size(2)
        
        # Create tensor to hold automaton distributions
        automaton_dists = torch.zeros_like(model_probs)
        
        # Process each sequence in the batch
        for b in range(batch_size):
            # Get the input sequence (without padding)
            seq_length = valid_positions[b].sum().item()
            if seq_length == 0:
                continue
                
            # Convert token IDs to symbols using vocabulary
            input_ids = model_input[b, :seq_length].cpu().tolist()
            input_symbols = []
            
            try:
                # Transformer
                exclude = (vocabulary.input_vocab.bos_index, )
            except:
                #exclude = (vocabulary.input_vocab.eos_index, )
                exclude = []

            for token_id in input_ids:
                if token_id in exclude:
                    continue
                # Convert token ID to automaton symbol using vocabulary
                symbol_str = vocabulary.input_vocab.to_string(token_id)
                input_symbols.append(symbol_str)

            # Get distributions from the automaton
            automaton_distributions = self.automaton.get_state_symbol_distribution(input_symbols)
            
            # Convert automaton distributions to tensor format
            for pos, dist in enumerate(automaton_distributions):
                if pos >= seq_len:
                    break
                    
                # Skip if this is a padding position
                if not valid_positions[b, pos]:
                    continue
                    
                # Fill in the distribution from the automaton
                for sym, prob in dist.items():
                    # Convert automaton symbol to token ID
                    if sym.value == '<eos>':
                        token_id = vocabulary.input_vocab.eos_index
                    else:
                        # Handle other types of symbols
                        sym_str = str(sym.value)
                        try:
                            token_id = vocabulary.input_vocab._first._string_list.index(sym_str)
                        except:
                            token_id = vocabulary.output_vocab._first._string_list.index(sym_str)

                    automaton_dists[b, pos, token_id] = prob
        
        row_sum = automaton_dists.sum(dim=2, keepdim=False)   # [B, T]
        valid_positions = valid_positions & (row_sum > 0)

        return model_probs, automaton_dists, valid_positions

    def get_loss_reverse_kl(self, model, model_interface, prepared_batch):
        """
        Return per-sequence reverse-KL loss against the automaton plus the
        number of (non-PAD) tokens.  Signature is identical to the original
        cross-entropy / forward-KL version so the surrounding training code
        needs no changes.
        """
        vocab = self.vocabulary                     # convenience alias

        # ----------------------------------------------------------------------
        # 1. Probability distributions from model and automaton
        # ----------------------------------------------------------------------
        model_p, auto_p, valid_pos = self._get_dists_for_batch(
            model, model_interface, prepared_batch, vocab
        )                                   # shapes: [B, T, V]

        # Ensure every automaton distribution is a *proper* prob. vector
        # (sum = 1).  Positions with sum==0 can only happen if neither
        # state→symbol table nor padding provided any prob.; we give them a
        # tiny uniform distribution so the KL stays finite.
        row_sum = auto_p.sum(dim=2, keepdim=True)          # [B, T, 1]
        auto_p  = torch.where(
            row_sum > 0.0,
            auto_p / row_sum,                              # normalise
            torch.full_like(auto_p, 1.0 / auto_p.size(2))  # uniform fallback
        )

        # ----------------------------------------------------------------------
        # 2. Reverse KL :  ∑_v  p_model(v) · log[ p_model(v) / p_auto(v) ]
        # ----------------------------------------------------------------------
        eps_model = 1e-10                     # keep log well-defined
        eps_auto  = 1e-4                      # pushes model away from illegal
        safe_auto = torch.clamp(auto_p, min=eps_auto)

        log_ratio = torch.log(model_p + eps_model) - torch.log(safe_auto)
        kl_tokens = (model_p * log_ratio).sum(dim=2)        # [B, T]

        # ----------------------------------------------------------------------
        # 3. Mask out PAD positions and average per sequence
        # ----------------------------------------------------------------------
        masked_kl   = kl_tokens * valid_pos.float()         # [B, T]
        kl_per_seq  = masked_kl.sum(dim=1)                  # [B]
        num_symbols = valid_pos.sum().item()                # scalar

        # import random
        # rnd = random.randint(0,100)
        # if rnd == 0:
        #     breakpoint()

        return kl_per_seq, num_symbols

    def get_loss(self, model, model_interface, prepared_batch):
        """
        Per-sequence *forward* KL  KL(auto || model)  plus the number of
        non-PAD tokens.
        """
        vocab = self.vocabulary

        # ────────────────────────────────────────────────────────────────
        # 1. probability tensors  [B, T, V]  and valid-position mask [B,T]
        # ────────────────────────────────────────────────────────────────
        model_p, auto_p, valid_pos = self._get_dists_for_batch(
            model, model_interface, prepared_batch, vocab
        )

        # normalise every automaton row; if a row is all-zero (padding or
        # dead-state) leave it zero — it will be masked out next.
        row_sum = auto_p.sum(dim=2, keepdim=True)          # [B,T,1]
        non_empty = row_sum > 0.0
        auto_p = torch.where(
            non_empty,
            auto_p / row_sum.clamp(min=1e-12),
            auto_p                                 # keep zeros as zeros
        )
        valid_pos = valid_pos & non_empty.squeeze(-1)

        # ────────────────────────────────────────────────────────────────
        # 2. forward KL  Σ p_auto * log(p_auto / p_model)
        #    (no ε needed on p_auto; rows with 0 are masked)
        # ────────────────────────────────────────────────────────────────
        eps_model = 1e-10                       # just a numerical guard
        log_ratio = torch.log(auto_p + 1e-12) - torch.log(model_p + eps_model)
        kl_tokens = (auto_p * log_ratio).sum(dim=2)         # [B,T]

        # ────────────────────────────────────────────────────────────────
        # 3. mask PAD / empty rows and sum per sequence
        # ────────────────────────────────────────────────────────────────
        kl_tokens = kl_tokens * valid_pos.float()
        kl_per_seq = kl_tokens.sum(dim=1)                    # [B]
        num_symbols = valid_pos.sum().item()

        return kl_per_seq, num_symbols


def generate_batches(examples, max_tokens):
    return group_into_batches(examples, lambda b, n: b * n <= max_tokens)

def get_cross_entropy_loss(training_loop, model, model_interface, prepared_batch):
    cross_entropy, num_symbols = get_cross_entropy(
        model,
        model_interface,
        prepared_batch,
        reduction='none',
        label_smoothing_factor=training_loop.label_smoothing_factor
    )
    loss_numerators = torch.sum(cross_entropy, dim=1)
    return loss_numerators, num_symbols

def evaluate_batch(model, model_interface, prepared_batch):
    cross_entropy, num_symbols = get_cross_entropy(
        model,
        model_interface,
        prepared_batch,
        reduction='sum',
        label_smoothing_factor=0.0
    )
    return {
        'cross_entropy_per_token' : (cross_entropy.item(), num_symbols)
    }

def get_cross_entropy(
    model: torch.nn.Module,
    model_interface: ModelInterface,
    prepared_batch: list[Example],
    reduction: str,
    label_smoothing_factor: float
) -> tuple[torch.Tensor, int, dict[str, Any]]:
    model_input, correct_target = prepared_batch
    pad_index = model_interface.output_padding_index
    logits = model_interface.get_logits(model, model_input)
    cross_entropy = torch.nn.functional.cross_entropy(
        logits.permute(0, 2, 1),
        correct_target,
        ignore_index=pad_index,
        reduction=reduction,
        label_smoothing=label_smoothing_factor
    )
    num_symbols = torch.sum(correct_target != pad_index).item()
    
    return cross_entropy, num_symbols
