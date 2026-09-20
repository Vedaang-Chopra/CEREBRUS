
import torch
import torch.nn as nn
from typing import Optional, Tuple, List, Union
from src.decoders.qwen import QwenDecoder
from src.decoders.trm import TRMConfig, TRMLayer


# ---------------------------------------------------------------------------
# Device-placement policy (see AUDIT_LEDGER.md, device-placement audit)
#
# The Qwen backbone (self.qwen.model) may be:
#   - quantized with bitsandbytes (4-bit / 8-bit) — nn.Module.to() on a
#     quantized model is forbidden by HF/bitsandbytes after load;
#   - placed/sharded by a HF device_map — moving it breaks the dispatch map;
#   - PEFT-wrapped — PeftModel does not even expose a `.device` attribute, so
#     naive `hasattr(model, 'device')` detection silently falls back.
# Therefore the backbone is placed ONLY by HuggingFace at load time
# (device_map / quantization) and NEVER moved afterwards.
#
# The custom non-HF modules (vision_proj, down_proj/up_proj, trm_layers,
# z_init) are plain fp32 nn.Modules created on CPU. They are aligned ONCE at
# setup via `align_custom_modules()` — at inference by the calling script, and
# in training by accelerator.prepare() + an explicit setup-time alignment.
# forward()/generate() must not mutate modules; they may only move tensors.
# ---------------------------------------------------------------------------


def _first_param_device(module):
    """Device of the first parameter of ``module`` (None if it has none)."""
    for p in module.parameters():
        return p.device
    return None


class QwenVLM(nn.Module):
    """
    Qwen-based VLM with TRM-inspired iterative latent refinement.

    Architecture:
    1. Input (Image + Text) -> Aligned Embeddings
    2. Qwen Backbone (Pretrained) -> Hidden States
    3. TRM-inspired iterative refinement (on Hidden States) -> Refined States
    4. LM Head -> Logits

    Note: the refinement step is inspired by TinyRecursiveModels (weight-shared
    blocks applied over a fixed number of steps with a y/z state motif) but is
    NOT the TRM architecture itself — it has no halting, deep supervision,
    input injection, or carry detach (see AUDIT_LEDGER.md, TRM reconciliation).
    """
    def __init__(
        self,
        qwen_decoder: QwenDecoder,
        vision_token_dim: int = 4096,
        use_trm_recursion: bool = False,
        trm_config: Optional[TRMConfig] = None,
        num_trm_layers: int = 2,
        num_recursion_steps: int = 4,
    ):
        super().__init__()
        self.qwen = qwen_decoder
        self.use_trm_recursion = use_trm_recursion
        self.num_recursion_steps = num_recursion_steps
        
        # Get dimensions from Qwen
        self.hidden_dim = self.qwen.hidden_dim
        self.vocab_size = self.qwen.vocab_size
        
        # Vision projection: Vision Dim -> Qwen Hidden Dim
        self.vision_proj = nn.Linear(vision_token_dim, self.hidden_dim)
        
        if use_trm_recursion:
            if trm_config is None:
                # Default TRM config matching Qwen dims if possible, or small
                # If we want "Tiny", we might project
                trm_config = TRMConfig(
                    hidden_dim=self.hidden_dim, # Default to matching
                    num_layers=num_trm_layers, # Tiny depth from arg
                    num_heads=16, # Adjust as needed
                )
            
            self.trm_config = trm_config
            self.trm_hidden_dim = trm_config.hidden_dim
            
            # Projectors if dims mismatch
            if self.trm_hidden_dim != self.hidden_dim:
                self.down_proj = nn.Linear(self.hidden_dim, self.trm_hidden_dim)
                self.up_proj = nn.Linear(self.trm_hidden_dim, self.hidden_dim)
            else:
                self.down_proj = nn.Identity()
                self.up_proj = nn.Identity()
                
            # TRM Layers
            self.trm_layers = nn.ModuleList([
                TRMLayer(trm_config) for _ in range(trm_config.num_layers)
            ])

            # Latent token initialization (learnable)
            # z is the "reasoning state"
            self.z_init = nn.Parameter(torch.randn(1, 1, self.trm_hidden_dim) * 0.02)

    def align_custom_modules(self, dtype: Optional[torch.dtype] = None):
        """Align the custom (non-HF) modules to the Qwen backbone at SETUP time.

        Moves ONLY the plain modules this class owns (vision_proj, down_proj/
        up_proj, trm_layers, z_init) to the backbone's first-parameter device —
        never the backbone itself. Safe for quantized / device_map-managed /
        PEFT-wrapped backbones because the backbone is not touched. Call once
        after construction (and once after checkpoint loading); forward() and
        generate() never mutate modules.

        Args:
            dtype: target dtype. Defaults to the Qwen backbone's dtype so
                projected prefix embeddings concatenate cleanly with input
                embeddings.
        """
        backbone_device = _first_param_device(self.qwen)
        if dtype is None:
            dtype = self.qwen.model.dtype
        for name in ("vision_proj", "down_proj", "up_proj", "trm_layers"):
            module = getattr(self, name, None)
            if module is not None and module.parameters():
                module.to(device=backbone_device, dtype=dtype)
        if self.use_trm_recursion and hasattr(self, "z_init"):
            self.z_init.data = self.z_init.data.to(device=backbone_device, dtype=dtype)

    def forward(
        self,
        # Notebook-style arguments
        vision_tokens: Optional[torch.Tensor] = None,
        images: Optional[torch.Tensor] = None,  # Alias used in training loop
        question_ids: Optional[torch.Tensor] = None,
        answer_ids: Optional[torch.Tensor] = None,
        answer_mask: Optional[torch.Tensor] = None,
        # Standard HF arguments
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        prefix_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        # 0. Argument Resolution
        # Handle 'images' alias for vision_tokens
        if vision_tokens is None and images is not None:
            vision_tokens = images
            
        # Handle 'vision_tokens' -> 'prefix_embeds' mapping
        if prefix_embeds is None and vision_tokens is not None:
            # We need to project vision tokens first!
            # Project logic moved here to ensure we pass projected embeddings as prefix
            # Device-placement: modules are aligned at setup (align_custom_modules /
            # accelerator.prepare / HF device_map) — forward() only moves TENSORS,
            # never mutates modules (see device-placement policy, module header).
            vision_tokens = vision_tokens.to(device=self.vision_proj.weight.device,
                                             dtype=self.vision_proj.weight.dtype)
            prefix_embeds = self.vision_proj(vision_tokens)

        # Handle question_ids + answer_ids -> input_ids
        if input_ids is None and question_ids is not None and answer_ids is not None:
            assert question_ids is not None and answer_ids is not None  # for type checkers
            input_ids = torch.cat([question_ids, answer_ids], dim=1)
            
            # Build the text attention mask from the question/answer masks.
            # answer_mask semantics (from scripts/train/train_vlm_accelerate.py:collate_fn):
            #   1 = real answer token, 0 = answer padding. It is a padding mask, NOT
            #   a loss mask — loss masking is handled separately via labels below.
            # If a caller passed no answer_mask, treat every answer token as valid.
            if answer_mask is not None:
                if answer_ids is not None and answer_mask.device != answer_ids.device:
                    answer_mask = answer_mask.to(answer_ids.device)
                answer_keep = answer_mask.to(torch.long)
            else:
                answer_keep = torch.ones_like(answer_ids)
            attention_mask = torch.cat(
                [torch.ones_like(question_ids), answer_keep], dim=1
            )
            
            # Create labels if not provided
            # Standard VLM QA: -100 for question AND for answer padding, answer_ids for real tokens
            if labels is None:
                question_labels = torch.full_like(question_ids, fill_value=-100)
                answer_labels = answer_ids.masked_fill(answer_keep == 0, -100)
                labels = torch.cat([question_labels, answer_labels], dim=1)
                
        # Fallback if just question_ids (e.g. generation start)
        if input_ids is None and question_ids is not None:
            input_ids = question_ids

        # 1. Run Qwen Backbone
        # We need hidden states, not just logits
        outputs = self.qwen(
            input_ids=input_ids,
            attention_mask=attention_mask,
            prefix_embeds=prefix_embeds,
            labels=labels, # Pass labels to get 'loss' automatically from Qwen
            output_hidden_states=True,
            return_dict=True,
        )
        
        # If no input_ids provided (impossible?), check
        if input_ids is None:
             raise ValueError("forward() requires input_ids OR (question_ids + answer_ids)")

        # Qwen hidden state (last layer)
        # Use last_hidden_state: (B, Seq, Dim)  [Seq includes the visual prefix]
        qwen_hidden = outputs.hidden_states[-1]
        # Note: QwenDecoder returns hidden states over prefix+text; the manual TRM
        # loss path below pads labels with -100 over the prefix region.
        
        if not self.use_trm_recursion:
            # Return standard dictionary matching what notebook expects ('loss', 'logits', 'confidence')
            # But QwenDecoder output is CausalLMOutputWithPast (loss, logits, hidden_states...)
            # We should wrap it into dictionary if notebook expects dict
            
            # Check what notebook expects:
            # outputs = model(...); loss = outputs.loss
            # It expects an object with .loss attribute.
            # CausalLMOutputWithPast has .loss attribute.
            return outputs
            
        # 2. TRM Recursion Mode
        
        # We need to refine the answer part of the sequence.
        # But this is training (teacher forcing), so we have the whole sequence.
        # TRM strategy usually: Fixed Context (Image+Question) | Answer
        # We refine Answer representation using Context.
        
        # For simplicity in this "VLM QA" setup where we train on the whole sequence:
        # We can apply TRM to the whole sequence (refining everything) OR just answer.
        # Applying to whole sequence is easier implementation-wise for a Transformer.
        
        # Project down if needed
        # Device-placement: TRM modules are aligned at setup — this section only
        # moves TENSORS to wherever the (never-mutated) modules live.
        # If modules were never aligned (CPU) while qwen_hidden is on GPU, the
        # down_proj call itself will raise a device-mismatch error — that is the
        # explicit signal to call align_custom_modules() at setup, not a reason
        # to mutate modules mid-forward.
        x = self.down_proj(qwen_hidden.to(device=self._trm_device(), dtype=self._trm_dtype()))
        
        # Initialize latent z
        batch_size, seq_len, _ = x.shape
        z = self.z_init.expand(batch_size, seq_len, -1) # (B, Seq, trm_dim)
        
        # State 'y' initialized from x
        y = x
        
        # Iterate
        for _ in range(self.num_recursion_steps):
            combined = torch.cat([y, z], dim=1) # Concat along sequence dim -> (B, 2*Seq, Dim)
            
            hidden = combined
            for layer in self.trm_layers:
                hidden = layer(hidden)
            
            # Split back
            output_y, output_z = hidden.chunk(2, dim=1)
            
            # Update
            y = output_y
            z = output_z
            
        # Final projection up and residual connection
        y_up = self.up_proj(y) # (B, Seq, Qwen_Dim)
        # Move y_up back to qwen_hidden dtype/device for residual add
        y_up = y_up.to(device=qwen_hidden.device, dtype=qwen_hidden.dtype)
        final_hidden = qwen_hidden + y_up
        
        # LM Head
        # (lm_head belongs to the HF backbone — never moved; the input is cast
        # to its weight dtype instead.)
        if hasattr(self.qwen.model, "lm_head"):
            lm_head = self.qwen.model.lm_head
        else:
             lm_head = self.qwen.model.base_model.model.lm_head if hasattr(self.qwen.model, "base_model") else self.qwen.model.lm_head
        logits = lm_head(final_hidden.to(device=lm_head.weight.device, dtype=lm_head.weight.dtype))

        # Calculate loss if labels provided
        tr_loss = None
        if labels is not None:
            # Logits cover [prefix (if any) + text]; labels cover text only.
            # Pad labels with -100 over the prefix region so shapes align and the
            # prefix contributes nothing to the CE loss (instead of guessing by
            # comparing lengths, which mis-detects any batch-length padding).
            if prefix_embeds is not None:
                num_prefix = prefix_embeds.shape[1]
                # Logits shape: (B, num_prefix + seq_len, Vocab)
                # Labels shape: (B, seq_len)
                prefix_labels = torch.full(
                    (labels.shape[0], num_prefix),
                    fill_value=-100,
                    dtype=labels.dtype,
                    device=labels.device,
                )
                labels = torch.cat([prefix_labels, labels], dim=1)
            logits_for_loss = logits

            # Shift so that tokens < n predict n
            shift_logits = logits_for_loss[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = nn.CrossEntropyLoss()
            tr_loss = loss_fct(shift_logits.view(-1, self.vocab_size), shift_labels.view(-1))
            
        from transformers.modeling_outputs import CausalLMOutputWithPast
        return CausalLMOutputWithPast(
            loss=tr_loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
        )

    def _trm_device(self):
        """Device of the (setup-aligned) TRM stack — first parameter wins; Identity
        projections have no parameters, so fall back to trm_layers / z_init."""
        for module in (self.down_proj, self.up_proj):
            dev = _first_param_device(module) if module is not None else None
            if dev is not None:
                return dev
        dev = _first_param_device(self.trm_layers)
        if dev is not None:
            return dev
        return self.z_init.device

    def _trm_dtype(self):
        """Dtype of the (setup-aligned) TRM stack."""
        for module in (self.down_proj, self.up_proj):
            for p in module.parameters():
                return p.dtype
        for p in self.trm_layers.parameters():
            return p.dtype
        return self.z_init.dtype

    @torch.no_grad()
    def generate(
        self,
        vision_tokens: torch.Tensor,
        question_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        max_new_tokens: int = 32,
        temperature: float = 0.0,
        return_stats: bool = False,
        **kwargs,
    ):
        """Generate answers with TRM-inspired iterative refinement."""
        # Project vision
        # Device-placement: vision_proj is aligned at setup (align_custom_modules /
        # accelerator.prepare / HF device_map) — never mutated here. Tensors are
        # moved to wherever the aligned modules and question tokens live.
        vision_tokens = vision_tokens.to(
            device=self.vision_proj.weight.device,
            dtype=self.vision_proj.weight.dtype,
        )

        vision_emb = self.vision_proj(vision_tokens)
        
        # Ensure mask exists if not provided
        if attention_mask is None:
            # Default to all-valid if no pad_token_id, or mask padding
            pad_id = self.qwen.tokenizer.pad_token_id
            if pad_id is None: 
                attention_mask = torch.ones_like(question_ids)
            else:
                attention_mask = (question_ids != pad_id).long()
        
        if attention_mask.device != question_ids.device:
            attention_mask = attention_mask.to(question_ids.device)
        
        if not self.use_trm_recursion:
            # Baseline: standard Qwen generation
            outputs = self.qwen.generate(
                input_ids=question_ids,
                prefix_embeds=vision_emb,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                temperature=max(temperature, 0.01) if temperature > 0 else 1.0,
                do_sample=temperature > 0,
                pad_token_id=self.qwen.tokenizer.pad_token_id,
                eos_token_id=self.qwen.tokenizer.eos_token_id,
            )
            
            # Extract only generated part.
            # QwenDecoder.generate passes ONLY `inputs_embeds` (no `input_ids` kwarg) to
            # HF `model.generate()`. In transformers >= 4.x (verified empirically on the
            # installed 4.57.6), `_prepare_model_inputs` initializes input_ids as a
            # zero-length tensor (torch.ones((B, 0))) when only inputs_embeds are given,
            # so the returned sequences contain ONLY the newly generated tokens — the
            # prompt/prefix tokens are not echoed as token IDs. Slicing
            # `outputs[:, prompt_len:]` here therefore double-stripped and discarded the
            # generated answer. Return the output unchanged.
            generated_ids = outputs

            return generated_ids
        
        else:
            # TRM mode: autoregressive with recursion
            # Re-implementation of the loop ensuring TRM is applied at each step
            batch_size = vision_tokens.shape[0]
            generated_ids_list = []
            
            # Helper to pad answer_ids for first step
            pad_token_id = self.qwen.tokenizer.pad_token_id
            
            # Initialize current attention mask with question mask
            current_attention_mask = attention_mask
            
            for step in range(max_new_tokens):
                # Construct current answer sequence
                if len(generated_ids_list) == 0:
                     # FIRST STEP: Just use the question (and vision prefix)
                     # Do NOT add a pad token, as it confuses the model
                     current_input_ids = question_ids
                else:
                    answer_ids = torch.stack(generated_ids_list, dim=1)
                    # Ensure answer_ids is on the same device as question_ids
                    if answer_ids.device != question_ids.device:
                        answer_ids = answer_ids.to(question_ids.device)
                    
                    current_input_ids = torch.cat([question_ids, answer_ids], dim=1)
                    
                    # Update mask for the new token (valid)
                    # We need to extend mask by 1 for each step. 
                    # Optimization: Extend just once per step
                    # current_attention_mask was for previous step?
                    # No, let's just rebuild/extend it.
                    # Base mask size: Q
                    # Current input size: Q + step
                    # We need to append '1' to current_attention_mask

                    ones = torch.ones((batch_size, 1), device=current_attention_mask.device, dtype=attention_mask.dtype)
                    current_attention_mask = torch.cat([current_attention_mask, ones], dim=1)

                out = self.forward(
                    input_ids=current_input_ids,
                    attention_mask=current_attention_mask,
                    prefix_embeds=vision_emb
                )
                
                # logits = out.logits
                logits = out['logits'][:, -1, :]  # Last position
                
                if temperature > 0:
                    logits = logits / temperature
                    probs = torch.softmax(logits, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1).squeeze(1)
                else:
                    next_token = torch.argmax(logits, dim=-1)
                
                # Stop if EOS
                # But we have batch.
                generated_ids_list.append(next_token)
                
            return torch.stack(generated_ids_list, dim=1)
