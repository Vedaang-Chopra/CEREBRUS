# Edge Glass System Architecture — maintained graph

Current architecture of `edge_glass_modular/`, grouped into the three maintained surfaces:
the **canonical alignment path**, the **VLM / TRM-inspired refinement path**, and the
**retained experimental audio support**. Retired subsystems (fusion, decoder Path C /
`MultimodalProjector`, standalone `TRMDecoder`) are intentionally absent.

Frozen/trainable styling: grey dashed = frozen; blue solid = trainable.

## 1. Canonical alignment path

```mermaid
graph TB
    classDef frozen fill:#e8e8e8,stroke:#666666,stroke-width:2px,stroke-dasharray: 5 5
    classDef trainable fill:#d6e9f8,stroke:#2e5c8a,stroke-width:2px
    classDef input fill:#ffffff,stroke:#333333,stroke-width:1.5px
    classDef head fill:#f0f0f0,stroke:#555555,stroke-width:2px

    I(Image):::input --> VE[CLIP ViT-L/14-336<br/>frozen]:::frozen
    VE --> Patches[Patch tokens<br/>B, T_img, 1024]:::frozen

    subgraph Canonical_Vision_Path ["canonical alignment path (use_perceiver = false)"]
        direction TB
        LP[Linear projection<br/>1024 to 4096]:::trainable
        Pool[SimpleAttentionPooling<br/>learnable, over patch tokens]:::trainable
        VN[L2 Norm]:::trainable
        VM[MRL prefix representations<br/>2048 / 1024 / 512 / 256 / 128]:::head

        LP --> Pool --> VN --> VM
    end

    subgraph Perceiver_Variant ["Perceiver variant (experimental)"]
        direction TB
        LP2[MLP adapter<br/>1024 to 512]:::trainable
        PR[PerceiverResampler<br/>executed V3 architecture<br/>K latents x 512]:::trainable
        PM[Mean pool over latents<br/>B, 512]:::trainable
        FP[LayerNorm + Linear<br/>512 to projection_dim]:::trainable
        VN2[L2 Norm]:::trainable
        VM2[MRL prefixes optional]:::head

        LP2 --> PR --> PM --> FP --> VN2 --> VM2
    end

    T(Text):::input --> TE[SentenceTransformer MPNet<br/>frozen]:::frozen
    TP[Text projection<br/>768 to 4096]:::trainable
    TN[L2 Norm]:::trainable
    TM[MRL prefixes<br/>2048 / 1024 / 512 / 256 / 128]:::head
    TE --> TP --> TN --> TM

    Patches --> LP
    Patches --> LP2

    VM --> Align[Alignment losses<br/>symmetric InfoNCE + MRL]:::head
    VM2 --> Align
    TM --> Align
```

The **canonical** path (`configs/pixmo_alignment.yaml`, attention pooling, no perceiver) produced the
authoritative retained alignment results in `results/alignment/alignment_eval_final/`.

The **Perceiver variant** (`src/encoders/perceiver.py`) now implements the executed V3-training
architecture. The historical Perceiver metric in the final CSVs is **provenance-compromised**
(historical evaluation architecture differed from training; `strict=False` checkpoint loading) and must
not be read as an architecture verdict; a controlled rerun is required.

## 2. VLM path — Qwen + TRM-inspired iterative refinement

```mermaid
graph TB
    classDef frozen fill:#e8e8e8,stroke:#666666,stroke-width:2px,stroke-dasharray: 5 5
    classDef trainable fill:#d6e9f8,stroke:#2e5c8a,stroke-width:2px
    classDef input fill:#ffffff,stroke:#333333,stroke-width:1.5px

    IMG[Image input]:::input --> AE[Aligned vision encoder<br/>canonical alignment path<br/>frozen]:::frozen
    AE -->|aligned visual tokens| VP[vision_proj<br/>trainable]:::trainable
    VP -->|prefix embeddings| QWEN[Qwen2.5-3B + LoRA<br/>backbone never moved after load]:::trainable
    TXT[Question tokens]:::input --> QWEN
    QWEN -->|final hidden states| REF[TRM-inspired iterative refinement<br/>shared TRMLayers<br/>fixed 4 y/z iterations]:::trainable
    REF -->|residual add| LM[LM head]:::trainable

    style AE fill:#e1f5fe,stroke:#01579b,stroke-width:2px
```

Precisely:

- Refinement reuses a **shared stack of `TRMLayer`s**; the y/z loop runs a **fixed 4 iterations** over
  the Qwen final hidden states, then the refined state is residual-added before the LM head.
- This is **TRM-inspired iterative refinement**, *not* a faithful TinyRecursiveModels implementation
  (no adaptive halting, fixed depth). **No standalone `TRMDecoder` exists anymore.**
- No quantitative VLM improvement is claimed in the retained evidence.

## 3. Retained experimental audio support

```mermaid
graph TB
    classDef frozen fill:#e8e8e8,stroke:#666666,stroke-width:2px,stroke-dasharray: 5 5
    classDef trainable fill:#d6e9f8,stroke:#2e5c8a,stroke-width:2px
    classDef input fill:#ffffff,stroke:#333333,stroke-width:1.5px

    A(Audio waveform):::input --> WH[Whisper encoder<br/>frozen]:::frozen
    WH --> AP[Projection / optional Perceiver<br/>trainable]:::trainable
    AP --> AN[L2 Norm + optional MRL prefixes]:::trainable
    AN --> TL[Tri-modal loss scaffolding<br/>vision-text + audio-text + vision-audio<br/>EXPERIMENTAL]:::trainable
```

Audio is **retained experimental support** (`src/encoders/audio.py`, tri-modal loss in
`src/models/losses.py`). No retained experiment demonstrates successful tri-modal alignment or tri-modal
VLM training; do not infer audio results from the vision-text results. Fusion is retired and not shown.
