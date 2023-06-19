Disclaimer: This is my research code, hence there could be additional lines commented out, or things like training epoch is way too high. Do not treat it as a production level code with proper polish and documentation. Also, check the code carefully if using on something important, as it might break things.

## Experiments Structure Description

- `clip_sbir` -- performs SBIR using CLIP with ViT architecture. Inside the ViT architecture, we train only the `LayerNorm` parameters while keeping the rest frozen.
-  `image-captioning` -- performs image captioning or sketch captioning. The architecture is an attention based pipeline, similar to Soft Attention in `Show, Attend, and Tell` paper: [https://arxiv.org/abs/1502.03044](https://arxiv.org/abs/1502.03044).
-  `photo2sketch` -- this is an additional experimental code (not part of the main paper used to generate sketches from photo. Note, as mentioned in the supplementary, the generated sketches are not good looking, but that was never the point either. I wanted to see if H-Decoder (hierarchical decoder) can act as an auxiliary loss.
-  `sbir_baseline` -- performs SBIR using the standard approaches (e.g., a VGG-backbone, or Inception-backbone etc.)
-  `stbir_baseline` -- performs sketch + text based image retrieval. The combination of sketch and text can be either concatenation or addition as found in [networks.py](https://github.com/pinakinathc/fscoco/blob/c5fd18662e151e5642d3966c1299166416d2785f/src/stbir_baseline/network.py#L39)
-  `tbir_baseline` -- performs text based image retrieval. The text encoder is a simple GRU based sequential encoder. The image encoder is a VGG-backbone.



## Code Structure inside each Experiment

Under each experiment folder (e.g., `clip_sbir`, or `sbir_baseline`), the files are organised as follows:

- `main.py` -- Loads dataloader, models, trainer and calls the training + evaluation modules.
- `model.py` -- Imports various networks required for that experiment. Defines the forward pass, training code, loss functions, optimisers, evaluation code, inference code.
- `network.py` -- Defines the various networks used by that experiment -- e.g., image encoder, sketch encoder, text encoder, sketch+text fusion modules, etc.
- `options.py` -- Defines hyper-parameters for training such as batch size, number of workers, path to dataset, and experiment specific parameters.
- `dataloader.py` -- Defines dataloader for the specific experiment.
