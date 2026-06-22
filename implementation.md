# Mamba-2 Text to Video - Implementation Plan

## Phase 1: Video Patching & Spatio-Temporal Tokenization
* **Step 1.1: Environment Setup**
  * Install macOS-compatible PyTorch and torchvision. Install an Apple Silicon optimized Mamba library or write custom Mamba-2 loops using native PyTorch targeting the MPS (`device = 'mps'`) backend. Verify that MPS acceleration is active via `torch.backends.mps.is_available()`.
* **Step 1.2: Video Dataset & Sampling Pipeline**
  * Build a data loader to read raw video files locally using standard Mac decoding utilities. Implement a uniform temporal sampling strategy to extract a fixed number of frames across any video timeline, scaling and normalizing the images to a standard resolution.
* **Step 1.3: Frozen Vision Feature Extraction**
  * Integrate a pre-trained vision model (such as a Vision Transformer or CLIP image encoder). Explicitly freeze all parameters to eliminate gradient tracking, keeping the backbone on the unified memory system to prevent out-of-memory errors. Pass video frames through this backbone to obtain frame/patch-level feature embeddings.
* **Step 1.4: Linear Spatio-Temporal Projection**
  * Flatten the structural dimensions of the video feature tensor into a singular, sequential 1D sequence dimension. Implement a single linear layer to project the vision feature dimension into the exact hidden dimension required by the Mamba-2 layer.

---

## Phase 2: Toy Multimodal Mamba-2 Backbone Architecture
* **Step 2.1: Text Embedding Configuration**
  * Initialize a standard word embedding table using a standard tokenizer vocabulary size. Match the text embedding channel dimension exactly to the projection output dimension of your video layer.
* **Step 2.2: Sequence Concatenation Layer**
  * Create a forward pass utility that joins the incoming data. Concatenate the video feature sequence and the text token embedding sequence along the sequence length axis to form a unified, multi-modal input stream.
* **Step 2.3: Mamba-2 Block Integration**
  * Instantiate your MPS-compatible Mamba-2 block. Route the combined multimodal sequence through this block targeting the Mac GPU, confirming that the output tensor maintains the matching shape without triggering CPU-fallback warnings.
* **Step 2.4: Language Modeling Head**
  * Add a final linear classification layer on top of the Mamba output to project token embeddings back to vocabulary space for next-token prediction.

---

## Phase 3: Alignment & Multi-Turn Training Setup
* **Step 3.1: Cross-Entropy Target Masking**
  * Build a target-label masking scheme. Ensure video tokens are assigned the standard ignore index value for cross-entropy calculations, restricting gradient corrections strictly to the text response tokens.
* **Step 3.2: Local Overfitting Verification Loop**
  * Construct a basic training loop using an MPS-compatible optimizer (like AdamW). Isolate a tiny sandbox dataset of 5 mock video-text pairs and run multiple iterations to verify the mathematical soundness of the pipeline by making the loss drop cleanly to zero.

---

## Phase 4: Long-Context Scaling (100k+ Tokens)
* **Step 4.1: Hierarchical Video Downsampling**
  * Modify your token processing pipeline to handle extended video contexts. Implement spatial pooling or temporal frame aggregation to condense local window outputs, effectively lowering token counts to avoid over-allocating Apple Silicon's Shared Unified Memory.
* **Step 4.2: Activation Checkpointing**
  * Wrap your model blocks with native activation checkpointing tools. This discards intermediate states during the forward pass and recomputes them during the backward pass, conserving memory.
* **Step 4.3: Mixed Precision Execution**
  * Scale your forward processing using automated mixed precision targeting `torch.float16` or `torch.bfloat16` (depending on whether your Mac hardware natively supports bfloat16 acceleration). This ensures optimal matrix math execution on the Apple GPU.
* **Step 4.4: Gradient Accumulation**
  * Maintain hardware batch sizes at a micro-batch size of 1 per device to allow extreme token lengths. Simulate larger, mathematically stable global training batch sizes by accumulating gradients over multiple steps before updating the model weights.

---

## Phase 5: Instruction Tuning & Evaluation Framework
* **Step 5.1: Real Dataset Integration**
  * Ingest a public text-video instruction dataset. Format your dataloader to convert these video-text alignment records into the token-target masking structure built in Phase 3.
* **Step 5.2: Autoregressive Inference & Generation**
  * Implement an autoregressive token generation loop for model evaluation. Track and append newly predicted tokens back into the sequence context stream during inference to measure output accuracy on long-form video comprehension tasks.