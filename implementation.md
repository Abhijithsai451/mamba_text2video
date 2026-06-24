# Mamba-2 Text to Video - Implementation Plan

## Phase 1: Video Patching & Spatio-Temporal Tokenization
* **Step 1.1: Environment Setup**
  * Install macOS-compatible PyTorch and torchvision. Install an Apple Silicon optimized Mamba library or write custom Mamba-2 loops using native PyTorch targeting the MPS (`device = 'mps'`) backend. Verify that MPS acceleration is active via `torch.backends.mps.is_available()`. ✅ 
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
---

## GIST
**Phase 1: The "Translator" Phase (Data Preprocessing)**
* The Gist: A Mamba model only understands numbers structured as a single 1D sequence (like a string of text). A video is a giant 3D block of pixels (Frames × Width × Height).

* What we are doing: We are taking a video file from the CinePile dataset, picking out a few evenly spaced frames so we don't crash your Mac's memory, running those frames through a pre-trained "Image Eyeball" (like a frozen CLIP or ViT model) to turn pixels into clean vector math, and flattening it out. By the end of this phase, your video looks exactly like a list of words to a computer.

**Phase 2: The "Frankenstein" Phase (The Backbone)**
* The Gist: We need the model to look at the video and the user's question at the exact same time.

* What we are doing: We take the text question, turn it into text vectors, and literally glue it to the end of the video vectors we created in Phase 1. Now we have one long multi-modal chain: [Video Tokens + Text Tokens]. We push this joint chain through a basic Mamba-2 block and attach a "Language Head" on top so it can output standard words.

**Phase 3: The "Sanity Check" Phase (Alignment Setup)**
* The Gist: Before scaling up to 100,000+ tokens, we need to make sure our code actually works and the model is capable of learning.

* What we are doing: We make a "cheat sheet" (mask) that tells the model: Don't try to guess what the next video frame looks like; only try to guess the text answer. Then, we give it a tiny sandbox of just 5 videos. If our training math is correct, the model will successfully overfit and learn to answer those 5 videos perfectly. If the loss drops to zero, we know our foundation is solid.

**Phase 4: The "Squeeze" Phase (Mac Scaling & Optimization)**
* The Gist: Processing 100k+ tokens normally requires a massive server rig. We need to make it fit on your M1 Pro Apple Silicon.

* What we are doing: We use clever memory tricks. We pools neighboring visual pixels together so they take up less space, use "Activation Checkpointing" (which discards intermediate calculations during the forward pass and recalculates them later to free up memory), and use Apple-optimized half-precision math (FP16). This forces the extreme long-context sequence to sit nicely inside your Mac's Unified Memory without crashing.

**Phase 5: The "Graduation" Phase (Fine-Tuning & Inference)**
* The Gist: Now that the machine works perfectly on your Mac without breaking, it's time to actually train it to be smart.

* What we are doing: We hook up the real, full CinePile dataset we downloaded. We train the model over these real movie clips and questions. Finally, we test it in an "inference loop" by giving it a brand-new video it has never seen before, allowing it to generate text responses word-by-word, and checking if its answers make sense.
---