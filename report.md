***
## i. Model Used

The primary model used for this project was **OpenAI's CLIP (Contrastive Language-Image Pre-Training)**, specifically the `ViT-B/32` variant.

This pre-trained Vision-Language Model (VLM) was chosen for its powerful zero-shot capabilities. It understands both images and text in a shared embedding space, allowing it to classify videos based on natural language prompts without requiring direct training on the specific action classes.

***
## ii. Experimental Setup

The experiment was conducted on a subset of **15 classes** from the **UCF101 dataset**, utilizing the official train/test splits for evaluation.

1.  **Zero-Shot Evaluation**: The core task involved evaluating the pre-trained CLIP model's ability to classify actions. This was done by creating text prompts for each class (e.g., *"a photo of a person Archery"*) and matching them against video representations using cosine similarity.
2.  **Temporal Input Comparison**: To assess the importance of motion, two methods were compared:
    * **Single Frame**: Using only the middle frame to represent the video.
    * **Short Clip**: Extracting a 2-second clip from the center of the video and averaging the features from 8 frames within that clip.
3.  **Few-Shot Fine-Tuning**: A targeted fine-tuning experiment was conducted on the 5 classes with the lowest initial accuracy. To prevent catastrophic forgetting, the CLIP model's vision backbone was **frozen**, and only the final linear projection layer was trained.

***
## iii. Accuracy and Example Outputs

* **Zero-Shot Accuracy**: The zero-shot approach achieved high performance on the test set. The key finding was that using a **short clip (temporal input) was more accurate** than using a single static frame, demonstrating the model's ability to leverage motion context.

* **Fine-Tuning Results**: The targeted fine-tuning process on the 5 worst-performing classes was successful, with the training loss decreasing over epochs. This showed the viability of specializing the model without destroying its pre-trained knowledge.

* **Example Outputs**:
    * **Confusion Matrix**: The matrices clearly visualized the model's performance, confirming high accuracy on most classes while pinpointing specific areas of confusion, such as between `Basketball` and `BasketballDunk`.
    * **t-SNE Visualization**: The 2D plot of the embedding space showed distinct, well-separated clusters for most action classes. This visually confirmed that the model has learned a powerful and semantically meaningful representation of the different actions.

***
## iv. Limitations or Observations

* **Catastrophic Forgetting**: A critical observation was that naively fine-tuning the entire model on a small dataset resulted in a catastrophic drop in performance. The targeted approach of **freezing most layers** is essential for effective fine-tuning of large VLMs.
* **Simple Temporal Modeling**: The method of averaging frame features from a clip is a basic way to incorporate temporal information. More advanced video-specific architectures could further improve performance.
* **Prompt Sensitivity**: The performance of zero-shot classification is known to be sensitive to the exact wording of the text prompts. The simple prompt structure used here could be further optimized for better results.