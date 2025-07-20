# Action Recognition with Vision-Language Models

This project demonstrates zero-shot and few-shot action recognition on the **UCF101 dataset** using OpenAI's **CLIP** model. It includes methods for evaluation, selective fine-tuning, and visualizing model embeddings to compare different approaches.

## Features

  * **Zero-Shot Classification**: Evaluates a pre-trained CLIP model on action recognition without any prior training on the dataset.
  * **Few-Shot Fine-Tuning**: Implements a strategy to fine-tune the model on a small subset of underperforming classes.
  * **Temporal Analysis**: Compares the model's performance using single-frame input versus short-clip (temporal) input.
  * **Visualization**: Includes methods to visualize the model's embedding space with t-SNE and similarity scores with heatmaps.

## Dataset

This project uses the **UCF101 – Action Recognition Data Set**. You can download it from the [official website](https://www.crcv.ucf.edu/data/UCF101.php).

-----

## Setup and Installation

### 1\. Create and Activate Virtual Environment

First, create a virtual environment.

```bash
python -m venv venv
```

Activate the environment.

**On Windows:**

```powershell
.\venv\Scripts\activate
```

**On macOS / Linux:**

```bash
source venv/bin/activate
```

### 2\. Install Dependencies

It's recommended to install PyTorch first to ensure GPU compatibility.

**Activate GPU PyTorch (Windows/Linux, CUDA 12.1+):**

```bash
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

*Note: The CUDA version (`cu121`) may need to be adjusted based on your system's NVIDIA driver.*

**Install remaining packages:**

```bash
pip install pandas matplotlib seaborn scikit-learn opencv-python tqdm
pip install git+https://github.com/openai/CLIP.git
```

or you can just run
```bash
pip install -r requirements.txt
```

-----

## How to Run

1.  **Download the Dataset**: Download the UCF101 dataset and the train/test split files and place them in the project directory.
2.  **Configure Paths**: Open the Jupyter Notebook (`.ipynb`) and update the `UCF101_VID_PATH` and `UCF101_SPLIT_PATH` variables in the configuration cell to point to your dataset locations.
3.  **Execute Notebook**: Run the notebook cells sequentially to perform the analysis, from data loading and zero-shot evaluation to fine-tuning and visualization.