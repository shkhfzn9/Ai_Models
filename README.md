cat <<EOF > README.md

# 🧠 DialogSum Summarizer using Pegasus

This project implements a dialogue summarizer using the **Pegasus** transformer model fine-tuned on the [DialogSum](https://huggingface.co/datasets/knkarthick/dialogsum) dataset. The model is trained and evaluated using the Hugging Face \`transformers\` and \`datasets\` libraries.

---

## 🔧 Author

Sheikh Faizan

## 🛠️ Framework

Hugging Face Transformers

## 💻 Dataset

DialogSum

## 🤖 Model

google/pegasus-cnn_dailymail

---

## 📁 Project Structure

\`\`\`
dialogsum_summarizer/
├── dialogsum_summarizer.py # Training and testing pipeline
├── pegasus-dialogsum-final/ # Directory for saving trained model
├── README.md # You're reading it now
\`\`\`

---

## 📌 Features

- ✅ Fine-tune \`google/pegasus-cnn_dailymail\` on \`DialogSum\` dialogue data.
- ✅ Generates abstractive summaries from dialogue input.
- ✅ Trains with Hugging Face \`Trainer\` API.
- ✅ Evaluation using ROUGE metrics.
- ✅ Works on both GPU (CUDA) and CPU.
- ✅ Easily test any sample from the test dataset.

---

## 📦 Installation

1. Create a virtual environment (optional):

\`\`\`bash
python -m venv venv
source venv/bin/activate # or venv\\Scripts\\activate on Windows
\`\`\`

2. Install dependencies:

\`\`\`bash
pip install transformers datasets evaluate tqdm pandas torch
\`\`\`

---

## 🚀 Usage

### 🔧 Train the model

\`\`\`bash
python dialogsum_summarizer.py --train
\`\`\`

### 📈 Test with a specific sample

\`\`\`bash
python dialogsum_summarizer.py --test 0
\`\`\`

---

## 🧪 Evaluation

Evaluation is done using ROUGE metrics:

- ROUGE-1
- ROUGE-2
- ROUGE-L
- ROUGE-Lsum

---

## 📊 Example Output

\`\`\`text
Input Dialogue:
Speaker 1: Hey, how are you?
Speaker 2: I'm fine, thanks. How about you?

Reference Summary:
Two friends catch up and exchange pleasantries.

Generated Summary:
Two people greet each other and ask about each other's well-being.
\`\`\`

---

## 📌 Notes

- Model: \`google/pegasus-cnn_dailymail\`
- Dataset: DialogSum
- Tokenization: Max input = 1024, Max summary = 128
- Batch size: 1
- Epochs: 1

---

## 📂 Save Model

The model and tokenizer will be saved to:

\`\`\`
pegasus-dialogsum-final/
\`\`\`

---

## 📬 Contact

- GitHub: [shkhfzn9](https://github.com/shkhfzn9)
- LinkedIn: [Sheikh Faizan](https://www.linkedin.com/in/sheikh-faizan-4a9a29326/)
- Email: Sheikhfaizan.w@gmail.com

---

## 📜 License

MIT License

---

# 🏋️‍♂️ My AI Model Training

## Training Results

### Training Loss

![Training Loss](screenshots/trainingLoss.png)

### Accuracy Plots and Training Details

#### Data Conversion Process

![](screenshots/dataConverstion.png)

#### Dataset Features, Rows, and Overview

![](screenshots/datasetFeaturesRowsEtc.png)

#### Generated Conversation Example Using Model Training

![](screenshots/generatedConversationUsingModelTraining.png)

#### Loading the Dataset

![](screenshots/LoadingDataSet.png)

#### Saving the Trained Model and Tokenizer

![](screenshots/SavingTheTrainingModelAndTokenizer.png)

#### Training Data Output Without Model

![](screenshots/traingDataOutPutWithoutModel.png)

EOF
