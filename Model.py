

!nvidia-smi

!pip install "transformers[sentencepiece]" datasets sacrebleu rouge_score py7zr

! pip install - -upgrade accelerate
! pip uninstall -y transformers accelerate
! pip install transformers accelerate

#precautionary measures
# Upgrade accelerate
!pip install --upgrade accelerate

# Uninstall transformers and accelerate
!pip uninstall -y transformers accelerate

# Reinstall them
!pip install transformers accelerate

# Hugging Face libraries
from transformers import pipeline, set_seed, AutoModelForSeq2SeqLM, AutoTokenizer
from datasets import load_dataset

# Data & Plotting libraries
import pandas as pd
import matplotlib.pyplot as plt

# NLP libraries
import nltk
from nltk.tokenize import sent_tokenize

# Progress bar
from tqdm import tqdm

# PyTorch
import torch

# Download NLTK resources
nltk.download('punkt')

#checking cuda is available or not cuda is the machine its running on
device = "cuda" if torch.cuda.is_available() else "cpu"
device

# 2. Choose the model checkpoint:
# "google/pegasus-cnn_dailymail" is a Pegasus model fine-tuned for summarizing news articles
model_ckpt = "google/pegasus-cnn_dailymail"

# 3. Load the tokenizer:
# Tokenizer converts raw text into input IDs that the model can understand
tokenizer = AutoTokenizer.from_pretrained(model_ckpt)
model_pegasus = AutoModelForSeq2SeqLM.from_pretrained(model_ckpt)

# Upgrade datasets and fsspec
!pip install --upgrade datasets fsspec

from datasets import load_dataset

# Load the DialogSum dataset from knkarthick
dataset = load_dataset("knkarthick/dialogsum")

print(dataset)

print(dataset["train"][0])

print(dataset["train"][0]["dialogue"])

print(dataset["train"][1]["summary"])

# Calculate the length of each split (train, validation, test)
split_lengths = [len(dataset[split]) for split in dataset]
print(f"Split lengths: {split_lengths}")

# Print column names in the 'train' split
print(f"Features: {dataset['train'].column_names}")

# Display an example dialogue from the 'test' split
print("\nDialogue:")
print(dataset["test"][1]["dialogue"])

# Display the summary of the above dialogue
print("\nSummary:")
print(dataset["test"][1]["summary"])

# Step 1: Import the tokenizer class from transformers
from transformers import AutoTokenizer

# Step 2: Define the model checkpoint and load the tokenizer
model_ckpt = "google/pegasus-cnn_dailymail"
tokenizer = AutoTokenizer.from_pretrained(model_ckpt)

# Step 3: Load the SamSum dataset from HuggingFace datasets library
from datasets import load_dataset
dataset_samsum = load_dataset("knkarthick/dialogsum")

# Step 4: Define a preprocessing function that converts raw text into token IDs
def convert_examples_to_features(example_batch):
    # Tokenize the dialogue input (source text)
    input_encodings = tokenizer(
        example_batch['dialogue'],
        max_length=1024,            # Limit input length to 1024 tokens
        truncation=True,           # Truncate if input is longer
        padding='max_length'       # Pad if input is shorter
    )

    # Tokenize the summary (target text)
    with tokenizer.as_target_tokenizer():
        target_encodings = tokenizer(
            example_batch['summary'],
            max_length=128,         # Limit target length to 128 tokens
            truncation=True,
            padding='max_length'
        )

    # Return processed inputs and labels
    return {
        'input_ids': input_encodings['input_ids'],
        'attention_mask': input_encodings['attention_mask'],
        'labels': target_encodings['input_ids']
    }

# Step 5: Apply the preprocessing function to all splits of the dataset using `.map()`
dataset_samsum_pt = dataset_samsum.map(
    convert_examples_to_features,
    batched=True                 # Apply function in batches for speed
)

dataset_samsum_pt = dataset_samsum.map(convert_examples_to_features, batched=True)

dataset_samsum_pt ["train"]

"""training starts here"""

from transformers import DataCollatorForSeq2Seq

# Data collator that dynamically pads inputs for sequence-to-sequence training
seq2seq_data_collator = DataCollatorForSeq2Seq(tokenizer, model=model_pegasus)

from transformers import TrainingArguments, Trainer

# Step: Define training arguments
training_args = TrainingArguments(
    output_dir='pegasus-samsum',              # Where to save the model
    num_train_epochs=1,                       # Number of epochs (you can increase later)
    per_device_train_batch_size=1,            # Batch size for training
    per_device_eval_batch_size=1,             # Batch size for evaluation
    logging_steps=10,                         # Log metrics every 10 steps
              # Evaluate during training every `eval_steps`
    save_steps=1_000_000,                     # Save model every 1M steps (basically disables saving during training)
    gradient_accumulation_steps=16,           # To simulate larger batch size
    report_to="none"                          # Avoid logging to external services like WandB
)

# Now you can use this in the Trainer setup later

from transformers import Trainer

trainer = Trainer(
    model=model_pegasus,                            # Your Pegasus model
    args=training_args,                             # TrainingArguments you defined earlier
    tokenizer=tokenizer,                            # Tokenizer
    data_collator=seq2seq_data_collator,            # Collator to dynamically pad inputs
    train_dataset=dataset_samsum_pt["test"],       # Use train set for training
    eval_dataset=dataset_samsum_pt["validation"]    # Use validation set for evaluation
)

trainer.train()

from tqdm import tqdm

# Function to yield batch-sized chunks
def generate_batch_sized_chunks(list_of_elements, batch_size):
    """
    Split the dataset into smaller batches that we can process simultaneously.
    Yield successive batch-sized chunks from list_of_elements.
    """
    for i in range(0, len(list_of_elements), batch_size):
        yield list_of_elements[i: i + batch_size]

# Function to calculate metrics like ROUGE
def calculate_metric_on_test_ds(
    dataset,
    metric,
    model,
    tokenizer,
    batch_size=16,
    device="cuda" or "cpu",
    column_text="article",
    column_summary="highlights"
):
    article_batches = list(generate_batch_sized_chunks(dataset[column_text], batch_size))
    target_batches = list(generate_batch_sized_chunks(dataset[column_summary], batch_size))

    decoded_preds = []
    decoded_labels = []

    for article_batch, target_batch in tqdm(
        zip(article_batches, target_batches),
        total=len(article_batches)
    ):
        inputs = tokenizer(
            article_batch,
            truncation=True,
            padding=True,
            return_tensors="pt"
        )

        inputs = {k: v.to(device) for k, v in inputs.items()}

        summaries = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            length_penalty=1.0,
            num_beams=8,
            max_length=128
        )

        decoded_summary = [tokenizer.decode(s, skip_special_tokens=True) for s in summaries]
        decoded_summary = [d.strip() for d in decoded_summary]

        decoded_preds.extend(decoded_summary)
        decoded_labels.extend(target_batch)

    # Compute ROUGE score
    result = metric.compute(predictions=decoded_preds, references=decoded_labels)
    return result

!pip install evaluate

# Import the 'evaluate' library to compute metrics like ROUGE
from evaluate import load

# Load the ROUGE metric from Hugging Face's 'evaluate' library
# ROUGE is used to compare generated summaries with actual summaries
rouge_metric = load("rouge")

# These are the specific types of ROUGE scores we want to calculate:
# - 'rouge1': overlap of unigrams (single words)
# - 'rouge2': overlap of bigrams (2-word combinations)
# - 'rougeL': longest common subsequence
# - 'rougeLsum': variant for summarization tasks
rouge_names = ["rouge1", "rouge2", "rougeL", "rougeLsum"]

import pandas as pd

# Call the function to calculate ROUGE scores on the test dataset
# Pass required arguments: dataset, metric, model, tokenizer, etc.
score = calculate_metric_on_test_ds(
    dataset=dataset_samsum['test'][0:10], #// to shorten the test
    metric=rouge_metric,
    model=trainer.model,
    tokenizer=tokenizer,
    batch_size=2,
    device="cuda",                      # or "cpu"
    column_text="dialogue",            # ✅ correct input column
    column_summary="summary"           # ✅ correct target column
)


# Extract specific ROUGE values from the score dictionary
# 'mid.fmeasure' gives the main F1-score value for each ROUGE type
rouge_dict = dict((rn, round(score[rn], 2)) for rn in rouge_names)


# Convert the result to a Pandas DataFrame for a cleaner display
rouge_df = pd.DataFrame([rouge_dict], index=[f'pegasus'])

# Show the DataFrame
print(rouge_df)

# Save the trained Pegasus model and tokenizer
model_pegasus.save_pretrained("pegasus-samsum-model")
tokenizer.save_pretrained("pegasus-samsum-model")

tokenizer.save_pretrained("tokenizer")

tokenizer = AutoTokenizer.from_pretrained("/content/tokenizer")

from transformers import pipeline

# Function to get max_length based on input text length to avoid warnings
def get_max_length_for_summary(input_text, tokenizer, max_ratio=0.5, max_limit=128):
    input_len = len(tokenizer.tokenize(input_text))
    max_len = min(int(input_len * max_ratio), max_limit)
    return max(max_len, 10)  # minimum 10 tokens for summary length

# Replace this with the index of the sample you want to test
e = 0

# Get sample text and reference summary from your test dataset at index e
sample_text = dataset_samsum['test'][e]['dialogue']
reference = dataset_samsum['test'][e]['summary']

# Load your summarization pipeline
pipe = pipeline(
    "summarization",
    model="pegasus-samsum-model",
    tokenizer=tokenizer,
    device=0  # Use GPU if available, else -1 for CPU
)

# Dynamically calculate max_length to avoid warnings
max_len = get_max_length_for_summary(sample_text, tokenizer)

# Setup generation parameters with dynamic max_length
gen_kwargs = {
    "length_penalty": 0.8,
    "num_beams": 8,
    "max_length": max_len
}

# Print the dialogue
print("Dialogue:")
print(sample_text)

# Print the reference summary
print("\nReference Summary:")
print(reference)

# Generate summary and print it
print("\nGenerated Summary:")
generated_summary = pipe(sample_text, **gen_kwargs)
print(generated_summary[0]["summary_text"])

