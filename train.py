from huggingface_hub import login
import os
from datasets import load_dataset, Dataset, DatasetDict
from transformers import AutoTokenizer, AutoModelForTokenClassification, TrainingArguments, Trainer
from transformers import DataCollatorForTokenClassification
import evaluate
import numpy as np

# apply huggingface token to os env
os.environ["HUGGINGFACE_WRITE_TOKEN"] = ''
login(HUGGINGFACE_WRITE_TOKEN)

# load dataset
train_data = load_dataset('Minggz/Vi-Ner', trust_remote_code=True, split="train[:10000]")
val_data = load_dataset('Minggz/Vi-Ner', trust_remote_code=True, split="validation[:2000]")
test_data = load_dataset('Minggz/Vi-Ner', trust_remote_code=True, split="test[:2000]")

dataset = DatasetDict({'train': train_data, 'test': test_data, 'validation': val_data})

# pretrain model
model_ckpt = "distilbert/distilbert-base-multilingual-cased"

# load tokenize
tokenizer = AutoTokenizer.from_pretrained(model_ckpt)


# function tokenzier input data
def tokenize_and_align_labels(examples):
    tokenized_inputs = tokenizer(examples['tokens'], truncation=True, is_split_into_words=True)
    labels = []
    for i, label in enumerate(examples['ner_idx']):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            # if id=-100 then loss is not calculated
            if word_idx is None:
                label_ids.append(-100)
            elif word_idx != previous_word_idx:
                label_ids.append(label[word_idx])
            else:
                label_ids.append(-100)
            previous_word_idx = word_idx
        labels.append(label_ids)
    tokenized_inputs['labels'] = labels
    return tokenized_inputs


# process dataset
tokenized_dataset = dataset.map(tokenize_and_align_labels, batched=True)

######################## compute metric to eval model##########################
metric = evaluate.load('seqeval')
label_names = list({'B-DATETIME': 0, 'B-LOCATION': 1, 'B-ORGANIZATION': 2, 'B-PERSON': 3, 'I-DATETIME': 4,
                    'I-LOCATION': 5, 'I-ORGANIZATION': 6, 'I-PERSON': 7, 'O': 8})


def compute_metrics(eval_preds):
    logits, labels = eval_preds
    predictions = np.argmax(logits, axis=-1)
    true_labels = [[label_names[l] for l in label if l!=-100] for label in labels]
    true_predictions = [[label_names[p] for p, l in zip(prediction, label) if l != -100]
                        for prediction, label in zip(predictions, labels)]
    all_metrics = metric.compute(predictions=true_predictions, references=true_labels)
    return {
        "precision": all_metrics['overall_precision'],
        'recall': all_metrics['overall_recall'],
        'f1': all_metrics['overall_f1'],
        'accuracy': all_metrics['overall_accuracy'],
    }
################################################################################


tag2index = {'B-DATETIME': 0, 'B-LOCATION': 1, 'B-ORGANIZATION': 2, 'B-PERSON': 3, 'I-DATETIME': 4, 'I-LOCATION': 5,
             'I-ORGANIZATION': 6, 'I-PERSON': 7, 'O': 8}
index2tag = {v: k for k, v in tag2index.items()}

# load pretrained model
model = AutoModelForTokenClassification.from_pretrained(model_ckpt, id2label=index2tag, label2id=tag2index)

# load data colator: auto pad and process input data, make sure model can work with different length of input data
data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

args = TrainingArguments(output_dir="Vi-DistilBert-NER", evaluation_strategy='epoch',
                         save_strategy='epoch',
                         learning_rate=2e-5,
                         num_train_epochs=3,
                         weight_decay=0.01)

trainer = Trainer(model=model, args=args,
                  train_dataset=tokenized_dataset['train'],
                  eval_dataset=tokenized_dataset['validation'],
                  data_collator=data_collator,
                  compute_metrics=compute_metrics,
                  tokenizer=tokenizer)

trainer.train()

trainer.push_to_hub()















