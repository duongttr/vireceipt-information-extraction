import os
from argparse import ArgumentParser
from transformers.data.data_collator import default_data_collator
from transformers import LayoutLMv3ForTokenClassification, TrainingArguments, Trainer

from src.util.utils import compute_metrics
from src.data.dataset import LayoutLMv3Dataset, id2label, label2id, processor


def main(args):
    train_path = os.path.join(args.dataset_path, "train.json")
    val_path = os.path.join(args.dataset_path, "val.json")
    dataset = LayoutLMv3Dataset(
        data_path_dict={'train': train_path, 'val': val_path}).get_dataset()
    train_dataset = dataset['train']
    val_dataset = dataset['val']

    train_dataset.set_format("torch")
    val_dataset.set_format("torch")

    # Init model
    model = LayoutLMv3ForTokenClassification.from_pretrained("microsoft/layoutlmv3-base",
                                                             id2label=id2label,
                                                             label2id=label2id)

    # Configure training arugments
    training_args = TrainingArguments(output_dir=args.output_dir,
                                      max_steps=args.max_steps,
                                      per_device_train_batch_size=args.batch_size,
                                      per_device_eval_batch_size=args.batch_size,
                                      learning_rate=args.learning_rate,
                                      evaluation_strategy="steps",
                                      eval_steps=args.eval_steps,
                                      load_best_model_at_end=args.load_best_model_at_end,
                                      metric_for_best_model=args.metric_for_best_model)

    # Start training
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=processor,
        data_collator=default_data_collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--dataset_path", type=str, default="dataset/")
    parser.add_argument("--output_dir", type=str, default="output/")
    parser.add_argument("--max_steps", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--eval_steps", type=int, default=100)
    parser.add_argument("--load_best_model_at_end", type=bool, default=True)
    parser.add_argument("--metric_for_best_model", type=str, default="f1")

    main(parser.parse_args())
