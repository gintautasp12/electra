import argparse

import numpy as np
import torch
from accelerate import Accelerator
from datasets import Dataset
from transformers import AutoConfig, EvalPrediction, ElectraForQuestionAnswering, ElectraTokenizerFast

from utils import postprocess_qa_predictions


def parse_args():
    parser = argparse.ArgumentParser(description="Finetune a transformers model on a Question Answering task")
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=None,
        help="The name of the dataset to use (via the datasets library).",
    )
    parser.add_argument(
        "--train_file",
        type=str,
        default="dataset/train-v2.json",
        help="A csv or a json file containing the training data."
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default="google/electra-small-discriminator",
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--validation_file",
        type=str,
        default=None,
        help="A csv or a json file containing the validation data."
    )
    parser.add_argument(
        "--test_file",
        type=str,
        default="dataset/test-v2.json",
        help="A csv or a json file containing the Prediction data."
    )
    parser.add_argument(
        "--max_predict_samples",
        type=int,
        default=None,
        help="For debugging purposes or quicker training, truncate the number of prediction examples to this",
    )
    parser.add_argument(
        "--preprocessing_num_workers",
        type=int,
        default=4,
        help="The number of processes to use for the preprocessing.",
    )
    parser.add_argument(
        "--overwrite_cache", type=bool, default=False, help="Overwrite the cached training and evaluation sets"
    )
    parser.add_argument(
        "--pad_to_max_length",
        action="store_true",
        help="If passed, pad all samples to `max_seq_length`. Otherwise, dynamic padding is used.",
    )
    parser.add_argument(
        "--max_seq_length",
        type=int,
        default=384,
        help="The maximum total input sequence length after tokenization. Sequences longer than this will be truncated,"
             " sequences shorter will be padded if `--pad_to_max_length` is passed.",
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="predictions",
        help="Where to store the final model.")

    return parser.parse_args()


def create_and_fill_np_array(start_or_end_logits, dataset, max_len):
    """
    Create and fill numpy array of size len_of_validation_data * max_length_of_output_tensor

    Args:
        start_or_end_logits(:obj:`tensor`):
            This is the output predictions of the model. We can only enter either start or end logits.
        eval_dataset: Evaluation dataset
        max_len(:obj:`int`):
            The maximum length of the output tensor. ( See the model.eval() part for more details )
    """

    step = 0
    # create a numpy array and fill it with -100.
    logits_concat = np.full((len(dataset), max_len), -100, dtype=np.float64)
    # Now since we have create an array now we will populate it with the outputs gathered using accelerator.gather
    for i, output_logit in enumerate(start_or_end_logits):  # populate columns
        # We have to fill it such that we have to take the whole tensor and replace it on the newly created array
        # And after every iteration we have to change the step

        batch_size = output_logit.shape[0]
        cols = output_logit.shape[1]

        if step + batch_size < len(dataset):
            logits_concat[step: step + batch_size, :cols] = output_logit
        else:
            logits_concat[step:, :cols] = output_logit[: len(dataset) - step]

        step += batch_size

    return logits_concat


def predict(question, context, id):
    args = parse_args()
    accelerator = Accelerator()

    question_column_name = "question"
    context_column_name = "context"

    config = AutoConfig.from_pretrained('google/electra-small-discriminator')
    tokenizer = ElectraTokenizerFast.from_pretrained('google/electra-small-discriminator')
    model = ElectraForQuestionAnswering.from_pretrained(
        args.model_name_or_path,
        from_tf=False,
        config=config,
    )

    def prepare_validation_features(examples):
        tokenized_examples = tokenizer(
            examples[question_column_name if pad_on_right else context_column_name],
            examples[context_column_name if pad_on_right else question_column_name],
            truncation="only_second" if pad_on_right else "only_first",
            max_length=max_seq_length,
            stride=128,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding="max_length" if args.pad_to_max_length else False,
        )

        # Since one example might give us several features if it has a long context, we need a map from a feature to
        # its corresponding example. This key gives us just that.
        sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")

        # For evaluation, we will need to convert our predictions to substrings of the context, so we keep the
        # corresponding example_id and we will store the offset mappings.
        tokenized_examples["example_id"] = []

        for i in range(len(tokenized_examples["input_ids"])):
            # Grab the sequence corresponding to that example (to know what is the context and what is the question).
            sequence_ids = tokenized_examples.sequence_ids(i)
            context_index = 1 if pad_on_right else 0

            # One example can give several spans, this is the index of the example containing this span of text.
            sample_index = sample_mapping[i]
            tokenized_examples["example_id"].append(examples["id"][sample_index])

            # Set to None the offset_mapping tha
            # t are not part of the context so it's easy to determine if a token
            # position is part of the context or not.
            tokenized_examples["offset_mapping"][i] = [
                (o if sequence_ids[k] == context_index else None)
                for k, o in enumerate(tokenized_examples["offset_mapping"][i])
            ]

        return tokenized_examples

    max_seq_length = min(args.max_seq_length, tokenizer.model_max_length)
    pad_on_right = tokenizer.padding_side == "right"

    def post_processing_function(examples, features, predictions, stage="eval"):
        # Post-processing: we match the start logits and end logits to answers in the original context.
        predictions = postprocess_qa_predictions(
            examples=examples,
            features=features,
            predictions=predictions,
            version_2_with_negative=True,
            n_best_size=10,
            max_answer_length=30,
            null_score_diff_threshold=0,
            output_dir=args.output_dir,
            prefix=stage,
        )

        # Format the result to the format the metric expects.
        formatted_predictions = [
            {"id": k, "prediction_text": v, "no_answer_probability": 0.0} for k, v in predictions.items()
        ]

        return EvalPrediction(predictions=formatted_predictions, label_ids=[])

    examples = {
        "question": [question],
        "context": [context],
        "id": [id],
        "answers": ["..."]}
    predict_data = prepare_validation_features(examples)

    predict_examples = Dataset.from_dict(examples)
    predict_dataset = Dataset.from_dict(predict_data.data)

    predict_data['input_ids'] = torch.LongTensor(np.array(predict_data['input_ids']))
    predict_data['token_type_ids'] = torch.LongTensor(np.array(predict_data['token_type_ids']))
    predict_data['attention_mask'] = torch.LongTensor(np.array(predict_data['attention_mask']))

    del predict_data['offset_mapping']
    del predict_data['example_id']

    with torch.no_grad():
        outputs = model(**predict_data)
        start_logits = outputs.start_logits
        end_logits = outputs.end_logits

        if not args.pad_to_max_length:  # necessary to pad predictions and labels for being gathered
            start_logits = accelerator.pad_across_processes(start_logits, dim=1, pad_index=-100)
            end_logits = accelerator.pad_across_processes(start_logits, dim=1, pad_index=-100)

        all_start_logits = [accelerator.gather(start_logits).cpu().numpy()]
        all_end_logits = [accelerator.gather(end_logits).cpu().numpy()]

    max_len = max([x.shape[1] for x in all_start_logits])  # Get the max_length of the tensor

    # concatenate the numpy array
    start_logits_concat = create_and_fill_np_array(all_start_logits, predict_dataset, max_len)
    end_logits_concat = create_and_fill_np_array(all_end_logits, predict_dataset, max_len)

    # delete the list of numpy arrays
    del all_start_logits
    del all_end_logits

    outputs_numpy = (start_logits_concat, end_logits_concat)
    prediction = post_processing_function(predict_examples, predict_dataset, outputs_numpy)

    return prediction.predictions[0]['prediction_text']


if __name__ == "__main__":
    result = predict(
        question="What is my name?",
        context="My name is John",
        id=1
    )
    print(result)
