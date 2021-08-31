import torch
import time
import json
import logging
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from multiprocessing import cpu_count

from transformers import (
    AutoConfig,
    AutoModelForQuestionAnswering,
    AutoTokenizer,
    squad_convert_examples_to_features
)

from transformers.data.processors.squad import SquadResult, SquadV2Processor, SquadExample
from squad_metrics import compute_predictions_logits 
#from transformers.data.metrics.squad_metrics import compute_predictions_logits

logger = logging.getLogger(__name__)

TOKENIZER_PATH = '/opt/ml/model/'
MODEL_PATH = '/opt/ml/model/'
### Setting hyperparameters
max_seq_length = 512
doc_stride = 256
#n_best_size = 1
max_query_length = 64
max_answer_length = 512
do_lower_case = False
null_score_diff_threshold = 0.0

def to_list(tensor):
    return tensor.detach().cpu().tolist()

tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH, do_lower_case=True, use_fast=False)

def model_fn(model_dir):
    device = get_device()
    config = AutoConfig.from_pretrained(MODEL_PATH)
    model = AutoModelForQuestionAnswering.from_pretrained(MODEL_PATH, config=config).to(device)
    return model

def input_fn(json_request_data, content_type='application/json'):  
    input_data = json.loads(json_request_data)
    return input_data

def predict_fn(input_data, model):
    
    device = get_device()
    
    print(input_data)
    question_texts = input_data['question']
    context_text = input_data['context']
    n_best_size = input_data['nbest']
    processor = SquadV2Processor()
    examples = []

    timer = time.time()
    for i, question_text in enumerate(question_texts):
        
        example = SquadExample(
            qas_id=str(i),
            question_text=question_text,
            context_text=context_text,
            answer_text=None,
            start_position_character=None,
            title="Predict",
            answers=None,
        )

        examples.append(example)
    print(f'Created Squad Examples in {time.time()-timer} seconds')

    print(f'Number of CPUs: {cpu_count()}')
    timer = time.time()
    features, dataset = squad_convert_examples_to_features(
        examples=examples,
        tokenizer=tokenizer,
        max_seq_length=max_seq_length,
        doc_stride=doc_stride,
        max_query_length=max_query_length,
        is_training=False,
        return_dataset="pt",
        threads=cpu_count(),
    )
    print(f'Converted Examples to Features in {time.time()-timer} seconds')

    eval_sampler = SequentialSampler(dataset)
    eval_dataloader = DataLoader(dataset, sampler=eval_sampler, batch_size=10)

    all_results = []

    timer = time.time()
    for batch in eval_dataloader:
        model.eval()
        batch = tuple(t.to(device) for t in batch)

        with torch.no_grad():
            inputs = {
                "input_ids": batch[0],
                "attention_mask": batch[1],
                "token_type_ids": batch[2],
            }

            example_indices = batch[3]

            outputs = model(**inputs)

            for i, example_index in enumerate(example_indices):
                eval_feature = features[example_index.item()]
                unique_id = int(eval_feature.unique_id)

                output = [to_list(output[i]) for output in outputs.to_tuple()]

                start_logits, end_logits = output
                result = SquadResult(unique_id, start_logits, end_logits)
                all_results.append(result)
    print(f'Model predictions completed in {time.time()-timer} seconds') 

    print(all_results)     

    timer = time.time()
    final_predictions,nbest = compute_predictions_logits(
        all_examples=examples,
        all_features=features,
        all_results=all_results,
        n_best_size=n_best_size,
        max_answer_length=max_answer_length,
        do_lower_case=do_lower_case,
        output_prediction_file=None,
        output_nbest_file=None,
        output_null_log_odds_file=None,
        verbose_logging=False,
        version_2_with_negative=True,
        null_score_diff_threshold=null_score_diff_threshold,
        tokenizer=tokenizer
    )
    print(f'Logits converted to predictions in {time.time()-timer} seconds')
    print(nbest)
    print(json.dumps(nbest))
    
#     text_input_ids = tokenizer.batch_encode_plus([text_to_summarize], 
#                                              return_tensors='pt',
#                                              max_length=1024,
#                                              return_token_type_ids=False, 
#                                              return_attention_mask=False).to(device)

#     summary_ids = model.generate(text_input_ids['input_ids'],
#                                  num_beams=4,
#                                  length_penalty=2.0,
#                                  max_length=256,
#                                  min_length=56,
#                                  no_repeat_ngram_size=3)

#     summary_txt = tokenizer.decode(summary_ids.squeeze(), skip_special_tokens=True)

    return nbest


def output_fn(nbest, accept='application/json'):
    return json.dumps(nbest), accept
# def output_fn(final_predictions, accept='application/json'):
#     return json.dumps(final_predictions), accept

def get_device():
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    return device