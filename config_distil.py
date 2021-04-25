# -*- coding: utf-8 -*-
"""
Created on Thu Feb  4 10:31:01 2021

@author: Admin
"""

#%%
import argparse
import os
from config import root_path
#%%

def parse(opt=None):
    parser = argparse.ArgumentParser()

    # Required parameters

    parser.add_argument(
        "--output_dir",
        default=os.path.join(root_path, 'model/generative'),
        type=str,
        help="The output directory where the model checkpoints will be written."
    )
    parser.add_argument(
        "--log_file",
        default=os.path.join(root_path, 'log/distil.log'),
        type=str,
        help="The output directory where the model checkpoints will be written."
    )
    # Other parameters
    parser.add_argument("--train_path",
                        default=os.path.join(root_path, 'data/generative/train.tsv'),
                        type=str)
    parser.add_argument("--dev_path",
                        default=os.path.join(root_path, 'data/generative/dev.tsv'),
                        type=str)
    parser.add_argument("--test_path",
                        default=os.path.join(root_path, 'data/generative/test.tsv'),
                        type=str)
    parser.add_argument(
        "--do_lower_case",
        default=True,
        type=bool,
        help="Whether to lower case the input text. Should be True for uncased "
        "models and False for cased models.")
    parser.add_argument(
        "--max_length",
        default=416,
        type=int,
        help="The maximum total input sequence length after WordPiece tokenization.\
        Sequences longer than this will be truncated, and sequences shorter than this will be padded."
    )
    parser.add_argument(
        "--doc_stride",
        default=128,
        type=int,
        help="When splitting up a long document into chunks, how much stride to take between chunks."
    )

    parser.add_argument("--do_train",
                        default=True,
                        action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_predict",
                        default=False,
                        action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--train_batch_size",
                        default=32,
                        type=int,
                        help="Total batch size for training.")
    parser.add_argument("--predict_batch_size",
                        default=32,
                        type=int,
                        help="Total batch size for predictions.")
    parser.add_argument("--learning_rate",
                        default=0.01,
                        type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--num_train_epochs",
                        default=50.0,
                        type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument(
        "--warmup_proportion",
        default=0.1,
        type=float,
        help="Proportion of training to perform linear learning rate warmup for. E.g., 0.1 = 10% of training.")
    parser.add_argument(
        "--n_best_size",
        default=20,
        type=int,
        help="The total number of n-best predictions to generate in the nbest_predictions.json output file.")
    parser.add_argument(
        "--max_answer_length",
        default=30,
        type=int,
        help="The maximum length of an answer that can be generated. This is needed because the start \
             and end predictions are not conditioned on one another.")
    parser.add_argument(
        "--verbose_logging",
        default=True,
        action='store_true',
        help="If true, all of the warnings related to data processing will be printed. \
             A number of warnings are expected for a normal SQuAD evaluation.")
    parser.add_argument("--is_cuda",
                        default=True,
                        type=bool,
                        help="Whether to use CUDA when available")
    parser.add_argument(
        '--gradient_accumulation_steps',
        type=int,
        default=1,
        help="Number of updates steps to accumualte before performing a backward/update pass."
    )
    parser.add_argument('--random_seed', type=int, default=10236797)
    parser.add_argument('--load_model_type',
                        type=str,
                        default='bert',
                        choices=['bert', 'none'])
    parser.add_argument('--weight_decay_rate', type=float, default=0.01)
    parser.add_argument('--do_eval', action='store_true')
    parser.add_argument('--PRINT_EVERY', type=int, default=200)
    parser.add_argument('--weight', type=float, default=1.0)
    parser.add_argument('--ckpt_frequency', type=int, default=1)

    parser.add_argument('--tuned_checkpoint_T',
                        type=str,
                        default=os.path.join(root_path, 'model/generative/bert.model.epoch.29'))
    parser.add_argument('--tuned_checkpoint_S', type=str, default=None)
    parser.add_argument("--init_checkpoint_S",
                        default=os.path.join(root_path,
                        'lib/rbt3/pytorch_model.bin'),
                        type=str)
    parser.add_argument("--temperature", default=1, type=float, required=False)
    parser.add_argument("--teacher_cached", action='store_true')

    parser.add_argument('--s_opt1',
                        type=float,
                        default=1.0,
                        help="release_start / step1 / ratio")
    parser.add_argument('--s_opt2',
                        type=float,
                        default=0.0,
                        help="release_level / step2")
    parser.add_argument('--s_opt3',
                        type=float,
                        default=1.0,
                        help="not used / decay rate")
    parser.add_argument('--schedule',
                        type=str,
                        default='warmup_linear_release')
    parser.add_argument('--matches', nargs='*', type=list, default=[
                                                                    'L3_hidden_smmd',
                                                                    'L3_hidden_mse',
                                                                    'L3_attention_mse',
                                                                    'L3_attention_ce',
                                                                    'L3_attention_mse_sum',
                                                                    'L3_attention_ce_mean',
                                                                    ])
    
    global args
    if opt is None:
        args = parser.parse_args()
    else:
        args = parser.parse_args(opt)
#%%































