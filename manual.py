import tensorflow as tf
from basic.main import *

config = tf.app.flags.FLAGS

# directories
config.model_name = 'single'
config.data_dir = 'inter_single/'
config.eval_path = 'inter_single/eval.pklz'
config.nodump_answer = True
config.load_path = None
config.save_dir = 'out-test/basic/00/save/'
config.shared_path = 'out-test/basic/00/shared.json'
config.out_base_dir = 'out-test'
config.out_dir = 'out-test'

# device
config.device = '/cpu:0'
config.device_type = 'gpu'
config.num_gpus = 1

# training and testing
config.mode = 'forward'
config.forward_name = 'single'
config.load = True
config.debug = True
config.single = False
config.load_ema = True
config.wy = False
config.na = False
config.th = 0.5

# parameters
config.batch_size = 10
config.val_num_batches = 10
config.test_num_batches = 0
config.num_epochs = 1
config.num_steps = 1
config.load_step = 0
config.init_lro = 0.001
config.input_keep_prob = 0.8
config.keep_prob = 0.8
config.wd = 0.0
config.hidden_size = 100
config.char_out_size = 100
config.char_emb_size = 8
config.finetune = False
config.highway = True
config.highway_num_layers = 2
config.share_cnn_weights = True
config.share_lstm_weights = True
config.var_decay = 0.999

# optimizations
config.len_opt = True
config.cluster = True
config.cpu_opt = True

config.progress = True
config.log_period = 100
config.eval_period = 1000
config.save_period = 1000
config.max_to_keep = 20
config.dump_eval = True
config.dump_answer = True
config.vis = True
config.dump_pickle = True
config.decay = 0.9

# thresholds for speed and memory usage
config.word_count_th = 10
config.char_count_th = 50
config.sent_size_th = 400
config.num_sents_th = 8
config.ques_size_th = 30
config.word_size_th = 16
config.para_size_th = 256

# advanced options
config.use_glove_for_unk = True
config.known_if_glove = True
config.logit_func = 'tri_linear'
config.answer_func = 'linear'
config.sh_logit_func = 'tri_linear'
config.lower_word = False
config.squash = False
config.use_char_emb = False
config.use_word_emb = True
config.q2c_att = True
config.c2q_att = True
config.dynamic_att = False

# read data from json files
print('reading data from files...')
test_data = read_data(config, 'single', True)
update_config(config, [test_data])

# build the model
print('build the model')
models = get_multi_gpu_models(config)
