import os
import gc
import sys
import json
import copy
import keras
import pickle
import gensim
import random
import argparse
import datetime
import subprocess

import numpy as np
import tensorflow as tf

from random import shuffle
from models.pacrr import PACRR
from os import listdir
from os.path import isfile, join
from gensim.models.keyedvectors import KeyedVectors
from keras.preprocessing.sequence import pad_sequences
from keras.callbacks import ReduceLROnPlateau
from tqdm import tqdm
from keras.backend.tensorflow_backend import set_session


#config = tf.ConfigProto()
#config.gpu_options.per_process_gpu_memory_fraction = 0.7
#set_session(tf.Session(config=config))

SEED = 1234
random.seed(SEED)

def log(data, filename, cl=False):
	f = open(retr_dir + '/{0}'.format(filename), 'a')
	f.write(data + '\n')
	if cl:
		f.write('\n')
	f.close()

def results_to_string(prefix, results):
	new_str =  '\t'.join(['{0} {1}: {2}'.format(prefix, m, results[m]) for m in metrics])
	return new_str

def myprint(s):
    with open('log_{0}.txt'.format(params_file.split('/')[-1], 'w')) as f:
        print(s, file=f)

def write_trec_eval_results(q_id, sorted_retr_scores, filename):

	path = '{0}/{1}'.format(retr_dir, filename)
	file = open(path, 'a')
	i = 1
	for doc in sorted_retr_scores:
		print("{0} Q0 {1} {2} {3} PACRR".format(q_id, doc[0], i, doc[1]), end="\n", file=file)
		i += 1
	file.close()
	return path

def shuffle_train_pairs(train_data_dict):
	num_pairs = train_data_dict['num_pairs']
	inds = np.arange(num_pairs)     
	np.random.shuffle(inds)              
	for k in train_data_dict.keys():
		if isinstance(train_data_dict[k], list) and len(train_data_dict[k]) == train_data_dict['num_pairs']:
			#print(k + ' shuffled.')
			train_data_dict[k] = np.array(train_data_dict[k])[inds]
	return train_data_dict

def write_bioasq_results_dict(bioasq_res_dict, filename):

	path = '{0}/{1}'.format(retr_dir, filename.replace('dev_', 'dev_bioasq_'))
	with open(path, 'w') as f:
		json.dump(bioasq_res_dict, f, indent=2)
	return path

def trec_eval_custom(q_rels_file, path):
	eval_res = subprocess.Popen(
		['python', 'eval/run_eval.py', q_rels_file, path],
		stdout=subprocess.PIPE, shell=False)
	(out, err) = eval_res.communicate()
	eval_res = out.decode("utf-8")

	results = {}

	for line in eval_res.split('\n'):
	
		splitted_line = line.split()
		try:
			first_element = splitted_line[0]
			for metric in metrics:
				if first_element == metric:
					value = float(splitted_line[2])
					results[metric] = value
		except:
			continue

	#print(results)
		

	file = open(path + '_trec_eval', 'w')
	file.write(eval_res)
	file.close()
	return results

def bioasq_eval_custom(path, golden_file):

	eval_res = subprocess.Popen(
		['java', '-Xmx10G', '-cp', '$CLASSPATH:./bioasq_eval/flat/BioASQEvaluation/dist/BioASQEvaluation.jar', 'evaluation.EvaluatorTask1b', '-phaseA', '-e', '5', golden_file, path],
		stdout=subprocess.PIPE, shell=False)

	(out, err) = eval_res.communicate()
	eval_res = out.decode("utf-8")

	results = {}
	splitted_eval_res = eval_res.split()
	results['map'] = float(splitted_eval_res[8])
	results['gmap'] = float(splitted_eval_res[9])

	file = open(path + '_bioasq_eval', 'w')
	file.write(eval_res)
	file.close()
	return results

def get_precision_at_k(res_dict, qrels, k):
	custom_metrics = {}
	
	sum_prec_at_k = 0
	for q in res_dict['questions']:
		hits = sum([1 for doc_id in q['documents'][:k] if doc_id in qrels[q['id']]])
		sum_prec_at_k += hits / k
	return sum_prec_at_k / len(res_dict['questions'])

def load_qrels(path):

	with open(path , 'r') as f:
		data = json.load(f)

	qrels = {}
	for q in data['questions']:
		qrels[q['id']] = set(q['documents'])
	return qrels

def load_embeddings(path, term2ind):
	word2vec = gensim.models.KeyedVectors.load_word2vec_format(path, binary=True)
	word2vec.init_sims(replace=True)
	
	# Retrieve dimension space of embedding vectors
	dim = word2vec['common'].shape[0]
	
	# Initialize (with zeros) embedding matrix for all vocabulary
	embedding = np.zeros((len(term2ind)+1, dim))
	rand_count = 0

	# Fill embedding matrix with know embeddings
	for key, value in term2ind.items():
		if value == 0:
			print('ZERO ind found in vocab.')
		try:
			embedding[value] = word2vec[key]
		except:
			rand_count += 1
			continue

	embedding[-1, :] = np.mean(embedding[1:-1, :], axis=0)

	print("No of OOV tokens: %d"%rand_count)
	return embedding

def rerank_query_generator(filename):
	with open(filename, 'rb') as f:
		unpickler = pickle.Unpickler(f)
		i = 0
		while True:
			try:
				q = unpickler.load()
				yield q
				q = None
			except EOFError:
				break	

def rerank(queries_to_rerank, filename, scoring_model, qrels, qrels_file):
		
		res_dict = {'questions': []}
		for q in tqdm(queries_to_rerank, desc='queries'):
			#progress(i, 500, 'test queries')
			scores = scoring_model.predict(
					{
						'query_inds': pad_sequences(np.tile(q['token_inds'], (len(q['retrieved_samples']['doc_list']), 1)), maxlen=model_params['maxqlen'], padding=model_params['padding_mode']),
						'query_idf': np.expand_dims(pad_sequences(np.tile(q['idf'], (len(q['retrieved_samples']['doc_list']), 1)), model_params['maxqlen'], padding=model_params['padding_mode']), axis=-1),
						'pos_doc_inds': pad_sequences(q['retrieved_samples']['doc_list'], maxlen=model_params['simdim'], padding=model_params['padding_mode']),
						'pos_doc_bm25_scores': np.array(q['retrieved_samples']['doc_normBM25']), 
						'pos_doc_overlap_vec' : np.array(q['retrieved_samples']['doc_overlap'])
					}
					,
					batch_size=model_params['predict_batch_size']

				)
			scores = [s[0] for s in scores]
			retr_scores = list(zip(q['retrieved_samples']['documents'], scores))
			shuffle(retr_scores) # Shuffle docs to make sure re-ranker works.
			sorted_retr_scores = sorted(retr_scores, key=lambda x: x[1], reverse=True)

			res_dict['questions'].append({'id': q['id'], 'documents': ['http://www.ncbi.nlm.nih.gov/pubmed/' + d[0] for d in sorted_retr_scores]})

		path_bioasq = write_bioasq_results_dict(res_dict, filename)
		trec_eval_metrics = trec_eval_custom(qrels_file, path_bioasq)

		for i in range(len(res_dict['questions'])):
			res_dict['questions'][i]['documents'] = res_dict['questions'][i]['documents'][:10]
		
		path_bioasq = write_bioasq_results_dict(res_dict, filename + '_top10')
		bioasq_metrics = bioasq_eval_custom(path_bioasq, qrels_file)
		precision_at_5 = get_precision_at_k(res_dict, qrels, 5)
		
		reported_results = {'P_5': precision_at_5, 'MAP(bioasq)': bioasq_metrics['map'], 'GMAP(bioasq)': bioasq_metrics['gmap']}

		return reported_results

def pacrr_train(train_pairs, dev_pairs, train_rerank_path, dev_rerank_path):

	class Evaluate(keras.callbacks.Callback):
		def on_epoch_end(self, epoch, logs={}):
			epoch += 1
			path = '{0}/{1}'.format(retr_dir, 'weights.keras.epoch_{0}'.format(epoch))
			train_rerank = False
			
			if train_rerank:
				queries_to_rerank = rerank_query_generator(train_rerank_path)
				train_rerank_results = rerank(queries_to_rerank, 'train_ranking_epoch{0}'.format(epoch), scoring_model, train_qrels, config['QRELS_TRAIN'])
				train_res_str = results_to_string('Train', train_rerank_results)
			else:
				train_res_str = ''
				train_rerank_results = {}
				train_rerank_results['MAP(bioasq)'], train_rerank_results['GMAP(bioasq)'], train_rerank_results['P_5'] = 0, 0, 0

			queries_to_rerank = rerank_query_generator(dev_rerank_path)
			dev_rerank_results = rerank(queries_to_rerank, 'dev_ranking_epoch{0}'.format(epoch), scoring_model, dev_qrels, config['QRELS_DEV'])
			dev_res_str = results_to_string('Dev', dev_rerank_results)
			print(logs)
			print('Epoch: {0} \n{1} \n'.format(epoch, '\n'.join(dev_res_str.split('\t'))))
			#log('Epoch: {0} \t {1} \t Training_loss: {2} \t Training_acc: {3} \t Val_loss: {4} \t Val_Acc: {5}'.format(epoch, train_res_str, dev_res_str, logs['loss'], logs['acc'], logs['val_loss'], logs['val_acc']))
			log('|'.join(list(map(str, [epoch, dev_rerank_results['MAP(bioasq)'], dev_rerank_results['GMAP(bioasq)'], dev_rerank_results['P_5'], logs['val_acc'], logs['val_loss'], train_rerank_results['MAP(bioasq)'], train_rerank_results['GMAP(bioasq)'], train_rerank_results['P_5'], logs['acc'], logs['loss']]))), 'log.txt')
			global best_epoch
			global best_map
			global best_map_weights_path

			best_epoch = epoch
			if dev_rerank_results['MAP(bioasq)'] > best_map:
				print('========== Best epoch so far: {0} =========='.format(epoch))
				log('^^^', 'log.txt')
				if best_map_weights_path is not None:
					if os.path.exists(best_map_weights_path):
						os.remove(best_map_weights_path)
				best_map = dev_rerank_results['MAP(bioasq)']
				best_map_weights_path = path
				pacrr_model.dump_weights(path)

			print('\n')

	def progress(i, total, suffix):
		sys.stdout.write('Progress: %d/%d %s completed.\r' % (i, total, suffix))
		sys.stdout.flush()

	

	pacrr_train_labels = np.ones((len(train_pairs['queries']), 1), dtype=np.int32)
	pacrr_dev_labels = np.ones((len(dev_pairs['queries']), 1), dtype=np.int32)

	print('Number of samples: {0}'.format(len(train_pairs['queries'])))

	model_params['maxqlen'] = config['MAX_Q_LEN']
	model_params['simdim'] = config['SIMDIM']
	model_params['nsamples'] = train_pairs['num_pairs']
	model_params['embed'] = load_embeddings(config['WORD_EMBEDDINGS_FILE'], term2ind)
	model_params['vocab_size'] = model_params['embed'].shape[0]
	model_params['emb_dim'] = model_params['embed'].shape[1]

	pacrr_model = PACRR(model_params, 3)
	training_model, scoring_model = pacrr_model.build()
	
	training_model.summary()
	
	#with open(retr_dir + '/log_{0}.txt'.format(params_file.split('/')[-1]), 'w') as fh:
	#    training_model.summary(print_fn=lambda x: fh.write(x + '\n'))
	    
	train_params = {}
	print(model_params)
	log(str(model_params), 'model.txt', True)
	log('epoch|dev_map|dev_gmap|dev_p_5|dev_acc|dev_loss|train_map|train_gmap|train_p_5|train_acc|train_loss', 'log.txt')

	train_params['query_inds'] = pad_sequences(train_pairs['queries'], maxlen=30, padding=model_params['padding_mode'])
	train_params['query_idf'] = np.expand_dims(pad_sequences(train_pairs['queries_idf'], maxlen=30, padding=model_params['padding_mode']), axis=-1)
	train_params['pos_doc_inds'] = pad_sequences(train_pairs['pos_docs'], maxlen=model_params['simdim'], padding=model_params['padding_mode'])
	train_params['pos_doc_bm25_scores'] = np.array(train_pairs['pos_docs_normBM25'])
	train_params['pos_doc_overlap_vec'] = np.array(train_pairs['pos_docs_overlap'])
	for n in range(model_params['numneg']):
		train_params['neg{0}_doc_inds'.format(n)] = pad_sequences(train_pairs['neg_docs'], maxlen=model_params['simdim'], padding=model_params['padding_mode'])
		train_params['neg{0}_doc_bm25_scores'.format(n)] = np.array((train_pairs['neg_docs_normBM25']))
		train_params['neg{0}_doc_overlap_vec'.format(n)] = np.array((train_pairs['neg_docs_overlap']))

	dev_params = {}
	dev_params['query_inds'] = pad_sequences(dev_pairs['queries'], maxlen=30, padding=model_params['padding_mode'])
	dev_params['query_idf'] = np.expand_dims(pad_sequences(dev_pairs['queries_idf'], maxlen=30, padding=model_params['padding_mode']), axis=-1)
	dev_params['pos_doc_inds'] = pad_sequences(dev_pairs['pos_docs'], maxlen=model_params['simdim'], padding=model_params['padding_mode'])
	dev_params['pos_doc_bm25_scores'] = np.array(dev_pairs['pos_docs_normBM25'])
	dev_params['pos_doc_overlap_vec'] = np.array(dev_pairs['pos_docs_overlap'])
	for n in range(model_params['numneg']):
		dev_params['neg{0}_doc_inds'.format(n)] = pad_sequences(dev_pairs['neg_docs'], maxlen=model_params['simdim'], padding=model_params['padding_mode'])
		dev_params['neg{0}_doc_bm25_scores'.format(n)] = np.array((dev_pairs['neg_docs_normBM25']))
		dev_params['neg{0}_doc_overlap_vec'.format(n)] = np.array((dev_pairs['neg_docs_overlap']))

	for k in train_params:
		print(k, train_params[k].shape)

	eval = Evaluate()
	tbcallback = keras.callbacks.TensorBoard(log_dir=retr_dir)

	training_model.fit(train_params, pacrr_train_labels, 
		validation_data=[dev_params, pacrr_dev_labels], 
		batch_size=model_params['train_batch_size'], 
		epochs=model_params['epochs'], 
		callbacks=[eval, tbcallback], 
		verbose=1, 
		shuffle=True)

	print('Test evaluation on best epoch.')
	print(best_map)
	print(best_map_weights_path)
	print('==============================')
	training_model.load_weights(best_map_weights_path)
	scoring_model.load_weights(best_map_weights_path)
	#test_loss = training_model.evaluate(dev_params, pacrr_dev_labels)
	#print(test_loss)

	log('batch|epoch|test_map|test_gmap|test_p_5', 'test_log.txt')

	for i in range(5):
		batch = i + 1
		
		test_qrels = load_qrels(test_qrels_list[batch])
		test_rerank_path = test_rerank_path_list[batch]
		
		queries_to_rerank = rerank_query_generator(test_rerank_path)
		output_file = 'test_ranking_batch{0}'.format(batch)
		test_rerank_results = rerank(queries_to_rerank, output_file, scoring_model, test_qrels, test_qrels_list[batch])

		test_res_str = results_to_string('Test', test_rerank_results)
		print('Epoch: {0} \n{1}'.format(best_epoch, '\n'.join(test_res_str.split('\t')), '\n'.join(test_res_str.split('\t'))))

		log('|'.join(list(map(str, [batch, best_epoch, test_rerank_results['MAP(bioasq)'], test_rerank_results['GMAP(bioasq)'], test_rerank_results['P_5']]))), 'test_log.txt')
	if os.path.exists(best_map_weights_path):
		os.remove(best_map_weights_path)



if __name__=='__main__':
	best_epoch = -1
	best_map = 0
	best_map_weights_path = None

	parser = argparse.ArgumentParser()
	parser.add_argument('-config', dest='config_file')
	parser.add_argument('-params', dest='params_file')
	parser.add_argument('-out', dest='out_folder')
	args = parser.parse_args()

	print(args.config_file)
	print(args.params_file)

	config_file = args.config_file
	params_file = args.params_file

	with open(config_file, 'r') as f:
		config = json.load(f)  
	with open(params_file, 'r') as f:
		model_params = json.load(f)

	data_file = config['OUT_DATA_FILE'] 
	with open(data_file + '.aug_en_ja_de_en.train_pairs.pkl', 'rb') as f:
		train_pairs = pickle.load(f)

	with open(data_file + '.dev_pairs.pkl', 'rb') as f:
		dev_pairs = pickle.load(f)

	#Random shuffle training pairs
	train_pairs = shuffle_train_pairs(train_pairs)

	train_qrels = load_qrels(config['QRELS_TRAIN'])
	dev_qrels = load_qrels(config['QRELS_DEV'])
	

	train_rerank_path = data_file + '.train_rerank.pkl'
	dev_rerank_path = data_file + '.dev_rerank.pkl'
	
	with open(config['TERM_TO_IND'], 'rb') as f:
		term2ind = pickle.load(f)

	metrics = ['P_5', 'MAP(bioasq)', 'GMAP(bioasq)']

	retr_dir = 'res/' + datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S') + '_' + args.out_folder
	
	print(retr_dir)
	os.makedirs(os.path.join(os.getcwd(), retr_dir))

	json_model_params = copy.deepcopy(model_params)
	json_model_params['embed'] = []
	with open(retr_dir+ '/{0}'.format(params_file.split('/')[-1]), 'w') as f:
		json.dump(json_model_params, f, indent=4)

	test_qrels_list = {}
	test_rerank_path_list = {}
	for i in range(5):
		batch = i + 1
		test_qrels_list[batch] = 'data/test_batch_{0}/BioASQ-task6bPhaseB-testset{0}'.format(batch)
		test_rerank_path_list[batch] = 'data/test_batch_{0}/bioasq6.top100.test_rerank.pkl'.format(batch)



	pacrr_train(train_pairs, dev_pairs, train_rerank_path, dev_rerank_path)