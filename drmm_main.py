import os
import json
import pickle
import datetime
import argparse
import subprocess

import numpy as np
import dynet as dy

from tqdm import tqdm
from models.drmm import DRMM


def log(data, filename, cl=False):
    f = open(res_dir + '/{0}'.format(filename), 'a')
    f.write(data + '\n')
    if cl:
        f.write('\n')
    f.close()

def load_qrels(path):

    with open(path , 'r') as f:
        data = json.load(f)

    qrels = {}
    for q in data['questions']:
        qrels[q['id']] = set(q['documents'])
    return qrels

def chunks(l, n):
    chunks_list = []
    for i in range(0, len(l), n):
        chunks_list.append(l[i:i + n])
    return chunks_list

def rerank_query_generator_ram(filename):
        with open(filename, 'rb') as f:
            unpickler = pickle.Unpickler(f)
            i = 0
            while True:
                try:
                    q = unpickler.load()
                    yield q
                except EOFError:
                    break   

def shuffle_train_pairs(train_data_dict):
    num_pairs = train_data_dict['num_pairs']
    inds = np.arange(num_pairs)     
    np.random.shuffle(inds)              
    for k in train_data_dict.keys():
        if isinstance(train_data_dict[k], list) and len(train_data_dict[k]) == train_data_dict['num_pairs']:
            #print(k + ' shuffled.')
            train_data_dict[k] = np.array(train_data_dict[k])[inds]
    return train_data_dict

def results_to_string(prefix, results):
    new_str =  '\t'.join(['{0} {1}: {2}'.format(prefix, m, results[m]) for m in metrics])
    return new_str

def write_bioasq_results_dict(bioasq_res_dict, filename):

    path = '{0}/{1}'.format(res_dir, filename)
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
        ['java', '-Xmx10G', '-cp', '$CLASSPATH:./bioasq6_eval/flat/BioASQEvaluation/dist/BioASQEvaluation.jar', 'evaluation.EvaluatorTask1b', '-phaseA', '-e', '5', golden_file, path],
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


def rerank(data_file, filename, scoring_model, qrels, qrels_file):
    queries_to_rerank = rerank_query_generator_ram(data_file)

    res_dict = {'questions': []}
    pred_batch_size = 100
    for q in tqdm(queries_to_rerank, desc='queries', dynamic_ncols=True):
        
        query = q['token_inds']
        query_idf = q['idf']
        scores = []
        dev_batches = chunks(range(len(q['retrieved_documents']['doc_list'])), pred_batch_size)
        for batch in dev_batches:
            batch_preds = []
            dy.renew_cg() # new computation graph
            for i in batch:
                doc = q['retrieved_documents']['doc_list'][i]
                doc_bm25 = q['retrieved_documents']['doc_normBM25'][i]
                doc_overlap = q['retrieved_documents']['doc_overlap'][i]
                batch_preds.append(scoring_model.predict_doc_score(doc, query_idf, doc_bm25, doc_overlap))
            dy.forward(batch_preds)
            scores += [pred.npvalue()[0] for pred in batch_preds]
        retr_scores = list(zip(q['retrieved_documents']['doc_ids'], scores))
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
    
if __name__ == '__main__':
    
    train_pairs_file = 'data/bioasq.top100.train_pairs.30bins.pkl'
    dev_pairs_file = 'data/bioasq.top100.dev_pairs.30bins.pkl'
    dev_rerank_file = 'data/bioasq.top100.dev_rerank.30bins.pkl'
    
    with open(train_pairs_file, 'rb') as f:
        train_data = pickle.load(f)
    
    with open(dev_pairs_file, 'rb') as f:
        dev_data = pickle.load(f)
    
    
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-config', dest='config_file')
    parser.add_argument('-out', dest='out_file')
    args = parser.parse_args()
    
    print(args.config_file)
    
    config_file = args.config_file
    
    with open(config_file, 'r') as f:
        config = json.load(f)  
    
    train_qrels = load_qrels(config['QRELS_TRAIN'])
    dev_qrels = load_qrels(config['QRELS_DEV'])
    
    metrics = ['map']
    
    drmm_model = DRMM()
    
    train_batch_size = 32
    n_epochs = 100
    best_map = -1
    
    res_dir = 'res/' + datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S') + '_' + args.out_file
    print(res_dir)
    os.makedirs(os.path.join(os.getcwd(), res_dir))
    
    log('epoch|dev_map|dev_gmap|dev_p_5|dev_acc|dev_loss|train_map|train_gmap|train_p_5|train_acc|train_loss', 'log.txt')
    
    best_epoch = -1
    for epoch in range(1, n_epochs+1):
        print('\nEpoch: {0}/{1}'.format(epoch, n_epochs))
        sum_of_losses = 0.0
    
        #Random shuffle training pairs
        train_data = shuffle_train_pairs(train_data)
    
        train_batches = chunks(range(len(train_data['queries'])), train_batch_size)
        pbar = tqdm(total=len(train_data['queries']), mininterval=0, miniters=1, dynamic_ncols=True)
        hits = 0
        logs = {}
        for batch in train_batches:
            dy.renew_cg() # new computation graph
            batch_losses = []
            batch_preds = []
            
            for i in batch:
                #print(i)
                q_dpos_hist = train_data['pos_docs'][i]
                q_dneg_hist = train_data['neg_docs'][i]
                query_idf = train_data['queries_idf'][i]
                pos_bm25 = train_data['pos_docs_normBM25'][i]
                neg_bm25 = train_data['neg_docs_normBM25'][i]
                pos_overlap = train_data['pos_docs_overlap'][i]
                neg_overlap = train_data['neg_docs_overlap'][i]
    
                #print(query_idf)
    
                preds = drmm_model.predict_pos_neg_scores(q_dpos_hist, q_dneg_hist, query_idf, pos_bm25, neg_bm25, pos_overlap, neg_overlap)
                batch_preds.append(preds)
                loss = dy.hinge(preds, 0)
                batch_losses.append(loss)
            batch_loss = dy.esum(batch_losses) / len(batch)
            sum_of_losses += batch_loss.npvalue()[0] # this calls forward on the batch
            for p in batch_preds:
                p_v = p.value()
                if p_v[0] > p_v[1]:
                    hits += 1
            batch_loss.backward()
            drmm_model.trainer.update()
            pbar.update(train_batch_size)
    
        logs['acc'] = hits / len(train_data['queries'])
        logs['loss'] = sum_of_losses / len(train_batches)
        
        val_preds = []
        val_losses = []
        hits = 0
        for i in range(len(dev_data['pos_docs'])):
            q_dpos_hist = dev_data['pos_docs'][i]
            q_dneg_hist = dev_data['neg_docs'][i]
            query_idf = dev_data['queries_idf'][i]
            pos_bm25 = dev_data['pos_docs_normBM25'][i]
            neg_bm25 = dev_data['neg_docs_normBM25'][i]
            pos_overlap = dev_data['pos_docs_overlap'][i]
            neg_overlap = dev_data['neg_docs_overlap'][i]
            preds = drmm_model.predict_pos_neg_scores(q_dpos_hist, q_dneg_hist, query_idf, pos_bm25, neg_bm25, pos_overlap, neg_overlap)
            val_preds.append(preds)
            loss = dy.hinge(preds, 0)
            val_losses.append(loss)
        val_loss = dy.esum(val_losses)
        sum_of_losses += val_loss.npvalue()[0] # this calls forward on the batch
        for p in val_preds:
            p_v = p.value()
            if p_v[0] > p_v[1]:
                hits += 1
    
        
        logs['val_acc'] = hits / len(dev_data['queries'])
        logs['val_loss'] = sum_of_losses / len(dev_data['pos_docs'])
        print('Training loss: {0}'.format(logs['loss']))
        print('Training acc: {0}'.format(logs['acc']))
        print('Dev loss: {0}'.format(logs['val_loss']))
        print('Dev acc: {0}'.format(logs['val_acc']))
        pbar.close()
    
        res = rerank(dev_rerank_file, 'dev_ranking_epoch{0}'.format(epoch), drmm_model, dev_qrels, config['QRELS_DEV'])
    
        print('MAP*: {0}'.format(res['MAP(bioasq)']))
        print('GMAP: {0}'.format(res['GMAP(bioasq)']))
        log('|'.join(list(map(str, [epoch, res['MAP(bioasq)'], res['GMAP(bioasq)'], res['P_5'], logs['val_acc'], logs['val_loss'], 0, 0, 0, logs['acc'], logs['loss']]))), 'log.txt')
    
        is_best_epoch = False
        if res['MAP(bioasq)'] > best_map:
            print('===== Best epoch so far =====')
            best_map = res['MAP(bioasq)']
            is_best_epoch = True
            best_epoch = epoch
            drmm_model.dump_weights(res_dir)
            log('^^^', 'log.txt')
    
    
    drmm_model.load_weights(res_dir)
    
    test_qrels_list = {}
    test_rerank_path_list = {}
    for i in range(1, 6):
        test_qrels_list[i] = 'data/BioASQ-task6bPhaseB-testset{0}'.format(i)
        test_rerank_path_list[i] = 'data/bioasq.top100.test_rerank.b{0}.30bins.pkl'.format(i)
    
    for i in range(1, 6):
        res = rerank(test_rerank_path_list[i], 'test_ranking_batch{0}'.format(i), drmm_model, load_qrels(test_qrels_list[i]), test_qrels_list[i])
    
        print('MAP*: {0}'.format(res['MAP(bioasq)']))
        print('GMAP: {0}'.format(res['GMAP(bioasq)']))
        log('batch|epoch|test_map|test_gmap|test_p_5', 'test_log.txt')
        log('|'.join(list(map(str, [i, best_epoch, res['MAP(bioasq)'], res['GMAP(bioasq)'], res['P_5']]))), 'test_log.txt')
