import random
import pickle
import dynet as dy
import numpy as np
from gensim.models import KeyedVectors

class DRMM:

	def __init__(self):
		random.seed(1234)

		# Input hyperparameters
		self.hist_size = 29
		self.max_q_len = 30

		# MLP hyperparameters
		self.mlp_layers = 2
		self.hidden_size = 10

		# Model initialization
		self.model = dy.ParameterCollection()
		self.trainer = dy.AdamTrainer(self.model)

		
		self.w_g = self.model.add_parameters((1,))

		self.W_1 = self.model.add_parameters((self.hist_size, self.hidden_size))
		self.b_1 = self.model.add_parameters((1, self.hidden_size))

		if self.mlp_layers > 1:
			self.W_n = []
			self.b_n = []
			for i in range(self.mlp_layers):
				self.W_n.append(self.model.add_parameters((self.hidden_size, self.hidden_size)))
				self.b_n.append(self.model.add_parameters((1, self.hidden_size)))
		
		self.W_last = self.model.add_parameters((self.hidden_size, 1))
		self.b_last = self.model.add_parameters((1))

		self.W_scores = self.model.add_parameters((5, 1))
		self.b_scores = self.model.add_parameters((1))

	def dump_weights(self, w_dir):
		self.model.save(w_dir + '/weights.bin')

	def load_weights(self, w_dir):
		self.model.populate(w_dir + '/weights.bin')


	def predict_pos_neg_scores(self, q_dpos_hists, q_dneg_hists, q_idf, pos_bm25_score, neg_bm25_score, pos_overlap_features, neg_overlap_features):
		
		pos_score = self.scorer(q_dpos_hists, q_idf, pos_bm25_score, pos_overlap_features)
		neg_score = self.scorer(q_dneg_hists, q_idf, neg_bm25_score, neg_overlap_features)

		# return probability of first (relevant) document. 
		return dy.concatenate([pos_score, neg_score])

	def predict_doc_score(self, q_d_hists, q_idf, bm25_score, overlap_features):
		doc_score = self.scorer(q_d_hists, q_idf, bm25_score, overlap_features)
		return doc_score

	def scorer(self, q_d_hists, q_idf, bm25_score, overlap_features):
		
		idf_vec = dy.inputVector(q_idf)
		bm25_score = dy.scalarInput(bm25_score)
		overlap_features = dy.inputVector(overlap_features)
		
		# Pass each query term representation through the MLP
		term_scores = []
		for hist in q_d_hists:
			q_d_hist =  dy.reshape(dy.inputVector(hist), (1, len(hist)))
			hidd_out = dy.rectify(q_d_hist * self.W_1.expr() + self.b_1.expr())
			for i in range(0, self.mlp_layers):
				hidd_out = dy.rectify(hidd_out * self.W_n[i].expr() + self.b_n[i].expr())
			term_scores.append(hidd_out * self.W_last.expr() + self.b_last.expr())
		
		# Term Gating
		gating_weights = idf_vec*self.w_g.expr()
		drmm_score = dy.transpose(dy.concatenate(term_scores)) * dy.reshape(gating_weights, (len(q_idf), 1))
		#doc_score = drmm_score
		doc_score = dy.transpose(dy.concatenate([drmm_score, bm25_score, overlap_features])) * self.W_scores.expr() + self.b_scores

		return doc_score
