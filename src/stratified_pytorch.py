import numpy as np
import networkx as nx
import sys
import torch as t
import torch.nn as nn
import torch.nn.functional as F
import time
import itertools
from scipy.special import softmax
from joblib import Parallel, delayed
from tqdm.notebook import tqdm

class Embedding(nn.Module):
    """
    Redefining torch.nn.Embedding (see docs for that function)
    """
    def __init__(self, num_embeddings, embedding_dim, padding_idx=None, _weight=None):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        if padding_idx is not None:
            if padding_idx > 0:
                assert padding_idx < self.num_embeddings, 'Padding_idx must be within num_embeddings'
            elif padding_idx < 0:
                assert padding_idx >= -self.num_embeddings, 'Padding_idx must be within num_embeddings'
                padding_idx = self.num_embeddings + padding_idx
        self.padding_idx = padding_idx

        if _weight is None:
            self.weight = nn.Parameter(t.randn([self.num_embeddings, self.embedding_dim])/np.sqrt(self.num_embeddings))
        else:
            assert list(_weight.shape) == [num_embeddings, embedding_dim], \
                'Shape of weight does not match num_embeddings and embedding_dim'
            self.weight = nn.Parameter(_weight)

        if self.padding_idx is not None:
            with t.no_grad():
                self.weight[self.padding_idx].fill_(0)

    def forward(self, x):
        if self.padding_idx is not None:
            with t.no_grad():
                self.weight[self.padding_idx].fill_(0)                
        return self.weight[x]
    
    def numel(self):
        if self.padding_idx is not None:
            return (self.num_embeddings-1) * self.embedding_dim
        else:
            return self.num_embeddings * self.embedding_dim

class DataLoader():
    """
    Redefining torch.utils.data.DataLoader, see docs for that function
    Done so because it is faster for CPU only use.
    """
    def __init__(self, data, batch_size=None, shuffle=False):
        # data must be a list of tensors
        self.data = data
        self.data_size = data[0].shape[0]
        if shuffle:
            random_idx = np.arange(self.data_size)
            np.random.shuffle(random_idx)
            self.data = [item[random_idx] for item in self.data]
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.counter = 0
        self.stop_iteration_flag = False

    def __iter__(self):
        return self

    def __next__(self):
        if self.stop_iteration_flag:
            self.stop_iteration_flag = False
            raise StopIteration()
        if self.batch_size is None or self.batch_size >= self.data_size:
            self.stop_iteration_flag = True
            return self.data
        else:
            i = self.counter
            bs = self.batch_size
            self.counter += 1
            batch = [item[i * bs:(i + 1) * bs] for item in self.data]
            if self.counter * bs >= self.data_size:
                self.counter = 0
                self.stop_iteration_flag = True
                if self.shuffle:
                    random_idx = np.arange(self.data_size)
                    np.random.shuffle(random_idx)
                    self.data = [item[random_idx] for item in self.data]
            return batch

class ENDModel(nn.Module):
    '''
    Modeling partial orders of an m-set.
    '''
    def __init__(self, num_items, alternative_codex, ism, 
                 max_length=0, position_dependent_end=True,
                 mnl_fixed_effects=False, school_choice=False,
                 item_to_school=None, item_to_program=None, 
                 school_names=None, program_type_names=None, 
                 mnl_linear_terms=False, mnl_covariates=None, 
                 mnl_context=False, embedding_dim=5, forward_dependent=False, top_k=False,
                 mnl_k=1, rank_reg=0.0, num_ranks=5):
        super().__init__()
        self.num_items = num_items        
        self.alternative_codex = alternative_codex
        self.ism = ism
        self.position_dependent_end = position_dependent_end
        self.mnl_fixed_effects = mnl_fixed_effects
        self.school_choice = school_choice
        self.item_to_school = item_to_school
        self.item_to_program = item_to_program
        self.school_names = school_names
        self.program_type_names = program_type_names
        self.mnl_linear_terms = mnl_linear_terms
        self.mnl_covariates = mnl_covariates
        self.mnl_context = mnl_context
        self.embedding_dim = embedding_dim
        self.forward_dependent = forward_dependent
        self.top_k = top_k
        self.max_length = max_length
        
        self.mnl_k = min(mnl_k, max_length)
        self.rank_reg = rank_reg

        self.num_ranks = num_ranks
        if position_dependent_end:
            self.end_fixed_effects = Embedding(num_embeddings=max_length+1,
                                               embedding_dim=1,
                                               padding_idx=0) # for positions 2 through m+1
        else:
            self.end_fixed_effects = Embedding(num_embeddings=mnl_k+1,
                                               embedding_dim=1,
                                               padding_idx=0) # for positions 2 through mnl_k+1
        self.mnl_model = RankStratifiedMNL(num_items=num_items, alternative_codex=alternative_codex, ism=ism, 
                                           fixed_effects=mnl_fixed_effects, school_choice=school_choice,
                                           item_to_school=item_to_school, item_to_program=item_to_program,
                                           school_names=school_names, program_type_names=program_type_names, 
                                           linear_terms=mnl_linear_terms, covariates=mnl_covariates, 
                                           context=mnl_context, forward_dependent=forward_dependent, 
                                           embedding_dim=embedding_dim, k=mnl_k, lambda_reg=rank_reg, 
                                           top_k=top_k, num_ranks=num_ranks)
            
    def numel(self):
        # Calculate the total number of parameters with gradients
        return self.end_fixed_effects.numel() + self.mnl_model.numel()
        
    def choice_utilities(self, x, x_extra, mnl_covariates=None):
        """
        Inputs: 
        x - (batch_size, maximum sequence length, 2) array of item indices involved in the choice set or chosen set of interest.
        x_extra - (batch_size, 3) array, columns representing chooser, length of choice set, and length of chosen set for each observation in x.
        covariates - (num_agents, num_items, num_features)-sized numpy array of covariates
        inf_weight - used to "zero out" padding terms. Should not be changed.
        """    
        batch_size, seq_len, _ = x.size()
        mnl_log_likelihoods = t.zeros((batch_size, seq_len))
        
        if (mnl_covariates is None):
            mnl_covariates=self.mnl_covariates
        else:
            num_agents, num_alternatives, num_features = mnl_covariates.shape
            pad_mat = np.zeros((num_agents, num_features), dtype=np.float32)
            mnl_covariates = np.hstack([mnl_covariates, pad_mat[:,None,:]]) # add zeros matrix to second dimension of covariates (column-wise)
            mnl_covariates = t.from_numpy(mnl_covariates)

        mnl_utilities = self.mnl_model(x, x_extra, covariates=mnl_covariates, return_utilities=True) # batch_size, seq_len
        if self.position_dependent_end:
            positions = np.minimum(x_extra[:,2], self.end_fixed_effects.num_embeddings-1)
            end_utils = t.full(positions.shape, fill_value=-np.inf)
            try:
                end_utils[positions>0] = self.end_fixed_effects(positions[positions>0]).flatten()
            except:
                import pdb; pdb.set_trace()
        else:
            positions = x_extra[:,2]
            positions[positions>self.mnl_k] = self.mnl_k
            end_utils = t.full(positions.shape, fill_value=-np.inf)
            end_utils[positions>0] = self.end_fixed_effects(positions[positions>0]).flatten()
        utilities = t.hstack([mnl_utilities, end_utils[:,None]])

        return utilities
    
    def END_choice_utilities(self, x, x_extra, mnl_covariates=None):
        """
        Inputs: 
        x - (batch_size, maximum sequence length, 2) array of item indices involved in the choice set or chosen set of interest.
        x_extra - (batch_size, 3) array, columns representing chooser, length of choice set, and length of chosen set for each observation in x.
        covariates - (num_agents, num_items, num_features)-sized numpy array of covariates
        inf_weight - used to "zero out" padding terms. Should not be changed.
        """    
        if self.position_dependent_end:
            positions = np.minimum(x_extra[:,4], self.end_fixed_effects.num_embeddings-1)
        else:
            positions = np.minimum(x_extra[:,4], self.mnl_k)
        remaining_sets = x[...,2:]
        remaining_utilities = self.mnl_model(remaining_sets, x_extra, return_utilities=True)
        end_utils = t.full(positions.shape, fill_value=-np.inf)
        end_utils[positions>0] = self.end_fixed_effects(positions[positions>0]).flatten()
        utilities = t.hstack([remaining_utilities, end_utils[:,None]])

        return utilities

    def forward(self, x, x_extra, mnl_covariates=None):
        """
        Forward propagation, computing y_hat
        
        Inputs: 
        x - (batch_size, maximum sequence length, 2) array of item indices involved in the choice set or chosen set of interest.
        x_extra - (batch_size, 3) array, columns representing chooser, length of choice set, and length of chosen set for each observation in x.
        covariates - (num_agents, num_items, num_features)-sized numpy array of covariates
        inf_weight - used to "zero out" padding terms. Should not be changed.
        """    
        utilities = self.choice_utilities(x, x_extra, mnl_covariates)
        return F.log_softmax(utilities, 1).squeeze()         
    
    def repeated_selection(self, agent, entry):
        length = len(entry)
        univ = list(np.arange(self.num_items))
        padded_choice_sets = np.full((length, self.num_items), fill_value=self.num_items)
        padded_choice_sets[0,:] = np.arange(self.num_items)
        padded_context_sets = np.full((length, self.num_items), fill_value=self.num_items)
        slots_chosen = np.zeros((length,))
        
        for idx, item in enumerate(entry):
            slots_chosen[idx] = univ.index(item)
            padded_choice_sets[idx,:len(univ)] = univ
            padded_context_sets[idx,:idx] = entry[:idx]
            univ.remove(item)
        remaining_set = univ
        padded_remaining_sets = np.full((length, self.num_items), fill_value=self.num_items)
        padded_remaining_sets[:, :len(remaining_set)] = np.array(remaining_set)[None, :]
        
        x_extra = np.stack([np.full((length,), fill_value=agent), # whose choice
                            self.num_items - np.arange(length), # choice set lengths
                            np.arange(length), # chosen set lengths
                            np.full((length,), fill_value=len(remaining_set)), # remaining set lengths
                            np.full((length,), fill_value=length)], axis=1) # lengths
        x = np.stack([padded_choice_sets, 
                      padded_context_sets, 
                      padded_remaining_sets], axis=-1)
        return list(map(t.from_numpy, [x, x_extra, slots_chosen]))
    
    def inner_loop(self, i, length, permutation, mnl_covariates, return_log_prob=False):
        x, x_extra, y = self.repeated_selection(i, permutation)
        choice_utils = self.choice_utilities(x, x_extra, mnl_covariates)
        y_hat = F.log_softmax(choice_utils, 1).squeeze()  

        end_utils = self.END_choice_utilities(x, x_extra, mnl_covariates)
        end_y_hat = F.log_softmax(end_utils, 1).squeeze()
        end_target = t.full((end_y_hat.shape[0],), end_y_hat.shape[1]-1, dtype=t.long)
        log_probabilities = y_hat[np.arange(y_hat.shape[0]), y.long()].sum() + (end_y_hat/length)[np.arange(y_hat.shape[0]), end_target].sum()
        return log_probabilities if return_log_prob else t.exp(log_probabilities)

    
    def marginal_length_likelihood(self, lengths, mnl_covariates=None, inf_weight=float('-inf')):
        """
        Compute the marginal length likelihood, pi(k) = sum_{Q\in\Omega_k} pi(Q)

        Inputs: 
        lengths - (bs, ) array of integers representing list lengths
        covariates - (bs, num_features) array of covariates, each row representing an agent
        """
        max_len = lengths.max()
        if not self.school_choice:
            n, m = max_len, np.math.factorial(max_len)
            nll_matrix = np.zeros((n,m))
            
            for i in range(n):
                permutations = itertools.permutations(np.arange(self.num_items), i+1)
                # likelihoods = []
                for j, permutation in enumerate(permutations):
                    nll_matrix[i,j] = self.inner_loop(0, i+1, permutation, mnl_covariates, return_log_prob=True)
            nlls = -t.logsumexp(nll_matrix,1)    
            return nlls[lengths].mean()
            # return nll_matrix[lengths].mean()
        else:    
            n, m = lengths.size, np.math.factorial(max_len)
            nll_matrix = np.zeros((n,))

            for i, length in enumerate(lengths):
                permutations = itertools.permutations(np.arange(self.num_items), length)
                likelihoods = []
                for j, permutation in enumerate(permutations):
                    likelihoods.append(self.inner_loop(i, length, permutation, mnl_covariates))
                # likelihoods = Parallel(n_jobs=-4)(delayed(self.inner_loop)(i, length, permutation, mnl_covariates) for permutation in permutations)
                nll_matrix[i] = -t.log(sum(likelihoods))

            return nll_matrix.mean()
        
    def loss_func(self, y_hat, y, x=None, x_extra=None, mnl_covariates=None, train=True):
        """
        Evaluates the model
        Inputs: 
        y_hat - the log softmax values that come from the forward function
        y - actual labels - the choice in a set (i-th entry must be less than x_extra[i,1])
        x_extra - observation metadata (same as in forward function)
        train - bool. if True, returns full training loss (incl. regularization loss)
        """
        if (mnl_covariates is None):
            mnl_covariates=self.mnl_covariates
        else:
            num_agents, num_alternatives, num_features = mnl_covariates.shape
            pad_mat = np.zeros((num_agents, num_features), dtype=np.float32)
            mnl_covariates = np.hstack([mnl_covariates, pad_mat[:,None,:]]) # add zeros matrix to second dimension of covariates (column-wise)
            mnl_covariates = t.from_numpy(mnl_covariates)

        mnl_loss, mnl_terms = self.mnl_model.loss_func(y_hat, y, x, x_extra, train)
        
        lengths = x_extra[:,4]
        end_utilities = self.END_choice_utilities(x, x_extra, mnl_covariates)
        end_likelihoods = F.log_softmax(end_utilities, 1).squeeze()
        end_loss = F.nll_loss(end_likelihoods.div(lengths[:,None]), 
                              t.full((end_likelihoods.shape[0],), end_likelihoods.shape[1]-1, 
                                     dtype=t.long))
        
        tl = mnl_loss + end_loss
        return (tl, [mnl_loss, end_loss, mnl_terms[-1]])

    def acc_func(self, y_hat, y, x_lengths=None):
        return (y_hat.argmax(1).int() == y.int()).float().mean()
    
    def sample_preferences(self, num_samples=1, mnl_covariates=None):
        if self.mnl_linear_terms & (mnl_covariates is None):
            covariates = self.mnl_covariates
            num_agents, num_alternatives, num_features = covariates.shape
        elif self.mnl_linear_terms & (mnl_covariates is not None):
            num_agents, num_alternatives, num_features = mnl_covariates.shape
            covariates = mnl_covariates
        else:
            num_agents, num_alternatives = num_samples, self.num_items
            covariates=self.mnl_covariates

        preferences = t.full((num_agents, num_alternatives), fill_value=self.num_items+1)
        row_incomplete = t.full((num_agents,), fill_value=True)
        lengths = t.full((num_agents,), fill_value=num_alternatives)
        i = 0
        while any(row_incomplete) and i<num_alternatives:
            num_incomplete = row_incomplete.sum()
            utilities = self.mnl_model.models[min(i, self.mnl_k-1)].compute_utilities(covariates=covariates)
            if utilities.shape[0]==1:
                utilities = utilities.tile((num_agents,1))
            
            if i==0:
                end_utils = t.full((num_agents, 1), fill_value=-np.inf)
            else:
                end_utils = t.full((num_agents, 1), fill_value=self.end_fixed_effects(min(i,self.max_length)).item()) if self.position_dependent_end else t.full((num_agents, 1), fill_value=self.end_fixed_effects(min(i,self.mnl_k)).item())
                
            utilities = t.hstack([utilities, end_utils])
            utilities[t.arange(utilities.shape[0])[row_incomplete][:,None], preferences[row_incomplete,:i]] = -np.inf
            softmax_probs = softmax(utilities, axis=1)
            sampled_prefs = t.multinomial(softmax_probs, num_samples=1).flatten()
            lengths[row_incomplete&(sampled_prefs==(utilities.shape[1]-1))] = i
            row_incomplete[sampled_prefs==(utilities.shape[1]-1)] = False
            # try:
            preferences[row_incomplete, i] = sampled_prefs[row_incomplete]
            # except:
            #     import pdb; pdb.set_trace()
            i += 1
        return lengths.numpy(), preferences.numpy()

        
class JointModel(nn.Module):
    '''
    Modeling partial orders of an m-set. 
    Partial orders arise from a two step process: first sampling list length, then sampling the ordering given the list length.
    Comprised of a length model (Poisson) and a conditional preference model (stratified MNL).
    '''
    def __init__(self, num_items, alternative_codex, ism, 
                 max_length=1, dependent=True, categorical=False,
                 mnl_fixed_effects=False, school_choice=False,
                 item_to_school=None, item_to_program=None, 
                 school_names=None, program_type_names=None, 
                 mnl_linear_terms=False, mnl_covariates=None, 
                 mnl_context=False, embedding_dim=5, forward_dependent=False, top_k=False,
                 mnl_k=1, length_k=1, poisson_covariates=None,
                 rank_reg=0.0, length_reg=0.0, regularization_order=2, num_ranks=5):
        super().__init__()
        self.num_items = num_items
        self.alternative_codex = alternative_codex
        self.ism = ism
        self.dependent = dependent
        self.categorical = categorical
            
        self.mnl_fixed_effects = mnl_fixed_effects
        self.school_choice = school_choice
        self.item_to_school = item_to_school
        self.item_to_program = item_to_program
        self.school_names = school_names
        self.program_type_names = program_type_names
        self.mnl_linear_terms = mnl_linear_terms
        self.mnl_covariates = mnl_covariates
        self.mnl_context = mnl_context
        self.embedding_dim = embedding_dim
        self.forward_dependent = forward_dependent
        self.top_k = top_k
        
        self.max_length = max_length
        self.length_k = min(length_k, max_length)
        self.mnl_k = min(mnl_k, max_length)
        self.rank_reg = rank_reg
        self.length_reg = length_reg

        self.num_ranks = num_ranks
        if self.categorical:
            self.length_logits = Embedding(num_embeddings=num_items, embedding_dim=1)
        else:
            self.poisson_intercept = t.nn.parameter.Parameter(data=t.zeros([1,1]))
            if poisson_covariates is None:
                self.poisson_covariates = poisson_covariates
                self.poisson_num_features = 0
            else:
                self.poisson_num_features = poisson_covariates.shape[1]
                self.poisson_covariates = t.from_numpy(poisson_covariates)
                self.poisson_beta = t.nn.Linear(self.poisson_num_features, 1, bias=False)
        
        self.__build_model()
        
    def __build_model(self):
        """
        Helper function to initialize the model
        """
        if self.dependent:
            self.models = nn.ModuleDict({'{},{}'.format(str(i), str(j)): MNL(num_items=self.num_items, 
                                                                             alternative_codex=self.alternative_codex,
                                                                             ism=self.ism, 
                                                                             fixed_effects=self.mnl_fixed_effects,
                                                                             school_choice=self.school_choice,
                                                                             item_to_school=self.item_to_school, 
                                                                             item_to_program=self.item_to_program,
                                                                             school_names=self.school_names, 
                                                                             program_type_names=self.program_type_names,
                                                                             linear_terms=self.mnl_linear_terms, 
                                                                             covariates=self.mnl_covariates, 
                                                                             context=self.mnl_context, 
                                                                             forward_dependent=self.forward_dependent, 
                                                                             embedding_dim=self.embedding_dim, 
                                                                             top_k=self.top_k, 
                                                                             num_ranks=self.num_ranks)
                                         for i in range(self.length_k) 
                                         for j in range(self.mnl_k)})
            self.__build_G()
        else:
            self.model = RankStratifiedMNL(num_items=self.num_items, 
                                           alternative_codex=self.alternative_codex,
                                           ism=self.ism, 
                                           fixed_effects=self.mnl_fixed_effects, 
                                           school_choice=self.school_choice,
                                           item_to_school=self.item_to_school, 
                                           item_to_program=self.item_to_program,
                                           school_names=self.school_names, 
                                           program_type_names=self.program_type_names,
                                           linear_terms=self.mnl_linear_terms, 
                                           covariates=self.mnl_covariates, 
                                           context=self.mnl_context, 
                                           forward_dependent=self.forward_dependent, 
                                           embedding_dim=self.embedding_dim, 
                                           k=self.mnl_k, 
                                           lambda_reg=self.rank_reg, 
                                           top_k=self.top_k)
    def __build_G(self):
        G = nx.grid_2d_graph(self.length_k, self.mnl_k)
        for n1, n2 in G.edges:
            (col1, row1), (col2, row2) = n1, n2
            if (row1>col1) or (row2>col2): # either node is below the diagonal
                G[n1][n2]['weight'] = 0.0
            elif (np.abs(row1-row2) == 1) & (col1==col2): #rows differ by 1, and same column
                G[n1][n2]['weight'] = self.rank_reg
            elif (np.abs(col1-col2) == 1) & (row1==row2): #columns differ by 1, same row
                G[n1][n2]['weight'] = self.length_reg
            else:
                continue
        self.G = G
                
    def draw_regularization_graph(self, rank_color='r', length_color='b'):
        for n1, n2 in self.G.edges:
            (col1, row1), (col2, row2) = n1, n2
            if (row1>col1) or (row2>col2): # either node is below the diagonal
                G[n1][n2]['color'] = 'k'
            elif (np.abs(row1-row2) == 1) & (col1==col2): #rows differ by 1, and same column
                G[n1][n2]['color'] = rank_color
            elif (np.abs(col1-col2) == 1) & (row1==row2): #columns differ by 1, same row
                G[n1][n2]['color'] = length_color
            else:
                continue
        plt.figure(figsize=(30,1))
        edges = self.G.edges()
        colors = [self.G[u][v]['color'] for u,v in edges]
        weights = [self.G[u][v]['weight'] for u,v in edges]
        pos = {(col,row): (col,-row) for col,row in self.G.nodes()}
        nx.draw(self.G, pos, 
                node_size=15, 
                edge_color=colors, 
                width=weights)
            
    def mnl_log_likelihoods(self, x, x_extra, mnl_covariates=None):
        """
        Forward propagation, computing y_hat
        
        Inputs: 
        x - (batch_size, maximum sequence length, 2) array of item indices involved in the choice set or chosen set of interest.
        x_extra - (batch_size, 3) array, columns representing chooser, length of choice set, and length of chosen set for each observation in x.
        covariates - (num_agents, num_items, num_features)-sized numpy array of covariates
        inf_weight - used to "zero out" padding terms. Should not be changed.
        """    
        batch_size, seq_len, _ = x.size()
        mnl_log_likelihoods = t.zeros((batch_size, seq_len))
        if (mnl_covariates is None):
            mnl_covariates=self.mnl_covariates
        else:
            num_agents, num_alternatives, num_features = mnl_covariates.shape
            pad_mat = np.zeros((num_agents, num_features), dtype=np.float32)
            mnl_covariates = np.hstack([mnl_covariates, pad_mat[:,None,:]]) # add zeros matrix to second dimension of covariates (column-wise)
            mnl_covariates = t.from_numpy(mnl_covariates)

        if self.dependent:
            for i in range(self.length_k-1):
                for j in range(min(i+1, self.mnl_k-1)):
                    rows = (x_extra[:,4]==i+1)&(x_extra[:,2]==j)
                    mnl_log_likelihoods[rows] = self.models['{},{}'.format(str(i), str(j))](x[rows], x_extra[rows], mnl_covariates)
                if self.mnl_k<=i+1:
                    rows = (x_extra[:,4]==i+1)&(x_extra[:,2]>=(self.mnl_k-1))
                    mnl_log_likelihoods[rows] = self.models['{},{}'.format(str(i), str(self.mnl_k-1))](x[rows], x_extra[rows], mnl_covariates)
            for j in range(min(self.length_k, self.mnl_k-1)):
                rows = (x_extra[:,4]>=self.length_k)&(x_extra[:,2]==j)
                mnl_log_likelihoods[rows] = self.models['{},{}'.format(str(self.length_k-1), str(j))](x[rows], x_extra[rows], mnl_covariates)
            if self.mnl_k<=self.length_k:
                rows = (x_extra[:,4]>=self.length_k)&(x_extra[:,2]>=(self.mnl_k-1))
                mnl_log_likelihoods[rows] = self.models['{},{}'.format(str(self.length_k-1), str(self.mnl_k-1))](x[rows], x_extra[rows], mnl_covariates)
        else:
            mnl_log_likelihoods = self.model(x, x_extra, mnl_covariates)
                    
        return mnl_log_likelihoods

    def length_log_likelihoods(self, x, x_extra, poisson_covariates=None):
        """
        Forward propagation, computing y_hat
        
        Inputs: 
        x - (batch_size, maximum sequence length, 2) array of item indices involved in the choice set or chosen set of interest.
        x_extra - (batch_size, 3) array, columns representing chooser, length of choice set, and length of chosen set for each observation in x.
        covariates - (num_agents, num_items, num_features)-sized numpy array of covariates
        inf_weight - used to "zero out" padding terms. Should not be changed.
        """    
        batch_size, seq_len, _ = x.size()
        if self.categorical:
            log_likelihoods = F.log_softmax(self.length_logits.weight.flatten())[x_extra[:,4]-1]
            log_likelihoods = log_likelihoods.div(x_extra[:,4])
            return log_likelihoods[:,None]
        else:
            if poisson_covariates is None:
                poisson_covariates=self.poisson_covariates
            else:
                poisson_covariates = t.from_numpy(poisson_covariates)

            # try:
            poisson_lambdas = self.poisson_intercept[:,None] if poisson_covariates is None else self.poisson_intercept[:,None] + self.poisson_beta(poisson_covariates[x_extra[:,0]])
            # except:
            #     import pdb; pdb.set_trace()
            
            ##### SHOULD MODEL x_extra[:,4]==1 AS Pr(X==1 | X==0) 
            poisson_log_likelihoods = (-nn.functional.poisson_nll_loss(t.exp(poisson_lambdas.squeeze()), x_extra[:,4], full=True, log_input=False, reduction='none')).div(x_extra[:,4])

            return poisson_log_likelihoods[:,None]

    def forward(self, x, x_extra, poisson_covariates=None, mnl_covariates=None):
        """
        Forward propagation, computing y_hat
        
        Inputs: 
        x - (batch_size, maximum sequence length, 2) array of item indices involved in the choice set or chosen set of interest.
        x_extra - (batch_size, 3) array, columns representing chooser, length of choice set, and length of chosen set for each observation in x.
        covariates - (num_agents, num_items, num_features)-sized numpy array of covariates
        inf_weight - used to "zero out" padding terms. Should not be changed.
        """    
        mnl_log_likelihood = self.mnl_log_likelihoods(x, x_extra, mnl_covariates)
        length_log_likelihood = self.length_log_likelihoods(x, x_extra, poisson_covariates)
        return mnl_log_likelihood + length_log_likelihood
    
    def reg(self):
        '''
        Computes regularization term for loss. 
        NS terms regulate down rank (if self.mnl_k>1). 
        EW terms regulate across length.

        self.school_logits & self.program_logits are ModuleLists of Embeddings
        self.beta is a ModuleList of t.nn.Linear()
        self.targets & self.context are ModuleLists of Embeddings
        self.order_reg is the order of regularization (either L1 or L2 in this case)
        self.k is the number of stratified models learned (ie. length of self.beta)
        '''
        L = t.from_numpy(nx.laplacian_matrix(self.G).toarray().astype(np.float32))
        
        if self.school_choice:
            theta_school_fe = t.zeros((1, L.shape[0])) if not self.mnl_fixed_effects else t.stack([model.school_logits.weight for model in self.models.values()], axis=1).squeeze()
            theta_program_fe = t.zeros((1, L.shape[0])) if not self.mnl_fixed_effects else t.stack([model.program_logits.weight for model in self.models.values()], axis=1).squeeze()
            fe_reg = 0.5*t.trace(theta_school_fe@L@theta_school_fe.T) + 0.5*t.trace(theta_program_fe@L@theta_program_fe.T)
        else:
            theta_fe = t.zeros((1, L.shape[0])) if not self.mnl_fixed_effects else t.stack([model.logits.weight for model in self.models.values()], axis=1).squeeze()
            fe_reg = 0.5*t.trace(theta_fe@L@theta_fe.T)

        
        theta_linear = t.zeros((L.shape[0],1)) if not self.mnl_linear_terms else t.stack([model.beta.weight for model in self.models.values()], axis=1).squeeze()
        linear_reg = 0.5*t.trace(theta_linear.T@L@theta_linear)

        theta_target = t.zeros((1, L.shape[0])) if not self.mnl_context else t.stack([model.target_embeddings.weight.flatten() for model in self.models.values()], axis=1).squeeze()
        theta_context = t.zeros((1, L.shape[0])) if not self.mnl_context else t.stack([model.context_embeddings.weight.flatten() for model in self.models.values()], axis=1).squeeze()
        context_reg = 0.5*t.trace(theta_target@L@theta_target.T) + 0.5*t.trace(theta_context@L@theta_context.T)
        
        return fe_reg + linear_reg + context_reg
    
    def loss_func(self, y_hat, y, x=None, x_extra=None, mnl_covariates=None, poisson_covariates=None, train=True):
        """
        Evaluates the model
        Inputs: 
        y_hat - the log softmax values that come from the forward function
        y - actual labels - the choice in a set (i-th entry must be less than x_extra[i,1])
        x_extra - observation metadata (same as in forward function)
        train - bool. if True, returns full training loss (incl. regularization loss)
        """
        if self.dependent:
            mnl_log_likelihood = self.mnl_log_likelihoods(x, x_extra, mnl_covariates)
            length_log_likelihood = self.length_log_likelihoods(x, x_extra, poisson_covariates)
            terms = [F.nll_loss(mnl_log_likelihood, y.long()), -length_log_likelihood.mean()]

            tl = F.nll_loss(y_hat, y.long())
            if ((self.mnl_k>1) | (self.length_k>1)) & train:
                rl = self.reg()
                terms.append(rl)
                return (tl + rl, terms)
            else:
                terms.append(t.tensor(0.))
                return (tl, terms)
        else:
            mnl_log_likelihood = self.model(x, x_extra, mnl_covariates)
            length_log_likelihood = self.length_log_likelihoods(x, x_extra, poisson_covariates)
            terms = [F.nll_loss(mnl_log_likelihood, y.long()), -length_log_likelihood.mean()]
            loss, mnl_terms = self.model.loss_func(y_hat, y, x, x_extra, train)
            terms.append(mnl_terms[-1])
            return loss, terms
            
    def acc_func(self, y_hat, y, x_lengths=None):
        return (y_hat.argmax(1).int() == y.int()).float().mean()

    def numel(self):
        # Calculate the total number of parameters with gradients
        total_parameters = self.poisson_num_features + 1 if not self.categorical else self.length_logits.numel()
        if self.dependent:
            total_parameters += sum(model.numel() for model in self.models.values())
        else:
            total_parameters += self.model.numel()
        return total_parameters
    
    def sample_lengths(self, num_samples=1, covariates=None):        
        if self.categorical:
            softmax_probs = softmax(self.length_logits.weight.detach().numpy().flatten())
            lengths = t.multinomial(t.from_numpy(softmax_probs), num_samples=num_samples, replacement=True)+1
            return lengths.flatten()
        else:
            if self.poisson_num_features == 0:
                poisson_lambdas = t.full((num_samples, 1), fill_value=self.poisson_intercept.data.item())
            elif covariates is None:
                covariates = self.poisson_covariates
                poisson_lambdas = self.poisson_intercept[:,None] + self.poisson_beta(covariates)
            else:
                covariates = t.from_numpy(covariates)
                poisson_lambdas = self.poisson_intercept[:,None] + self.poisson_beta(covariates)
            lengths = t.poisson(t.exp(poisson_lambdas).squeeze())
            lengths[lengths==0] = 1
            lengths[lengths>self.num_items] = self.num_items
            return lengths.int().flatten()
    
    def sample_preferences(self, num_samples=1, length_covariates=None, mnl_covariates=None):
        lengths = self.sample_lengths(num_samples, length_covariates)
        num_agents = lengths.numel()
        preferences = t.full((num_agents, self.num_items), fill_value=self.num_items+1)
        if self.dependent:
            for i in range(self.length_k-1):
                # Select the rows and covariates that correspond to length i
                length = i+1
                rows = lengths==length
                num_rows = rows.sum()
                if self.mnl_linear_terms & (mnl_covariates is not None):
                    covariates = mnl_covariates[rows]
                elif self.mnl_linear_terms & (mnl_covariates is None):
                    covariates = self.mnl_covariates[rows]
                else:
                    covariates = mnl_covariates
                    
                # Fill rows to length or to self.mnl_k-1, whichever comes first
                for j in range(min(length, self.mnl_k-1)):
                    utilities = self.models['{},{}'.format(str(i), str(j))].compute_utilities(covariates=covariates)
                    if utilities.shape[0]==1:
                        utilities = utilities.tile((num_rows,1))
                    utilities[t.arange(utilities.shape[0])[:,None], preferences[rows,:j]]=-np.inf
                    softmax_probs = softmax(utilities, axis=1)
                    preferences[rows, j] = t.multinomial(softmax_probs, num_samples=1).flatten()

                # Fill remainder of length - self.mnl_k
                if self.mnl_k<=length:
                    utilities = self.models['{},{}'.format(str(i), str(self.mnl_k-1))].compute_utilities(covariates=covariates)
                    if utilities.shape[0]==1:
                        utilities = utilities.tile((rows.sum(),1))
                    utilities[t.arange(utilities.shape[0])[:,None], preferences[rows,:(self.mnl_k-1)]]=-np.inf
                    softmax_probs = softmax(utilities, axis=1)
                    preferences[rows, (self.mnl_k-1):length] = t.multinomial(softmax_probs, num_samples=length-(self.mnl_k-1), replacement=False)
            
            # Select the rows and covariates that correspond to length >= self.length_k
            rows = lengths>=self.length_k
            num_rows = rows.sum()
            if self.mnl_linear_terms & (mnl_covariates is not None):
                covariates = mnl_covariates[rows]
            elif self.mnl_linear_terms & (mnl_covariates is None):
                covariates = self.mnl_covariates[rows]
            else:
                covariates = mnl_covariates
                
            # Fill rows to self.length_k or self.mnl_k-1, whichever comes first
            for j in range(min(self.length_k, self.mnl_k-1)):
                utilities = self.models['{},{}'.format(str(self.length_k-1), str(j))].compute_utilities(covariates=covariates)
                if utilities.shape[0]==1:
                    utilities = utilities.tile((num_rows,1))
                utilities[t.arange(utilities.shape[0])[:,None], preferences[rows,:j]]=-np.inf
                softmax_probs = softmax(utilities, axis=1)
                preferences[rows, j] = t.multinomial(softmax_probs, num_samples=1).flatten()
            
            # Fill remainder of self.length_k - self.mnl_k-1
            if self.mnl_k<=self.length_k:
                utilities = self.models['{},{}'.format(str(self.length_k-1), str(self.mnl_k-1))].compute_utilities(covariates=covariates)
                if utilities.shape[0]==1:
                    utilities = utilities.tile((num_rows,1))
                utilities[t.arange(utilities.shape[0])[:,None], preferences[rows,:(self.mnl_k-1)]]=-np.inf
                softmax_probs = softmax(utilities, axis=1)
                bool_row = 0
                for ind, boolean in enumerate(rows):
                    if not boolean:
                        continue
                    length = lengths[ind]
                    # try:
                    preferences[ind, (self.mnl_k-1):length] = t.multinomial(softmax_probs[bool_row], num_samples=length-(self.mnl_k-1), replacement=False)
                    # except:
                    #     import pdb; pdb.set_trace()
                    bool_row+=1
        else:
            if self.mnl_linear_terms & (mnl_covariates is None):
                covariates = self.mnl_covariates
            else:
                covariates = mnl_covariates
            preferences = self.model.sample_preferences(num_samples=num_samples, covariates=mnl_covariates)
            try:
                preferences[t.arange(preferences.shape[1])[None, :] >= lengths[:,None]] = self.num_items+1
            except:
                import pdb; pdb.set_trace()
            
        return lengths.numpy(), preferences.numpy()

        
class RankStratifiedMNL(nn.Module):
    def __init__(self, num_items, alternative_codex, ism=True, 
                 fixed_effects=False, school_choice=False,
                 item_to_school=None, item_to_program=None,
                 school_names=None, program_type_names=None,
                 linear_terms=False, covariates=None, 
                 context=False, forward_dependent=False, embedding_dim=5, 
                 k=1, lambda_reg=1.0, order_reg=2, 
                 top_k=None, num_ranks=5):
        """
        Initializes the rank-stratified MNL model. 
        ModuleList of k MNL models.
        MNL utility can include fixed-effects, linear terms, and/or context-effects.
        
        Inputs: 
        num_items - int, total number of items in the choice system modeled
        ism - bool, True if dataset is multi-set, in which case padding is used
        fixed_effects - bool, if True, item-level fixed effects are used in utility
        item_to_school - array mapping offerings to their respective school index (only needed if fixed_effects is True)
        item_to_program - array mapping offerings to their respective program type index (only needed if fixed_effects is True)
        linear_terms = bool, if True, linear terms are used in utility
        covariates - (n_obs, num_items, num_features) array of covariates
        context - bool, if True, context-effects are used in utility (ie. model is a CDM)
        forward_dependent - bool, if True, forward-dependent CDM is used, else backward-dependent (unused in paper)
        embedding_dim - int > 0, embedding dimension of the low-rank CDM.
        top_k - int or None, rank of the last considered context effect in the top-k CDM (only relevant if forward_dependent is False)
        k - int > 0, number of rank-stratification buckets
        lambda_reg - float, weight of Laplacian regularization
        order_reg - int > 0, order of Laplacian regularization (always 2 in paper)
        num_ranks - int > 0, number of rank positions to return loss.
        """
        super().__init__()
        self.num_items = num_items
        self.alternative_codex = alternative_codex
        self.ism = ism
        self.school_choice = school_choice
        self.school_names = school_names
        self.program_type_names = program_type_names
        self.item_to_school = item_to_school
        self.item_to_program_type = item_to_program
        self.fixed_effects = fixed_effects
        self.linear_terms = linear_terms
        self.covariates = covariates
        self.context = context
        
        self.k = k
        self.lambda_reg = lambda_reg
        self.order_reg = order_reg
        self.models = nn.ModuleList([MNL(num_items=num_items, alternative_codex=alternative_codex, 
                                         ism=ism, school_choice=school_choice, 
                                         fixed_effects=fixed_effects, item_to_school=item_to_school,
                                         school_names=school_names, program_type_names=program_type_names,
                                         item_to_program=item_to_program,
                                         linear_terms=linear_terms, covariates=covariates, 
                                         context=context, forward_dependent=forward_dependent, embedding_dim=embedding_dim, 
                                         top_k=top_k, num_ranks=num_ranks) for i in range(k)])        
        self.num_ranks = num_ranks
        
    def numel(self):
        # Calculate the total number of parameters with gradients
        return sum(model.numel() for model in self.models)
        
    
    def forward(self, x, x_extra=None, covariates=None, return_utilities=False, inf_weight=float('-inf')):
        """
        Forward propagation, computing y_hat
        
        Inputs: 
        x - (batch_size, maximum sequence length, 2) array of item indices involved in the choice set or chosen set of interest.
        x_extra - (batch_size, 3) array, columns representing chooser, length of choice set, and length of chosen set for each observation in x.
        covariates - (num_agents, num_items, num_features)-sized numpy array of covariates
        inf_weight - used to "zero out" padding terms. Should not be changed.
        """   
        batch_size, seq_len, _ = x.size()
        return_object = t.zeros((batch_size, seq_len))
        
        if covariates is None:
            covariates=self.covariates
        else:
            num_agents, num_alternatives, num_features = covariates.shape
            pad_mat = np.zeros((num_agents, num_features), dtype=np.float32)
            covariates = np.hstack([covariates, pad_mat[:,None,:]]) # add zeros matrix to second dimension of covariates (column-wise)
            covariates = t.from_numpy(covariates)
        
        for i in range(self.k-1):
            # Find observations that correspond to i-th choice (x_extra[:,2] stores size of context set).
            rows = x_extra[:,2]==i # (batch_size, )
            
            # Collect corresponding outputs from submodels
            return_object[rows] = self.models[i](x[rows,...], x_extra[rows,...], covariates, return_utilities=return_utilities)
        rows = x_extra[:,2]>=(self.k-1)
        return_object[rows] = self.models[-1](x[rows,...], x_extra[rows,...], covariates, return_utilities=return_utilities)
        
        return return_object

    def reg(self):
        '''
        Computes regularization term for loss

        self.school_logits & self.program_logits are ModuleLists of Embeddings
        self.beta is a ModuleList of t.nn.Linear()
        self.targets & self.context are ModuleLists of Embeddings
        self.order_reg is the order of regularization (either L1 or L2 in this case)
        self.k is the number of stratified models learned (ie. length of self.beta)
        '''
        if not self.fixed_effects:
            fe_reg = t.zeros(1)
        elif self.school_choice:
            fe_reg = t.sum(t.stack([t.linalg.norm(t.sub(self.models[i].school_logits.weight, self.models[i-1].school_logits.weight), ord=self.order_reg)**2 for i in range(1,self.k)])) + t.sum(t.stack([t.linalg.norm(t.sub(self.models[i].program_logits.weight, self.models[i-1].program_logits.weight), ord=self.order_reg)**2 for i in range(1,self.k)]))
        else:
            fe_reg = t.sum(t.stack([t.linalg.norm(t.sub(self.models[i].logits.weight, self.models[i-1].logits.weight), ord=self.order_reg)**2 for i in range(1,self.k)]))
            
        linear_reg = t.zeros(1) if not self.linear_terms else t.sum(t.stack([t.linalg.norm(t.sub(self.models[i].beta.weight, self.models[i-1].beta.weight), ord=self.order_reg)**2 for i in range(1,self.k)]))

        context_reg = t.zeros(1) if ((not self.context) | (self.k<=2)) else t.sum(t.stack([t.linalg.norm(t.sub(self.models[i].target_embeddings.weight, self.models[i-1].target_embeddings.weight), ord=self.order_reg)**2 for i in range(2,self.k)])) + t.sum(t.stack([t.linalg.norm(t.sub(self.models[i].context_embeddings.weight, self.models[i-1].context_embeddings.weight), ord=self.order_reg)**2 for i in range(2,self.k)]))
        
        return fe_reg + linear_reg + context_reg
    
    def loss_func(self, y_hat, y, x=None, x_extra=None, train=True):
        """
        Evaluates the model
        Inputs: 
        y_hat - the log softmax values that come from the forward function
        y - actual labels - the choice in a set (i-th entry must be less than x_extra[i,1])
        x_extra - observation metadata (same as in forward function)
        train - bool. if True, returns full training loss (incl. regularization loss)
        """
        terms = []
        for i in range(self.num_ranks):
            rows=x_extra[:,2]==i
            terms.append(F.nll_loss(y_hat[rows], y[rows].long()))
        tl = F.nll_loss(y_hat, y.long())
        if (self.k>1) & train:
            rl = self.lambda_reg*self.reg()
            terms.append(rl)
            return (tl + rl, terms)
        else:
            terms.append(t.tensor(0.))
            return (tl, terms)

    def acc_func(self, y_hat, y, x_lengths=None):
        return (y_hat.argmax(1).int() == y.int()).float().mean()
    
    def sample_preferences(self, num_samples=1, covariates=None):
        assert not self.context, 'Function does not support context effects yet.'
        if self.linear_terms & (covariates is None):
            covariates = self.covariates
            num_agents, num_alternatives, num_features = covariates.shape
        elif self.linear_terms & (covariates is not None):
            num_agents, num_alternatives, num_features = covariates.shape
        else:
            num_agents, num_alternatives = num_samples, self.num_items
            covariates=self.covariates

        preferences = t.zeros((num_agents, num_alternatives), dtype=t.long)
        
        for i in range(self.k-1):
            utilities = self.models[i].compute_utilities(covariates)
            if utilities.shape[0]==1:
                utilities.tile((num_agents,1))
            utilities[t.arange(num_agents)[:,None], preferences[:,:i]]=-np.inf
            softmax_probs = softmax(utilities, axis=1)
            preferences[:,i] = t.multinomial(softmax_probs, num_samples=1, axis=1).flatten()
        utilities = self.models[-1].compute_utilities(covariates)
        if utilities.shape[0]==1:
            utilities = utilities.tile((num_agents,1))
        utilities[t.arange(num_agents)[:,None], preferences[:,:(self.k-1)]] = -np.inf
        softmax_probs = softmax(utilities, axis=1)
        preferences[:,(self.k-1):] = t.multinomial(softmax_probs, num_samples=num_alternatives-(self.k-1),replacement=False)

        return preferences

class MNL(nn.Module):
    """
    The MNL model.
    """
    def __init__(self, num_items, alternative_codex, ism=True, 
                 school_choice=False, fixed_effects=False, 
                 item_to_school=None, item_to_program=None,
                 school_names=None, program_type_names=None,
                 linear_terms=False, covariates=None, 
                 context=False, forward_dependent=False, embedding_dim=5, 
                 top_k=None, num_ranks=5):
        """
        Initializes the base MNL model. 
        Utility can include fixed-effects, linear terms, and/or context-effects.
        
        Inputs: 
        num_items - int, total number of items in the choice system modeled
        ism - bool, True if dataset is multi-set, in which case padding is used
        fixed_effects - bool, if True, item-level fixed effects are used in utility
        item_to_school - array mapping offerings to their respective school index (only needed if fixed_effects is True)
        item_to_program - array mapping offerings to their respective program type index (only needed if fixed_effects is True)
        linear_terms = bool, if True, linear terms are used in utility
        covariates - (n_obs, num_items, num_features) array of covariates
        context - bool, if True, context-effects are used in utility (ie. model is a CDM)
        forward_dependent - bool, if True, forward-dependent CDM is used, else backward-dependent (unused in paper)
        embedding_dim - int > 0, embedding dimension of the low-rank CDM.
        top_k - int or None, rank of the last considered context effect in the top-k CDM (only relevant if forward_dependent is False)
        num_ranks - int > 0, number of rank positions to return loss.
        """
        super().__init__()
        self.num_items = num_items
        self.alternative_codex = alternative_codex
        self.ism = ism
        
        self.fixed_effects = fixed_effects
        self.school_choice = school_choice
        self.school_names = school_names
        self.program_type_names = program_type_names
        if self.fixed_effects:
            if school_choice:
                self.num_schools=np.unique(item_to_school).size
                self.num_program_types=np.unique(item_to_program).size
                self.item_to_school = t.from_numpy(np.append(item_to_school, self.num_schools)).long()
                self.item_to_program_type = t.from_numpy(np.append(item_to_program, self.num_program_types)).long()
            else:
                self.item_to_school = item_to_school
                self.item_to_program_type = item_to_program
        
        self.linear_terms = linear_terms
        # try:
        num_agents, _, self.num_features = covariates.shape if linear_terms else (0,0,0)
        # except:
        #     import pdb; pdb.set_trace()
        if linear_terms:
            pad_vec = np.zeros([num_agents, self.num_features], dtype=np.float32)
            covariates = np.hstack([covariates, pad_vec[:,None,:]]) # add zeros matrix to second dimension of covariates (column-wise)
            self.covariates =  t.from_numpy(covariates) # num_items+1 x num_features tensor
        else:
            self.covariates = covariates
        
        self.context = context
        self.forward_dependent = forward_dependent
        if forward_dependent & (top_k is not None):
            raise Exception("Forward + top_k CDMs are not compatible.")
        self.top_k = num_items if top_k is None else top_k
        self.embedding_dim = embedding_dim

        self.num_ranks = num_ranks
        self.__build_model()

    def __build_model(self):
        """
        Helper function to initialize the model
        """            
        if self.fixed_effects:
            if self.school_choice:
                self.school_logits = Embedding(num_embeddings=self.num_schools + 1,  # +1 for the padding
                                          embedding_dim=1,
                                          padding_idx=self.num_schools) # requires_grad=True
                self.program_logits = Embedding(num_embeddings=self.num_program_types + 1,  # +1 for the padding
                                           embedding_dim=1,
                                           padding_idx=self.num_program_types) # requires_grad=True
            else:
                self.logits = Embedding(num_embeddings=self.num_items + 1,  # +1 for the padding
                                               embedding_dim=1,
                                               padding_idx=self.num_items) # requires_grad=True

        if self.linear_terms:
            self.beta = t.nn.Linear(self.num_features, 1, bias=False)
        
        if self.context:
            self.target_embeddings = Embedding(num_embeddings=self.num_items+1,
                                               embedding_dim=self.embedding_dim,
                                               padding_idx=self.num_items,
                                               _weight=t.zeros([self.num_items+1, self.embedding_dim]))
            self.context_embeddings = Embedding(num_embeddings=self.num_items+1,
                                                embedding_dim=self.embedding_dim,
                                                padding_idx=self.num_items)
            
    def numel(self):
        # Calculate the total number of parameters with gradients
        total_parameters=0
        if self.school_choice & self.fixed_effects:
            total_parameters += self.school_logits.numel() + self.program_logits.numel()
        elif (not self.school_choice) & self.fixed_effects:
            total_parameters += self.logits.numel()
        else:
            pass
        
        if self.linear_terms:
            total_parameters += self.beta.weight.numel()
            
        if self.context:
            total_parameters += self.target_embeddings.numel() + self.context_embeddings.numel()
        return total_parameters

    def forward(self, x, x_extra=None, covariates=None, return_utilities=False, inf_weight=float('-inf')):
        """
        Forward propagation, computing y_hat
        
        Inputs: 
        x - (batch_size, maximum sequence length, 2) array of item indices involved in the choice set or chosen set of interest.
        x_extra - (batch_size, 3) array, columns representing chooser, length of choice set, and length of chosen set for each observation in x.
        covariates - (num_agents, num_items, num_features)-sized numpy array of covariates
        inf_weight - used to "zero out" padding terms. Should not be changed.
        """    
        batch_size, seq_len, _ = x.size()
        
        utilities = t.zeros((batch_size, seq_len))
        if self.fixed_effects:
            logits = t.zeros((batch_size, seq_len)) # Initialize empty array to populate
            if self.school_choice:
                to_program_types = self.item_to_program_type[x[:,:,0]] # (num_rows, seq_len, 1)
                program_logits = self.program_logits(to_program_types).squeeze() # (num_rows, seq_len)
                to_schools = self.item_to_school[x[:,:,0]] # (num_rows, seq_len, 1)
                school_logits = self.school_logits(to_schools).squeeze()  # (num_rows, seq_len)
                logits = school_logits + program_logits  # (num_rows, seq_len)
            else: 
                logits = self.logits(x[:,:,0]).squeeze()
                
            utilities += logits # (batch_size, seq_len)
            pass
        
        if self.linear_terms:
            if covariates is None:
                covariates=self.covariates
            else:
                num_agents, num_alternatives, num_features = covariates.shape
                pad_mat = np.zeros((num_agents, num_features), dtype=np.float32)
                covariates = np.hstack([covariates, pad_mat[:,None,:]]) # add zeros matrix to second dimension of covariates (column-wise)
                covariates = t.from_numpy(covariates)
            cov = self.covariates[x_extra[:,0,None], x[:,:,0]]
            linear = self.beta(cov).squeeze()
            utilities += linear
            pass
        
        if self.context & self.forward_dependent:
            rows = x_extra[:,1]>=2
            context_vecs = self.context_embeddings(x[rows,:,0])
            context_vecs = context_vecs.sum(-2, keepdim=True) - context_vecs
            utilities[rows] += (self.target_embeddings(x[rows,:,0]) * context_vecs).sum(-1).div((x_extra[rows,1]-1.)[:,None]) 
        elif self.context & ~self.forward_dependent:
            rows = x_extra[:,2]>=1
            context_vecs = self.context_embeddings(x[rows,:self.top_k,1])
            context_vecs = context_vecs.sum(-2, keepdim=True) 
            utilities[rows] += (self.target_embeddings(x[rows,:,0]) * context_vecs).sum(-1).div(x_extra[rows,2][:,None])
        else:
            pass

        if self.ism:
            utilities[t.arange(seq_len)[None, :] >= x_extra[:, 1, None]] = inf_weight
        if return_utilities:
            return utilities
        else:
            return F.log_softmax(utilities, 1).squeeze()
    
    def loss_func(self, y_hat, y, x=None, x_extra=None):
        """
        Evaluates the model
        Inputs: 
        y_hat - the log softmax values that come from the forward function
        y - actual labels - the choice in a set (i-th entry must be less than x_extra[i,1])
        x_extra - observation metadata (same as in forward function)
        """        
        terms = []
        for i in range(self.num_ranks):
            rows=x_extra[:,2]==i
            terms.append(F.nll_loss(y_hat[rows], y[rows].long()))
        terms.append(t.tensor(0.))
        loss = F.nll_loss(y_hat, y.long())
        return (loss, terms)

    def acc_func(self, y_hat, y, x_lengths=None):
        return (y_hat.argmax(1).int() == y.int()).float().mean()
    
    def compute_utilities(self, covariates=None):
        assert not self.context, 'Function does not compute context effects yet.'
        if self.linear_terms:
            if covariates is None:
                covariates=self.covariates[:,:-1,:]
                num_agents, num_alternatives, num_features = covariates.shape
            else:
                num_agents, num_alternatives, num_features = covariates.shape
                # pad_mat = np.zeros((num_agents, num_features), dtype=np.float32)
                # covariates = np.hstack([covariates, pad_mat[:,None,:]]) # add zeros matrix to second dimension of covariates (column-wise)
                covariates = t.from_numpy(covariates)
        else:
            num_agents, num_alternatives = 1, self.num_items
            covariates=self.covariates

        utilities = t.zeros((num_agents, num_alternatives))
        if self.fixed_effects:
            if self.school_choice:
                school_logits = self.school_logits.weight[self.item_to_school].detach().numpy()[:-1].flatten()
                program_type_logits = self.program_logits.weight[self.item_to_program_type].detach().numpy()[:-1].flatten()
                utilities += school_logits[None,:]+program_type_logits[None,:]
            else:
                logits = self.logits.weight.detach().numpy()[:-1].flatten()
                utilities += logits[None,:]
        if self.linear_terms:
            linear = covariates@self.beta.weight.detach().numpy().squeeze()
            utilities += linear

        return utilities
    
def get_data(train_ds, val_ds, batch_size=None):
    # Note: can change val_bs to 2* batch_size if ever becomes a problem
    if batch_size is not None:
        tr_bs, val_bs = (batch_size, len(val_ds[0]))
    else: 
        tr_bs, val_bs = (len(train_ds[0]), len(val_ds[0]))

    train_dl = DataLoader(train_ds, batch_size=tr_bs, shuffle=batch_size is not None)
    val_dl = DataLoader(val_ds, batch_size=val_bs)
    return train_dl, val_dl

def loss_batch(model, xb, yb, xlb, opt=None, retain_graph=None):
    if opt is not None:
        loss, terms = model.loss_func(model(xb, xlb), yb, xb, xlb)

        opt.zero_grad()
        loss.backward(retain_graph=retain_graph)
        opt.step()
    else:
        with t.no_grad():
            loss, terms = model.loss_func(model(xb, xlb), yb, xb, xlb)

    return loss, terms

def acc_batch(model, xb, yb, xlb):
    with t.no_grad():
        return model.acc_func(model(xb, xlb), yb, xlb)

def fit(epochs, model, opt, train_dl, verbose=True, epsilon=1e-4, val_dl=None):
    val_loss = t.zeros(1)
    losses=[]
    loss1=-1.0
    loss2=0.0
    epoch=0
    while (np.abs(loss2-loss1)>epsilon) & (epoch<epochs):
        loss1=loss2
        epoch+=1
        model.train()  # good practice because these are used by nn.BatchNorm2d and nn.Dropout
        for xb, xlb, yb in train_dl:
            loss, terms = loss_batch(model, xb, yb, xlb, opt, retain_graph=None if epoch != epochs - 1 else True)
        loss2 = float(loss.detach().numpy())
        losses.append([float(t.detach().numpy()) for t in terms])
        if val_dl is not None:
            model.eval() # good practice like model.train()
            val_loss = [loss_batch(model, xb, yb, xlb) for xb, xlb, yb in val_dl]
            val_loss = sum(val_loss)/len(val_loss)
            val_acc = [acc_batch(model, xb, yb, xlb) for xb, xlb, yb in val_dl]
            val_acc = sum(val_acc) / len(val_acc)
            if (epoch%25==0) & verbose:
                print(f'Epoch: {epoch}, Training Loss: {loss2}, Val Loss: {val_loss}, \
                    Val Accuracy {val_acc}')
        else:
            if (epoch%25==0) & verbose:
                print(f'Epoch: {epoch}, Training Loss: {loss2}')

    return loss2, epoch, losses, val_loss.numpy()

def train(ds, num_items, alternative_codex, ism=True, batch_size=None, epochs=500, lr=1e-3, seed=2, wd=1e-4, Model=RankStratifiedMNL, val_ds=None, verbose=True, **kwargs):
    t.autograd.set_detect_anomaly(True)
    tr_bs = batch_size if batch_size is not None else 1000
    if val_ds is not None:
        train_dl, val_dl = get_data(ds, val_ds, batch_size=batch_size)
    else:
        train_dl = DataLoader(ds, batch_size=tr_bs, shuffle=batch_size is not None)
        val_dl = None
    if seed is not None:
        t.manual_seed(seed)
    model = Model(num_items, alternative_codex, ism, **kwargs)
    no_params = model.numel()
    print('No. params: ', no_params)
    opt = t.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)

    s = time.time()
    tr_loss, num_epochs, losses, val_loss = fit(epochs, model, opt, train_dl, verbose=verbose, val_dl=val_dl)
    runtime = time.time() - s
    if verbose:
        print(f'Runtime: {runtime}')
        print(f'Loss: {tr_loss}')
        
    return model, tr_loss, no_params, num_epochs, runtime, losses, val_loss
