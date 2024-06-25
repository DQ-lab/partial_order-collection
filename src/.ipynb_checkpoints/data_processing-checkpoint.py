# This file contains some utils to help load and play around with choice data
import copy
import pandas as pd
import numpy as np
import torch as t

def sfrcv_clean_dataframe(df, top_k=None):
    data_list = []
    for i, row in df.iterrows():
        voter = row['pref_voter_id']
        first_round = str(row['first_report_seen'])
        candidates = row['candidate']
        ranks = row['vote_rank']
        year = row['year']
        race = row['race']
        num_ranked = len(eval(candidates))
        if top_k and num_ranked>top_k:
            candidates = candidates[:top_k]
            ranks=ranks[:top_k]
            num_ranked=top_k
        data_list.append([voter, eval(candidates), num_ranked, eval(ranks), year, race])
    dataset = pd.DataFrame(data_list, columns = ['agent', 'alternatives', 'num_ranked', 'ranks', 'year', 'race'])
    return dataset

def remove_repeats(entry):
    # if someone voted for the" same candidate twice, it will remove the repeats
    # while still preserving the order of the original list
    if len(list(np.array(entry)[np.sort(np.unique(entry, return_index=True)[1])])) == 0:
        print(entry)
    return list(np.array(entry)[np.sort(np.unique(entry, return_index=True)[1])])

def final_choice(entry):
    return [entry[-1]]

def prep_for_pytorch(ballot, n):
    voters, choices, choice_sets, context_sets, remaining_sets, remaining_set_lengths, rankings, list_lengths = ballot['agent_id'].values, ballot['choices'].values, ballot['choice_sets'].values, ballot['context_sets'].values, ballot['remaining_set'].values, ballot['remaining_set_length'].values, ballot['alternative_id'].values, ballot['length'].values
    choice_sets_concat, choices_concat, context_sets_concat, whose_choice_concat, remaining_sets_concat, rs_lengths_concat, rankings_concat, lengths_concat = [], [], [], [], [], [], [], []
    for idx, choice_set in enumerate(choice_sets):
        assert(len(choice_set)==len(context_sets[idx])==len(choices[idx]))
        choice_sets_concat += choice_set
        context_sets_concat += context_sets[idx]
        choices_concat += choices[idx]
        whose_choice_concat += [voters[idx]]*len(choices[idx])
        remaining_sets_concat += [remaining_sets[idx]]*len(choices[idx])
        rankings_concat += [rankings[idx]]*len(choices[idx])
        rs_lengths_concat += [remaining_set_lengths[idx]]*len(choices[idx])
        lengths_concat += [list_lengths[idx]]*len(choices[idx])
    
    choices = choices_concat
    choice_set_lengths = np.array([len(choice_set) for choice_set in choice_sets_concat])
    context_sets_lengths = np.array([len(context_set) for context_set in context_sets_concat])
    x_extra = np.stack([np.array(whose_choice_concat), choice_set_lengths, context_sets_lengths, np.array(rs_lengths_concat), np.array(lengths_concat)], axis=1)
    slots_chosen = np.array([choice_set.index(choices[idx]) for idx, choice_set in enumerate(choice_sets_concat)])
    
    kmax = choice_set_lengths.max() #should always be size of universe/greater than max chosen set length
    padded_choice_sets = np.full([len(choice_sets_concat), kmax], fill_value=n, dtype=np.compat.long)
    choice_sets = np.concatenate(choice_sets_concat)
    padded_choice_sets[np.arange(kmax)[None, :] < choice_set_lengths[:, None]] = choice_sets

    padded_context_sets = np.full([len(context_sets_concat), kmax], fill_value=n, dtype=np.compat.long)
    context_sets = np.concatenate(context_sets_concat)
    padded_context_sets[np.arange(kmax)[None, :] < context_sets_lengths[:, None]] = context_sets

    padded_remaining_sets = np.full([len(remaining_sets_concat), kmax], fill_value=n, dtype=np.compat.long)
    remaining_sets = np.concatenate(remaining_sets_concat)
    padded_remaining_sets[np.arange(kmax)[None, :] < np.array(rs_lengths_concat)[:, None]] = remaining_sets

    padded_rankings = np.full([len(rankings_concat), kmax], fill_value=n, dtype=np.compat.long)
    rankings = np.concatenate(rankings_concat)
    try:
        padded_rankings[np.arange(kmax)[None, :] < np.array(lengths_concat)[:, None]] = rankings
    except:
        import pdb; pdb.set_trace()

    x = np.stack([padded_choice_sets, padded_context_sets, padded_remaining_sets, padded_rankings], axis=-1)

    return list(map(t.from_numpy, [x, x_extra, slots_chosen]))

def prep_valset(data, alternatives_codex, end_token=False):
    codex_alternatives = copy.deepcopy(alternatives_codex)
    n=len(codex_alternatives) if not end_token else len(codex_alternatives)-1
    
    all_restricted = []
    data['alternative_id'] = [[codex_alternatives.index(alt) for alt in alt_list] 
                                  if all([(alt in codex_alternatives) for alt in alt_list]) else None
                                  for alt_list in data.alternatives]
    
    data = data.dropna(subset=['alternative_id'])
    
    cat=pd.Categorical(data['agent'])
    data = data.assign(agent_id=cat.codes)
    codex_agents = list(cat.categories)

    def choice_set_func(entry, univ):
        for item in entry:
            if item not in univ:
                univ.append(item)
        choice_sets = []
        choice_sets.append(list(univ))
        for idx, item in enumerate(entry[1:]):
            univ.remove(entry[idx])
            choice_sets.append(list(univ))
        return choice_sets
    
    def chosen_set_func(entry):
        chosen_sets = [[]]
        for idx, item in enumerate(entry[1:]):
            chosen_set = chosen_sets[-1].copy()
            chosen_set.extend([entry[idx]])
            chosen_sets.append(chosen_set)
        return chosen_sets

    ballots = []
    for year in data.year.unique():
        year_data = data[data["year"] == year]
        subset=list(np.arange(len(codex_alternatives)))

        ballot_sort = []
        for ind, item in enumerate(year_data[['alternative_id', 'ranks']].values):
            remaining_set = subset.copy()
            sort_idx = np.argsort(item[1])
            for alt in item[0]:
                if alt in remaining_set:
                    remaining_set.remove(alt)
            if end_token:
                ballot_sort.append([year_data.iloc[ind]['agent_id'], 
                                    year_data.iloc[ind]['year'], 
                                    list(np.array(item[0])[sort_idx])+[n+1], 
                                    list(np.array(item[1])[sort_idx])+[len(item[0])+1], 
                                    remaining_set, 
                                    len(remaining_set), 
                                    len(item[0])])
            else:
                ballot_sort.append([year_data.iloc[ind]['agent_id'], 
                                    year_data.iloc[ind]['year'], 
                                    list(np.array(item[0])[sort_idx]), 
                                    list(np.array(item[1])[sort_idx]), 
                                    remaining_set,
                                    len(remaining_set), 
                                    len(item[0])])
        ballot = pd.DataFrame(ballot_sort, columns = ['agent_id', 
                                                      'year',
                                                      'alternative_id', 
                                                      'ranks',
                                                      'remaining_set',
                                                      'remaining_set_length',
                                                      'length'])
        indices_to_remove = np.array([~np.all(np.array(item) == (np.arange(len(item))+1)) for item in ballot['ranks'].values])
        ballot = ballot[~indices_to_remove]

        ballot['choices'] = ballot['alternative_id'].apply(remove_repeats)
        ballot['choice_sets'] = ballot.apply(lambda x: choice_set_func(x['choices'], list(subset)), axis=1)
        ballot['context_sets'] = ballot['choices'].apply(lambda x: chosen_set_func(x))
        ballots.append(ballot)
    ballot = pd.concat(ballots, ignore_index=True)
    ds = prep_for_pytorch(ballot, n)
    return ds, codex_agents

def prep_dataset(data, end_token=False):
    cat = pd.Categorical(data.alternatives.explode())
    codex_alternatives = list(cat.categories)
        
    data['alternative_id'] = [[codex_alternatives.index(alternative) for alternative in alternative_list] for alternative_list in data.alternatives]
    n=len(codex_alternatives)
        
    cat=pd.Categorical(data['agent'])
    data['agent_id'] = cat.codes
    codex_agents = list(cat.categories)
    
    if end_token:
        codex_alternatives.append('END')            
    
    def choice_set_func(entry, univ):
        for item in entry:
            if item not in univ:
                univ.append(item)
        choice_sets = []
        choice_sets.append(list(univ))
        for idx, item in enumerate(entry[1:]):
            univ.remove(entry[idx])
            choice_sets.append(list(univ))
        return choice_sets
    
    def chosen_set_func(entry):
        chosen_sets = [[]]
        for idx, item in enumerate(entry[1:]):
            chosen_set = chosen_sets[-1].copy()
            chosen_set.extend([entry[idx]])
            chosen_sets.append(chosen_set)
        return chosen_sets
    
    ballots = []
    for year in data.year.unique():
        year_data = data.query('year==@year')
        subset = list(year_data.alternative_id.explode().unique())
        ballot_sort = []
        for ind, item in enumerate(year_data[['alternative_id', 'ranks']].values):
            remaining_set = subset.copy()
            sort_idx = np.argsort(item[1])
            for alt in item[0]:
                if alt in remaining_set:
                    remaining_set.remove(alt)
            if end_token:
                ballot_sort.append([year_data.iloc[ind]['agent_id'], 
                                    year_data.iloc[ind]['year'], 
                                    list(np.array(item[0])[sort_idx])+[n+1], 
                                    list(np.array(item[1])[sort_idx])+[len(item[0])+1], 
                                    remaining_set, 
                                    len(remaining_set), 
                                    len(item[0])])
            else:
                ballot_sort.append([year_data.iloc[ind]['agent_id'], 
                                    year_data.iloc[ind]['year'], 
                                    list(np.array(item[0])[sort_idx]), 
                                    list(np.array(item[1])[sort_idx]),
                                    remaining_set, 
                                    len(remaining_set), 
                                    len(item[0])])
        ballot = pd.DataFrame(ballot_sort, columns = ['agent_id', 
                                                      'year',
                                                      'alternative_id', 
                                                      'ranks',
                                                      'remaining_set',
                                                      'remaining_set_length',
                                                      'length'])
        indices_to_remove = np.array([~np.all(np.array(item) == (np.arange(len(item))+1)) for item in ballot['ranks'].values])
        ballot = ballot[~indices_to_remove]

        ballot['choices'] = ballot['alternative_id'].apply(remove_repeats)
        ballot['choice_sets'] = ballot.apply(lambda x: choice_set_func(x['choices'], list(subset)), axis=1)
        ballot['context_sets'] = ballot['choices'].apply(lambda x: chosen_set_func(x))
        ballots.append(ballot)
    ballot = pd.concat(ballots, ignore_index=True)
    ds = prep_for_pytorch(ballot, n)

    return ds, codex_agents, codex_alternatives, ballot