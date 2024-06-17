import datetime

import pandas as pd
import os

import yaml

with open("config_files/basic_configuration_drop_pop.yml", "r") as basic_file:
    basic_config = yaml.load(basic_file)

from elliot.run import run_experiment

result_elliot = []
print("Done! We are now starting the Elliot's experiment")


def rename_model_names(long_name):
    return long_name.replace(
        "UserKNN_nn=40_sim=cosine_imp=aiolli_bin=False_shrink=0_norm=True_asymalpha=_tvalpha=_tvbeta=_rweights=",
        "User-KNN").replace(
        "ItemKNN_nn=40_sim=cosine_imp=aiolli_bin=False_shrink=0_norm=True_asymalpha=_tvalpha=_tvbeta=_rweights=",
        "Item-KNN").replace(
        "UserKNN_nn=40_sim=cosine_imp=classical_bin=False_shrink=0_norm=True_asymalpha=_tvalpha=_tvbeta=_rweights=",
        "User-KNN").replace(
        "ItemKNN_nn=40_sim=cosine_imp=classical_bin=False_shrink=0_norm=True_asymalpha=_tvalpha=_tvbeta=_rweights=",
        "Item-KNN").replace('PureSVD_factors=10', 'SVD').replace(
        "BPRMF_seed=42_e=10_bs=1_f=10_lr=0$001_bias_reg=0_u_reg=0$0025_pos_i_reg=0$0025_neg_i_reg=0$0025_up_neg_i_f=True_up_u=True_up_i=True_up_b=True",
        'BPRMF').replace("Random_seed=42", "Random")


rec_scores = []
for activity in ['physical', 'nature', 'cultural', 'social', 'environmental']:
    for drop_pop in [0.5, 1, 1.5, 2, 2.5]:
        for dataset in ['prolific', "flickr"]:
            train_path = f'data/drop_pop/splitting/train_{dataset}_{activity}.tsv'
            test_path = f'data/drop_pop/splitting/test_{dataset}_{activity}.tsv'
            interactions_category = pd.read_csv(f"data/greens/interactions_{dataset}_{activity}.tsv", sep='\t',
                                                header=None, names=['user', 'park', 'rating', 'timestamp'])
            max_popularity = interactions_category.groupby('park')['user'].count().sort_values().quantile(0.01 * (100 - drop_pop))
            to_keep = interactions_category.groupby('park')['user'].count() <= max_popularity
            interactions_category = interactions_category[interactions_category['park'].isin(to_keep[to_keep == True].index)]

            train_aware = interactions_category.groupby('user').apply(lambda visits: (visits.sample(n=len(visits) - 1)))
            train_aware.to_csv(train_path, sep='\t', index=False, header=False)
            test_aware = pd.concat([interactions_category, train_aware]).drop_duplicates(['user', 'park'], keep=False)
            test_aware.to_csv(test_path, sep='\t', index=False, header=False)

            print(dataset, activity, drop_pop, max_popularity, len(interactions_category.index), len(train_aware.index),
                  len(test_aware.index))

            basic_config['experiment']['data_config']['train_path'] = f"../{train_path}"
            basic_config['experiment']['data_config']['test_path'] = f"../{test_path}"

            with open(f"config_files/basic_configuration_drop_pop_{activity}_{dataset}.yml", "w") as config_file:
                yaml.dump(basic_config, config_file)
            run_experiment(f"config_files/basic_configuration_drop_pop_{activity}_{dataset}.yml")
            for model in ['MostPop.tsv', 'PureSVD_factors=10.tsv', 'Random_seed=42.tsv',
                          'UserKNN_nn=40_sim=cosine_imp=classical_bin=False_shrink=0_norm=True_asymalpha=_tvalpha=_tvbeta=_rweights=.tsv',
                          'ItemKNN_nn=40_sim=cosine_imp=classical_bin=False_shrink=0_norm=True_asymalpha=_tvalpha=_tvbeta=_rweights=.tsv',
                          'BPRMF_seed=42_e=10_bs=1_f=10_lr=0$001_bias_reg=0_u_reg=0$0025_pos_i_reg=0$0025_neg_i_reg=0$0025_up_neg_i_f=True_up_u=True_up_i=True_up_b=True_it=10.tsv']:
                scores = pd.read_csv(f"results/drop_pop/recs/{model}", sep='\t', header=None,
                                     names=['user', 'park', 'score'])
                scores['model'] = rename_model_names(model[:-4])
                scores['activity'] = activity
                scores['dataset'] = dataset
                rec_scores.append(scores)

            for cutoff in [5, 10, 20]:

                latest_result = pd.read_csv(
                    f'results/drop_pop/performance/{max(filter(lambda f: f.startswith(f"rec_cutoff_{cutoff}"), os.listdir("results/drop_pop/performance")), key=lambda d: datetime.datetime.strptime(d, f"rec_cutoff_{cutoff}_relthreshold_1_%Y_%m_%d_%H_%M_%S.tsv"))}',
                    sep='\t')
                for _, r in latest_result.iterrows():
                    intermediate_result = {"activity": activity, 'model': rename_model_names(r['model']),
                                           'variant': dataset, 'cutoff': cutoff,
                                           "nDCG": r['nDCG'], 'ARP': r['ARP'], 'drop_pop': drop_pop,
                                           'max_visits_per_parks': max_popularity}
                    result_elliot.append(intermediate_result)

result_df = pd.DataFrame(result_elliot)
result_df.to_csv('results-drop_pop.csv', index=False)
scores_df = pd.concat(rec_scores)
scores_df.to_csv("rec_scores_aware.csv", index=False)

fli_sizes = scores_df[scores_df['dataset'] == 'flickr'].groupby(['user', 'park']).size()
fli_sizes = fli_sizes.rename("count")
fli_input = pd.merge(scores_df[scores_df['dataset'] == 'flickr'], fli_sizes[fli_sizes == 30], left_on=['user', 'park'],
                     right_index=True)
fli_input.to_csv("flickr_CA-scores.csv", index=False)

prolific_sizes = scores_df[scores_df['dataset'] == 'prolific'].groupby(['user', 'park']).size()
prolific_sizes = prolific_sizes.rename("count")
prolific_input = pd.merge(scores_df[scores_df['dataset'] == 'prolific'], prolific_sizes[prolific_sizes == 30],
                          left_on=['user', 'park'], right_index=True)
prolific_input.to_csv("prolific_CA-scores.csv", index=False)
