import datetime

import pandas as pd
import os

import yaml

with open("config_files/basic_configuration_full_recommendations.yml", "r") as basic_file:
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


#for activity in [ 'environmental']:
rec_scores = []
for activity in ['physical', 'nature', 'cultural', 'social', 'environmental']:
    for dtv in [0.5]:
    #for dataset in ['flickr']:
        for dataset in ['prolific','flickr']:
            all_interactions = pd.read_csv(f"data/greens/interactions_{dataset}.tsv", sep='\t', header=None, names=['user', 'park', 'rating', 'timestamp'])
            all_interactions = all_interactions.drop_duplicates(['user', 'park'], keep="first")
            max_popularity = all_interactions.groupby('park')['user'].count().sort_values().quantile(0.01 * (100 - dtv))
            to_keep = all_interactions.groupby('park')['user'].count() <= max_popularity
            all_interactions = all_interactions[all_interactions['park'].isin(to_keep[to_keep == True].index)]

            train_path = f'data/full_recommendations/splitting/train_{dataset}_{activity}.tsv'
            test_path = f'data/full_recommendations/splitting/test_{dataset}_{activity}.tsv'
            interactions_category = pd.read_csv(f"data/greens/interactions_{dataset}_{activity}.tsv", sep='\t', header=None, names=['user', 'park', 'rating', 'timestamp'])
            #print(activity, dataset, len(interactions_category.index), len(interactions_category.drop_duplicates(['user', 'park']).index))
            interactions_category = interactions_category.drop_duplicates(['user', 'park'], keep="first")

            train_aware = interactions_category.groupby('user').apply(lambda visits: (visits.sample(n=len(visits) - 1)))
            test_aware = pd.concat([interactions_category, train_aware]).drop_duplicates(['user', 'park'], keep=False)
            test_aware = test_aware[test_aware['park'].isin(to_keep[to_keep == True].index)]

            test_aware.to_csv(test_path, sep='\t', index=False, header=False)
            train_unaware = pd.concat([all_interactions, test_aware]).drop_duplicates(['user', 'park'], keep=False)
            train_unaware.to_csv(train_path, sep='\t', index=False, header=False)
            #print("train-test-merge",activity, dataset, len(interactions_category), len(pd.concat([train_aware,test_aware]).index))

            basic_config['experiment']['data_config']['train_path'] = f"../{train_path}"
            basic_config['experiment']['data_config']['test_path'] = f"../{test_path}"
            basic_config['experiment']['top_k'] = 10**7

            with open(f"config_files/basic_configuration_full_recommendations_{activity}_{dataset}.yml", "w") as config_file:
                yaml.dump(basic_config, config_file)
            run_experiment(f"config_files/basic_configuration_full_recommendations_{activity}_{dataset}.yml")
            for model in ['MostPop.tsv','PureSVD_factors=10.tsv','Random_seed=42.tsv','UserKNN_nn=40_sim=cosine_imp=classical_bin=False_shrink=0_norm=True_asymalpha=_tvalpha=_tvbeta=_rweights=.tsv',
                          'ItemKNN_nn=40_sim=cosine_imp=classical_bin=False_shrink=0_norm=True_asymalpha=_tvalpha=_tvbeta=_rweights=.tsv',
                'BPRMF_seed=42_e=10_bs=1_f=10_lr=0$001_bias_reg=0_u_reg=0$0025_pos_i_reg=0$0025_neg_i_reg=0$0025_up_neg_i_f=True_up_u=True_up_i=True_up_b=True_it=10.tsv']:
                scores = pd.read_csv(f"results/full_recommendations/recs/{model}", sep='\t', header=None, names=['user', 'park', 'score'])
                scores['model'] = rename_model_names(model[:-4])
                scores['activity'] = activity
                scores['dataset'] = dataset
                rec_scores.append(scores)
            for cutoff in [5, 10, 20]:

                latest_result = pd.read_csv(
                    f'results/full_recommendations/performance/{max(filter(lambda f: f.startswith(f"rec_cutoff_{cutoff}"), os.listdir("results/full_recommendations/performance")), key=lambda d: datetime.datetime.strptime(d, f"rec_cutoff_{cutoff}_relthreshold_1_%Y_%m_%d_%H_%M_%S.tsv"))}',
                    sep='\t')
                for _, r in latest_result.iterrows():
                    intermediate_result = {"activity": activity, 'model': r['model'].replace(
                        "UserKNN_nn=40_sim=cosine_imp=aiolli_bin=False_shrink=0_norm=True_asymalpha=_tvalpha=_tvbeta=_rweights=",
                        "User-KNN").replace(
                        "ItemKNN_nn=40_sim=cosine_imp=aiolli_bin=False_shrink=0_norm=True_asymalpha=_tvalpha=_tvbeta=_rweights=",
                        "Item-KNN").replace(
                        "UserKNN_nn=40_sim=cosine_imp=classical_bin=False_shrink=0_norm=True_asymalpha=_tvalpha=_tvbeta=_rweights=",
                        "User-KNN").replace(
                        "ItemKNN_nn=40_sim=cosine_imp=classical_bin=False_shrink=0_norm=True_asymalpha=_tvalpha=_tvbeta=_rweights=",
                        "Item-KNN").replace('PureSVD_factors=10', 'SVD').replace(
                        "BPRMF_seed=42_e=10_bs=1_f=10_lr=0$001_bias_reg=0_u_reg=0$0025_pos_i_reg=0$0025_neg_i_reg=0$0025_up_neg_i_f=True_up_u=True_up_i=True_up_b=True",
                        'BPRMF').replace("Random_seed=42", "Random"), 'variant': dataset,'dtv':dtv, 'cutoff': cutoff,
                                           "nDCG": r['nDCG'], 'ARP': r['ARP']}
                    result_elliot.append(intermediate_result)

result_df = pd.DataFrame(result_elliot)
result_df.to_csv(f'results-full_recommendations0.5.csv', index=False)
scores_df = pd.concat(rec_scores)
scores_df.to_csv("rec_scores.csv",index=False)

