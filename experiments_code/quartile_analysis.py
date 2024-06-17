import datetime

import pandas as pd
import os

import yaml

with open("config_files/basic_configuration.yml", "r") as basic_file:
    basic_config = yaml.load(basic_file)

from elliot.run import run_experiment

result_elliot = []
print("Done! We are now starting the Elliot's experiment")
for seed in range(100):

    for activity in ['physical', 'nature', 'cultural', 'social', 'environmental']:

        for variant in ['prolific']:

            distances = pd.read_csv(f"data/greens/distances_{variant}_{activity}.tsv", sep='\t')
            interactions = pd.read_csv(f"data/greens/interactions_{variant}_{activity}.tsv", sep='\t', header=None,
                                       names=['user', 'park', 'rating', 'timestamp'])
            for quartile in [1, 2, 3, 4]:
                relevant_parks = distances[distances['quantile'] < quartile]
                q_visits = pd.merge(interactions, relevant_parks, on=['user', 'park'])[['user', 'park', 'rating', 'timestamp']]
                sampled = q_visits.sample(frac=1 / quartile)
                sampled.to_csv(f"data/greens/interactions_{variant}_{activity}_{quartile}.tsv", sep='\t', index=False, header=False)
                basic_config['experiment']['data_config'][
                    'dataset_path'] = f'../data/greens/interactions_{variant}_{activity}_{quartile}.tsv'
                with open(f"config_files/basic_configuration_greens_{activity}_{variant}_{quartile}.yml", "w") as config_file:
                    yaml.dump(basic_config, config_file)
                run_experiment(f"config_files/basic_configuration_greens_{activity}_{variant}_{quartile}.yml")

                for cutoff in [5, 10, 20]:

                    latest_result = pd.read_csv(
                        f'results/greens/performance/{max(filter(lambda f: f.startswith(f"rec_cutoff_{cutoff}"), os.listdir("results/greens/performance")), key=lambda d: datetime.datetime.strptime(d, f"rec_cutoff_{cutoff}_relthreshold_1_%Y_%m_%d_%H_%M_%S.tsv"))}',
                        sep='\t')
                    for _, r in latest_result.iterrows():
                        intermediate_result = {"seed": seed, "activity": activity, "interactions": len(sampled.index),
                                               'model': r['model'].replace(
                                                   "UserKNN_nn=40_sim=cosine_imp=aiolli_bin=False_shrink=0_norm=True_asymalpha=_tvalpha=_tvbeta=_rweights=",
                                                   "User-KNN").replace(
                                                   "ItemKNN_nn=40_sim=cosine_imp=aiolli_bin=False_shrink=0_norm=True_asymalpha=_tvalpha=_tvbeta=_rweights=",
                                                   "Item-KNN").replace(
                                                   "UserKNN_nn=40_sim=cosine_imp=classical_bin=False_shrink=0_norm=True_asymalpha=_tvalpha=_tvbeta=_rweights=",
                                                   "User-KNN").replace(
                                                   "ItemKNN_nn=40_sim=cosine_imp=classical_bin=False_shrink=0_norm=True_asymalpha=_tvalpha=_tvbeta=_rweights=",
                                                   "Item-KNN").replace('PureSVD_factors=10', 'SVD').replace(
                                                   "BPRMF_seed=42_e=10_bs=1_f=10_lr=0$001_bias_reg=0_u_reg=0$0025_pos_i_reg=0$0025_neg_i_reg=0$0025_up_neg_i_f=True_up_u=True_up_i=True_up_b=True",
                                                   'BPRMF').replace("Random_seed=42", "Random"), 'variant': variant,
                                               'quartile': quartile, 'cutoff': cutoff,
                                               "nDCG": r['nDCG'], 'ARP': r['ARP']
                                               }
                        result_elliot.append(intermediate_result)

result_df = pd.DataFrame(result_elliot)
result_df.to_csv('results.csv', index=False)
