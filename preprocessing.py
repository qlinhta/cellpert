import pandas as pd
import dask.dataframe as dd
import matplotlib.pyplot as plt
import numpy as np
import logging

plt.style.use('default')
plt.rc('text', usetex=True)
plt.rc('font', family='sans-serif')
plt.rc('font', size=14)
plt.rc('axes', titlesize=14)
plt.rc('axes', labelsize=14)
plt.rc('xtick', labelsize=14)
plt.rc('ytick', labelsize=14)
plt.rc('legend', fontsize=14)
plt.rc('lines', markersize=10)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

logger.info('🫡 Loading the data')
de_train_dd = dd.read_parquet('datasets/de_train.parquet')
logger.info('👌 de_train_dd loaded')
adata_train_dd = dd.read_parquet('datasets/adata_train.parquet')
logger.info('👌 adata_train_dd loaded')
multiome_train_dd = dd.read_parquet('datasets/multiome_train.parquet')
logger.info('👌 multiome_train_dd loaded')

adata_obs_meta = pd.read_csv('datasets/adata_obs_meta.csv')
multiome_obs_meta = pd.read_csv('datasets/multiome_obs_meta.csv')
multiome_var_meta = pd.read_csv('datasets/multiome_var_meta.csv')
id_map = pd.read_csv('datasets/id_map.csv')
final_prediction = pd.DataFrame(id_map, columns=["cell_type", "sm_name"])
pubchem = pd.read_csv('datasets/pubchem_df.csv')

de_train = de_train_dd.compute()
adata_train = adata_train_dd.compute()
multiome_train = multiome_train_dd.compute()

logger.info('🫡 Preprocessing the data')
for i in range(len(de_train['sm_name'])):
    if de_train['sm_name'][i] != '5-(9-Isopropyl-8-methyl-2-morpholino-9H-purin-6-yl)pyrimidin-2-amine' and \
            de_train['sm_name'][
                i] != '2-(4-Amino-5-iodo-7-pyrrolo[2,3-d]pyrimidinyl)-5-(hydroxymethyl)oxolane-3,4-diol':
        de_train['sm_name'][i] = de_train['sm_name'][i].split('(')[0].strip()
for i in range(len(de_train['sm_name'])):
    if de_train['sm_name'][i] != '5-(9-Isopropyl-8-methyl-2-morpholino-9H-purin-6-yl)pyrimidin-2-amine' and \
            de_train['sm_name'][
                i] != '2-(4-Amino-5-iodo-7-pyrrolo[2,3-d]pyrimidinyl)-5-(hydroxymethyl)oxolane-3,4-diol':
        de_train['sm_name'][i] = de_train['sm_name'][i].split(';')[0].strip()

features_columns = ["cell_type", "sm_name", "sm_lincs_id", "SMILES", "control"]
targets = de_train.drop(columns=features_columns)
features = pd.DataFrame(de_train, columns=features_columns)

# Aggregating adata_train and multiome_train
agg_func = {'count': ['mean', 'sum'], 'normalized_count': ['mean', 'sum']}
adata_train_grouped = adata_train.groupby(['obs_id']).agg(agg_func).reset_index()
multiome_train_grouped = multiome_train.groupby(['obs_id']).agg(agg_func).reset_index()

# Flatten the multi-level column names in adata_train_grouped and multiome_train_grouped
adata_train_grouped.columns = ['_'.join(col).strip() for col in adata_train_grouped.columns.values]
multiome_train_grouped.columns = ['_'.join(col).strip() for col in multiome_train_grouped.columns.values]

# Rename the 'obs_id_' column back to 'obs_id' to make it ready for the merge
adata_train_grouped.rename(columns={'obs_id_': 'obs_id'}, inplace=True)
multiome_train_grouped.rename(columns={'obs_id_': 'obs_id'}, inplace=True)

# Join aggregated dataframes with metadata
adata_full = pd.merge(adata_train_grouped, adata_obs_meta, on='obs_id', how='left')
multiome_full = pd.merge(multiome_train_grouped, multiome_obs_meta, on='obs_id', how='left')

# Aggregating by 'cell_type' and 'sm_name' or 'donor_id'
agg_func_meta = {'count_mean': 'mean', 'count_sum': 'sum', 'normalized_count_mean': 'mean',
                 'normalized_count_sum': 'sum'}
adata_summary = adata_full.groupby(['cell_type', 'sm_name']).agg(agg_func_meta).reset_index()
multiome_summary = multiome_full.groupby(['cell_type', 'donor_id']).agg(agg_func_meta).reset_index()

features_enriched = pd.merge(features, adata_summary, on=['cell_type', 'sm_name'], how='left')
features_enriched = pd.merge(features_enriched, multiome_summary, on=['cell_type'],
                             how='left')  # Donor ID can be included if it aligns
features_enriched['sm_name'] = features_enriched['sm_name'].replace('O-Demethylated Adapalene', 'O-Desmethyl Adapalene')
features_enriched['sm_name'] = features_enriched['sm_name'].replace('IN1451',
                                                                    '2-(4-Amino-5-iodo-7-pyrrolo[2,3-d]pyrimidinyl)-5-(hydroxymethyl)oxolane-3,4-diol')

features_enriched = features_enriched.merge(pubchem, on='sm_name', how='left')
# Concatenate the features and targets
de_train_enriched = pd.concat([features_enriched, targets], axis=1)
# Save the enriched dataframe
de_train_enriched.to_csv('enriched/de_train_enriched.csv', index=False)

final_prediction = pd.read_csv('datasets/final_prediction.csv')

final_prediction['sm_name'] = final_prediction['sm_name'].replace('O-Demethylated Adapalene', 'O-Desmethyl Adapalene')
final_prediction['sm_name'] = final_prediction['sm_name'].replace('IN1451',
                                                                  '2-(4-Amino-5-iodo-7-pyrrolo[2,3-d]pyrimidinyl)-5-(hydroxymethyl)oxolane-3,4-diol')
for i in range(len(final_prediction['sm_name'])):
    if final_prediction['sm_name'][i] != '5-(9-Isopropyl-8-methyl-2-morpholino-9H-purin-6-yl)pyrimidin-2-amine' and \
            final_prediction['sm_name'][
                i] != '2-(4-Amino-5-iodo-7-pyrrolo[2,3-d]pyrimidinyl)-5-(hydroxymethyl)oxolane-3,4-diol':
        final_prediction['sm_name'][i] = final_prediction['sm_name'][i].split('(')[0].strip()

for i in range(len(final_prediction['sm_name'])):
    if final_prediction['sm_name'][i] != '5-(9-Isopropyl-8-methyl-2-morpholino-9H-purin-6-yl)pyrimidin-2-amine' and \
            final_prediction['sm_name'][
                i] != '2-(4-Amino-5-iodo-7-pyrrolo[2,3-d]pyrimidinyl)-5-(hydroxymethyl)oxolane-3,4-diol':
        final_prediction['sm_name'][i] = final_prediction['sm_name'][i].split(';')[0].strip()

final_prediction = final_prediction.merge(pubchem, on='sm_name', how='left')
final_prediction.to_csv('enriched/final_prediction.csv', index=False)
