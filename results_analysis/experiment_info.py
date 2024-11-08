SEED = 0
N_EXPERIMENTS = 30
N_POPULATION = 100

algorithms = []
algorithms.append('n3o')
# algorithms.append('sms_emoa')
# algorithms.append('sms_moneat')

iter_num = {'n3o': 200, 'sms_emoa': 6000, 'sms_moneat': 18000, 'sfe': 6000}
experiment = {'n3o': '2', 'sms_emoa': '5x', 'sms_moneat': '6', 'sfe': 1}

datasets = []

datasets.append('Colorectal_GSE25070')
datasets.append('Colorectal_GSE32323')
datasets.append('Colorectal_GSE44076')
datasets.append('Colorectal_GSE44861')
# datasets.append('Liver_GSE14520_U133A') 
# datasets.append('Liver_GSE50579')
# datasets.append('Liver_GSE62232') 
datasets.append('Leukemia_GSE22529_U133A') 
datasets.append('Leukemia_GSE22529_U133B') 
# datasets.append('Leukemia_GSE33615')
# datasets.append('Leukemia_GSE63270') 
# datasets.append('ALL-AML-full')
datasets.append('Breast_GSE22820') 
datasets.append('Breast_GSE59246') 
datasets.append('Breast_GSE70947')	
# datasets.append('breastCancer-full') 
# datasets.append('Prostate_GSE6919_U95Av2')
# datasets.append('Prostate_GSE6919_U95B')
# datasets.append('Prostate_GSE11682')
# datasets.append('prostate_tumorVSNormal-full')

# datasets.append('Leukemia_GSE71935') # Only 9
# datasets.append('Liver_GSE57957')
# datasets.append('Prostate_GSE46602')
# datasets.append('Colorectal_GSE8671') # SMS-MONEAT 
# datasets.append('Breast_GSE42568')

# datasets.append('Prostate_GSE6919_U95C')

dataset_names = []
dataset_names.append('Colon$_1$')
dataset_names.append('Colon$_2$')
dataset_names.append('Colon$_3$')
dataset_names.append('Colon$_4$')
dataset_names.append('Hígado$_1$')
dataset_names.append('Hígado$_2$')
dataset_names.append('Hígado$_3$')
dataset_names.append('Leucemia$_1$')
dataset_names.append('Leucemia$_2$')
dataset_names.append('Leucemia$_3$')
dataset_names.append('Leucemia$_4$')
dataset_names.append('Leucemia$_5$')
dataset_names.append('Mama$_1$')
dataset_names.append('Mama$_2$')
dataset_names.append('Mama$_3$')
dataset_names.append('Mama$_4$')
dataset_names.append('Próstata$_1$')
dataset_names.append('Próstata$_2$')
dataset_names.append('Próstata$_3$')
dataset_names.append('Próstata$_4$')