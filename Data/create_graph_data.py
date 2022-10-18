
from Data.data_prepare import *

args={}
args['input_csv'] = '../data/admet.csv'
args['output_bin'] = '../data/admet.bin'
args['output_csv'] = '../data/admet_group.csv'

built_data_and_save_for_splited(
        origin_path=args['input_csv'],
        save_path=args['output_bin'],
        group_path=args['output_csv'],
        task_list_selected=None
         )






