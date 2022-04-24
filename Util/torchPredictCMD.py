py_env = '/data/yylaiai/anaconda3/envs/fyp_v3/bin/python'
Util_path = "../Util"

get_prediction_py = f"{Util_path}/get_prediction_torch.py"

'''
Arguments:
[1]: db flag of redis server
[2]: model key
[3]: input key
[4]: layer idx
'''

# generate command for computing model prediction (only last layer)
def get_prediction_cmd(db_flag, model_key, input_key, layer_idx):
    cmd = f"{py_env} {get_prediction_py} {db_flag} {model_key} {input_key} {layer_idx}"
    return cmd