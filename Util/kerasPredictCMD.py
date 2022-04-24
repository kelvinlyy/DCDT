py_env = "/data/yylaiai/anaconda3/envs/audee_test/bin/python"

Util_path = "../Util"
get_outputs_py = f"{Util_path}/get_outputs_keras.py"
get_prediction_py = f"{Util_path}/get_prediction_keras.py"
get_coverage_py = f"{Util_path}/get_coverage_keras.py"


# generate command for computing intermediate outputs
def get_outputs_cmd(model_framework, db_flag, model_key, input_key, predictions_key, layer_idx):
    cmd = f"{py_env} {get_outputs_py} {model_framework} {db_flag} {model_key} {input_key} {predictions_key} {layer_idx}"
    return cmd

# generate command for computing model prediction (only last layer)
def get_prediction_cmd(model_framework, db_flag, model_key, input_key, predictions_key):
    cmd = f"{py_env} {get_prediction_py} {model_framework} {db_flag} {model_key} {input_key} {predictions_key}"
    return cmd

# generate command for computing coverage of DL backend
def get_coverage_cmd(model_framework, db_flag, model_key, input_key, coverage_key):
    cmd = f"{py_env} {get_coverage_py} {model_framework} {db_flag} {model_key} {input_key} {coverage_key}"
    return cmd