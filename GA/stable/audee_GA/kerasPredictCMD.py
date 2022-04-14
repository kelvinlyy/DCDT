py_env = "/data/yylaiai/anaconda3/envs/audee_test/bin/python"
get_outputs_py = "get_outputs_keras.py"
get_prediction_py = "get_prediction_keras.py"


# generate command for computing intermediate outputs
def get_outputs_cmd(model_framework, db_flag, model_key, input_key, predictions_key, layer_idx):
    cmd = f"{py_env} {get_outputs_py} {model_framework} {db_flag} {model_key} {input_key} {predictions_key} {layer_idx}"
    return cmd

# generate command for computing model prediction (only last layer)
def get_prediction_cmd(model_framework, db_flag, model_key, input_key, predictions_key):
    cmd = f"{py_env} {get_prediction_py} {model_framework} {db_flag} {model_key} {input_key} {predictions_key}"
    return cmd
