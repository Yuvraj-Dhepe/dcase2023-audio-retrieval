#!/bin/bash
export $(xargs<.env)

# Original with 30 epochs
# python postprocessing/modular_multi_run_processing.py --config conf_yamls/base_configs/cap_0_new_conf.yaml --run_id wykbp35b

# Original with 100 epochs
python postprocessing/modular_multi_run_processing.py --config conf_yamls/base_configs/cap_0_new_conf.yaml --run_id tqhxk9vc

python postprocessing/modular_multi_run_processing.py --config conf_yamls/base_configs/cap_1x_conf.yaml --run_id vj2b6kld

python postprocessing/modular_multi_run_processing.py --config conf_yamls/base_configs/cap_2x_conf.yaml --run_id 73wk6hjv

python postprocessing/modular_multi_run_processing.py --config conf_yamls/base_configs/cap_3x_conf.yaml --run_id dkvnyj0u

python postprocessing/modular_multi_run_processing.py --config conf_yamls/base_configs/cap_4x_conf.yaml --run_id e0s30cv1

python postprocessing/modular_multi_run_processing.py --config conf_yamls/base_configs/cap_5x_conf.yaml --run_id xunedzbr