import os
import uuid
import time

template = '''#!/bin/bash
#SBATCH --job-name=bzs_4m_lr_4e5
#SBATCH --time=999:59:00
#SBATCH --output=/var/cr01_data/logs/slurm_%j.log
#SBATCH --nodes=32
#SBATCH -n 32
#SBATCH --chdir=/var/cr01_data/gpt-neox-jue

nvidia-smi

export JOB_ID={{JOB_ID}}

export CUDA_HOME=/usr/local/cuda-11.6

export GPUS_PER_NODE=8
export WORLD_SIZE=256
export MASTER_PORT=9901
#srun echo $MASTER_ADDR

netif=enp12s0
export GLOO_SOCKET_IFNAME=${netif}
export NCCL_SOCKET_IFNAME=${netif}
#export MASTER_ADDR=172.27.6.25
export WANDB_API_KEY=6fae2eb8adcb7b687f143acdf784e301ad45d82a

# >>> conda initialize >>>
# !! Contents within this block are managed by 'conda init' !!
__conda_setup="$('/root/miniconda3/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "/root/miniconda3/etc/profile.d/conda.sh" ]; then
        . "/root/miniconda3/etc/profile.d/conda.sh"
    else
        export PATH="/root/miniconda3/bin:$PATH"
    fi
fi
unset __conda_setup

if [ -f "/root/miniconda3/etc/profile.d/mamba.sh" ]; then
    . "/root/miniconda3/etc/profile.d/mamba.sh"
fi
# <<< conda initialize <<<

mamba activate neox
git pull

python ./deepy.py train.py  /var/cr01_data/gpt-neox-jue/configs/rp_7b_512_nodes_continue.yml /var/cr01_data/gpt-neox-jue/configs/rp_data_setup_test.yml

'''

if __name__ == '__main__':

    job_id = str(uuid.uuid4())
    node_size = 32
    template = template.replace('{{JOB_ID}}', job_id)

    with open('configs/train_to_submit.slurm.sh', 'w') as f:
        f.write(template)
        
    for i in range(node_size):
        os.system('sbatch configs/train_to_submit.slurm.sh')
        time.sleep(10)