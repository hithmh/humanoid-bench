<h1>SSM\_RL</span></h1>

\---

export TASK="h1hand-sit\_simple-v0"

python -m ssmrl.train exp\_name=SSMRL task=humanoid\_${TASK} seed=0

cd workspace/humanoid-bench/SSM_RL/
conda activate tdmpc2
python -m ssmrl.train