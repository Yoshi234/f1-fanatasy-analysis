eval "$(conda shell.bash hook)"
conda activate R_env
CUR_DIR=$(pwd)
cd honors
./f1_analysis.sh
cd "$CUR_DIR"
quarto render
git add .
git commit -m "new publication of 2024 f1 analysis"
git push
