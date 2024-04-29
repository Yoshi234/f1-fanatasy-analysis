# make sure to chmod +x script.sh before running this file
python3 run_analysis.py
python3 write_quarto.py
quarto render outcomes.qmd
git add .
git commit -m "made predictions for most recent weekend"
git push