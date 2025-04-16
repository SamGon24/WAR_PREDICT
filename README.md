# WAR_PREDICT
Machine Learning model that estimates a player's WAR and HR in a season using data scrapped from BaseballReference. 
The model was trained using the script alt_trainer_2.py and the main script is alt_predict_2.py.

The primary philosophy of this project is to effectively predict a baseball player's Wins Above Replacement (WAR) by scaling variables and using a Random Forest Regressor to make accurate predictions.

Until now, the model is only applicable to hitters and their specific implications, pitchers would require a different approach. Also, defensive stats were skipped for this preliminar analysis.

The were implemented in APP_FINAL.py which serves as the back-end for a website that allows a simple to use emo of the script.

To locally run this: 

1. Open a terminal and activate the venv mlb_env
2. Run APP_FINAL.py to locally run the website

If you want to train the model again:

1. Run alt_trainer_2.py, this will train the model again and preload the pkl files (the model and scaler)
2. Run alt_predict_2.py (alt_predict_graph.py will show a histogram for a better graphic understanding of a player's history)


Here is the link: https://web-production-1a459.up.railway.app/

Special thanks to https://www.baseball-reference.com/ for provinding the data for this project.