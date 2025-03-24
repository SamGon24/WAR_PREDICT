# WAR_PREDICT
Machine Learning model that estimates a player's WAR and HR in a season using data scrapped from BaseballReference. 
The model was trained using the script alt_trainer_2.py and the main script is alt_predict_2.py.

The primary philosophy of this project is to effectively predict a baseball player's Wins Above Replacement (WAR) by scaling variables and using a Random Forest Regressor to make accurate predictions.

Until now, the model is only applicable to hitters and their specific implications, pitchers would require a different approach. Also, defensive stats were skipped for this preliminar analysis.

The were implemented in APP_FINAL.py which serves as the back-end for a website that allows a simple to use emo of the script.

Here is the link: https://web-production-1a459.up.railway.app/

Special thanks to https://www.baseball-reference.com/ for provinding the data for this project.