from taipy.gui import Gui, Icon
import taipy.gui.builder as tgb
import pandas as pd
import tp_funcs

# read in predictions for a specific race
predictions = pd.read_csv('../../results/baku/predictions.csv')

# get the drivers in the race
driverRefs = list(predictions['Driver'])

# read in and filter driver data for name and number information
drivers = pd.read_csv('../../data/drivers.csv')

drivers = drivers[drivers['code'].isin(driverRefs)]

# Duplicate codes for ALB and VER
try:
    drivers = drivers.drop(26, axis=0)
    drivers = drivers.drop(817)
except:
    pass

#print(drivers)

# function to get top 3 features from the race for each driver and return in dataframe,
# function should probably have a parameter to set the race
features = tp_funcs.get_features()


with tgb.Page() as page:
    with tgb.part(class_name="card full"):
        with tgb.layout(columns="1 1"):
            for i in range(2):
                # determine the order that the cards are shown with slicing
                # i == 0 is the first column and i == 1 is the second column
                if i == 0:
                    #slice = driverRefs[:len(driverRefs) // 2]
                    slice = driverRefs[0::2]
                elif i == 1:
                    #slice = driverRefs[len(driverRefs) // 2:]
                    slice = driverRefs[1::2]

                with tgb.part(class_name='column'):
                    for d in slice:


                        # changing name to match image files
                        team = predictions.loc[predictions['Driver'] == d, 'Constructor'].item()
                        if team == 'aston_martin':
                            team = 'astonmartin'
                        elif team == 'haas':
                            team = 'haasf1team'
                        elif team == 'sauber':
                            team = 'kicksauber'
                        elif team == 'rb':
                            team = 'racingbulls'
                        elif team == 'red_bull':
                            team = 'redbullracing'

                        team_logo_path = 'images/team_logos/2025' + team + 'logowhite.avif'
                        
                        first_name = drivers.loc[drivers['code'] == d, 'forename'].item()
                        last_name = drivers.loc[drivers['code'] == d, 'surname'].item()
                        driver_name = first_name + ' ' + last_name

                        driver_number = drivers.loc[drivers['code'] == d, 'number'].item()

                        # changing to haas to match haas driver image file names
                        if team == 'haasf1team':
                            team = 'haas'
                        
                        first_name_3 = first_name[:3].lower()
                        last_name_3 = last_name[:3].lower()

                        # specific name edits to match driver image file names
                        if last_name_3 == 'kim':
                            last_name_3 = 'ant'
                        elif last_name_3 == 'fra':
                            first_name_3 = 'fra'
                            last_name_3 = 'col'
                        elif first_name_3 == 'nic':
                            last_name_3 = 'hul'

                        driver_image_path = 'images/drivers/2025' + team + first_name_3 + last_name_3 + '01right.avif'

                        # get features for specific driver
                        driver_features = list(features.loc[features['driverRef'] == d, 'feature'])
                        driver_feature_scores = list(features.loc[features['driverRef'] == d, 'feature_score'])

                        # get prediction for specific driver
                        driver_avg = predictions.loc[predictions['Driver'] == d, 'adj_pred_order2'].item()
                        driver_pred = predictions.loc[predictions['Driver'] == d, 'fp'].item()

                        with tgb.part(class_name="card driver"):
                            with tgb.layout(columns="1 5 2", class_name=team):
                                with tgb.part():
                                    if team == 'mercedes' or team == 'alpine':
                                        tgb.image(team_logo_path, class_name='team-logo alp-mer')
                                    else:
                                        tgb.image(team_logo_path, class_name='team-logo')
                                with tgb.part():
                                    tgb.text(driver_name, class_name='driver-name')
                                with tgb.part():
                                    tgb.text(driver_number, class_name='driver-number')
                            with tgb.layout(columns="1 1"):
                                with tgb.part():
                                    tgb.image(driver_image_path, class_name='driver-image')
                                with tgb.part(class_name='features'):
                                    tgb.text(driver_features[0] + ': ' + str(driver_feature_scores[0]))
                                    tgb.text(driver_features[1] + ': ' + str(driver_feature_scores[1]))
                                    tgb.text(driver_features[2] + ': ' + str(driver_feature_scores[2]))
                            with tgb.layout(columns="1 1"):
                                with tgb.part(class_name='avg-fin'):
                                    tgb.text('Average Finish')
                                    tgb.text(str(round(driver_avg, 2)))
                                if int(driver_pred) <= 3:
                                    pred_class = ' place' + str(int(driver_pred))
                                else:
                                    pred_class = ''
                                with tgb.part(class_name='pred'+pred_class):
                                    tgb.text('Prediction')
                                    tgb.text('P' + str(int(driver_pred)))

# to prevent mozilla firefox from opening, run this command export BROWSER=false
# runs well for editing with the command ls f1_taipy.py | entr -r python3 f1_taipy.py
if __name__ == "__main__":
    Gui(page=page, css_file='f1_taipy.css').run(title='Driver Cards', web_browser = False)