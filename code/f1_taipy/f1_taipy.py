from taipy.gui import Gui, Icon
import taipy.gui.builder as tgb
import pandas as pd

team_logo_path = 'images/team_logos/2025mercedeslogowhite.avif'
driver_name = 'George Russell'
driver_number = 63

driver_image_path = "images/drivers/2025mercedesgeorus01right.avif"

predictions = pd.read_csv('../../results/baku/predictions.csv')

driver_avg = predictions.loc[predictions['Driver'] == 'RUS', 'adj_pred_order2'].item()
#print(driver_avg)
#print(str(round(driver_avg, 2)))

driver_pred = predictions.loc[predictions['Driver'] == 'RUS', 'fp'].item()

with tgb.Page() as page:
    with tgb.part(class_name="card"):
        with tgb.layout(columns="1 5 2"):
            with tgb.part():
                tgb.image(team_logo_path, class_name='team-logo')
            with tgb.part():
                tgb.text(driver_name, class_name='driver-name')
            with tgb.part():
                tgb.text(driver_number, class_name='driver-number')
        with tgb.layout(columns="1 1"):
            with tgb.part():
                tgb.image(driver_image_path, class_name='driver-image')
            with tgb.part(class_name='features'):
                tgb.text('Feature 1: -5.24')
                tgb.text('Feature 2: -1.16')
                tgb.text('Feature 3: -0.63')
        with tgb.layout(columns="1 1"):
            with tgb.part():
                tgb.text('Average Finish')
                tgb.text(str(round(driver_avg, 2)))
            with tgb.part():
                tgb.text('Prediction')
                tgb.text('P' + str(int(driver_pred)))

# runs well for editing with the command ls f1_taipy.py | entr -r python3 f1_taipy.py
if __name__ == "__main__":
    Gui(page=page, css_file='f1_taipy.css').run(title='Driver Cards', web_browser = False)