from urllib.request import urlopen
import json
import pandas as pd
from playsound import playsound
import time
from datetime import timedelta

start_time = time.time()

# Set equal to False for just sound 
sound_and_lights = False
    
# If there is a problem after this line, consider changing session_key=9690 for api requests in lines 16, 21, and 61

# Getting driver data
drivers_r = urlopen('https://api.openf1.org/v1/drivers?&session_key=latest')
driver_data = json.loads(drivers_r.read().decode('utf-8'))
driver_df = pd.DataFrame(driver_data)

# Getting leader at current time (start of the race)
response = urlopen('https://api.openf1.org/v1/position?meeting_key=1254&session_key=latest&position=1')
data = json.loads(response.read().decode('utf-8'))
positions_df = pd.DataFrame(data)
#print(positions_df)

positions_df = positions_df.merge(driver_df, how='left', left_on='driver_number', right_on='driver_number')

# Initialize light
if sound_and_lights:
    from phue import Bridge

    b = Bridge(open('bridge_ip_address.txt').readline())
    
    # Connect if necessary
    #b.connect()

    light_names = b.get_light_objects('name')

    # Dictionary of team hue colors
    hues = {'Ferrari': 100, 'McLaren': 2000,
            'Red Bull Racing': 10000, 'Aston Martin': 24000,
            'Mercedes': 34000, 'Williams': 42000,
            'Racing Bulls': 46000, 'Alpine': 55000}
    
    # Default light setting and initialize hue based on leading team
    light_names['Hue Go 1'].on = True
    light_names['Hue Go 1'].hue = hues[positions_df.iloc[-1, ]['team_name']]
    light_names['Hue Go 1'].saturation = 240
    light_names['Hue Go 1'].brightness = 70

prev_leader = positions_df.iloc[-1, ]['full_name'].title()

playsound('F1_Sounds/Race_Start.mp3')

# Should run for 1 hour and 45 minutes assuming that each loop takes about 5 seconds
for i in range(1260):
    print('Time since start of session:', timedelta(seconds = time.time() - start_time))

    sounds_played = 0

    response = urlopen('https://api.openf1.org/v1/position?meeting_key=1254&session_key=9689&position=1')
    data = json.loads(response.read().decode('utf-8'))

    positions_df = pd.DataFrame(data)

    positions_df = positions_df.merge(driver_df, how='left', left_on='driver_number', right_on='driver_number')

    #print(positions_df)


    leader = positions_df.iloc[-1, ]['full_name'].title()
    #print(leader)

    if leader != prev_leader:

        # Play narration of new leader
        playsound('F1_Sounds/Lead_Change/' + leader.replace(' ', '_') + '_Lead.mp3')

        sounds_played += 1

        # Special sounds
        if leader == 'Max Verstappen':
            playsound('F1_Sounds/Du-Du-Du-Du-Max_Verstappen.mp3')
            sounds_played += 1
        elif leader == 'Charles Leclerc':
            playsound('F1_Sounds/Charles_Leclerc-No.mp3')
            sounds_played += 1
        
        # Change color of light to leading team
        if sound_and_lights:
            light_names['Hue Go 1'].hue = hues[positions_df.iloc[-1, ]['team_name']]

        # Sleep based on number of sounds played
        if sounds_played == 1:
            time.sleep(3)
        elif sounds_played >= 2:
            time.sleep(1)

        prev_leader = leader
    
    # Show the last 2 leaders of the session
    print(positions_df[['date', 'position', 'name_acronym', 'team_name']].tail(2))

    # Sleep for 5 seconds if no sounds played
    if sounds_played == 0:
        time.sleep(5)