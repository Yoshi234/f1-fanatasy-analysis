import fastf1
import pandas as pd
import numpy as np
from setup.get_track_augment import get_track_speeds
from fastf1 import SessionNotAvailableError

def get_data():
    session = fastf1.get_session(2023, 22, 'FP1')
    session.load()
    fastest_lap = session.laps.pick_fastest()
    print(fastest_lap['LapTime'])  

if __name__ == "__main__":
    test()