import pandas as pd
import matplotlib as mpl
import numpy as np 
from matplotlib import pyplot as plt
from matplotlib.collections import LineCollection
import fastf1

def set_fig(track):
    # borrowing from fastf1 code examples

    year = track["ref_year"].item()
    track_name = track["ref_name"].item()
    ses = "Q"
    colormap = mpl.cm.plasma

    # include the telemetry data
    session = fastf1.get_session(year, track_name, ses)
    session.load()
    fastest_lap = session.laps.pick_fastest()

    x = fastest_lap.telemetry["X"]
    y = fastest_lap.telemetry["Y"]
    color = fastest_lap.telemetry["Speed"]

    # create line segments for coloring individual segments
    points = np.array([x,y]).T.reshape(-1,1,2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)

    # plot the data
    fig, ax = plt.subplots(sharex=True, sharey=True, figsize=(12, 6.75))
    fig.suptitle(f"{track_name} - Speed", size=24, y=0.97)

    plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.12)
    ax.axis("off")

    ax.plot(fastest_lap.telemetry["X"], fastest_lap.telemetry["Y"], 
            color="black", linestyle="-", linewidth=16, zorder=0)

    norm = plt.Normalize(color.min(), color.max())
    lc = LineCollection(segments, cmap=colormap, norm=norm, 
                        linestyle="-", linewidth=5)
    lc.set_array(color)

    line = ax.add_collection(lc)

    cbaxes = fig.add_axes([0.25, 0.05, 0.5, 0.05])
    normlegend = mpl.colors.Normalize(vmin=color.min(), vmax=color.max())
    legend = mpl.colorbar.ColorbarBase(cbaxes, norm=normlegend, cmap=colormap,
                                    orientation="horizontal")

    # Show the plot
    plt.savefig("track.png")

def write_quarto():
    year = None
    event = None
    results = pd.read_csv("new_results.csv")
    with open("info.txt", 'r') as f:
        x = f.readlines()
        event = x[0].strip('\n')
        year = x[1].strip('\n')
        round = x[2].strip('\n')

    # set important stats
    top3 = results.sort_values(by='prob of top 3 finish', ascending=False)
    top3 = top3.loc[:3].reset_index()

    # get the year and fastest lap of the most recent year as reference
    track_dat = pd.read_feather("../data/track_data.feather")
    track = track_dat.loc[track_dat['ref_name'] == event]

    set_fig(track)

    # the main text to write to the quarto file for running later
    main_text = f'''# F1-Analysis for the {year} {event}

## Track Layout / Information

![track layout and speeds](track.png){{fig-align="center"}}

Analysis for conducted using a logistic regression with 30 features
for additional information about the modeling process, please send me a 
message on GitHub issues

## Top3 Finishing Position Predictions

The table below presents odds of placing in the top 3 positions for each 
of the drivers in the current standings for F1. 

```{{python}}
#| echo: false
import pandas as pd

results = pd.read_csv("new_results.csv")
results.sort_values(by="prob of top 3 finish", ascending=False)
```

We can see here that the top 3 predicted finishers are:
    
+ {top3["driver name"][0]} : probability of top 3 finish - {top3["prob of top 3 finish"][0]*100:.2f}%
+ {top3["driver name"][1]} : probability of top 3 finish - {top3["prob of top 3 finish"][1]*100:.2f}%
+ {top3["driver name"][2]} : probability of top 3 finish - {top3["prob of top 3 finish"][2]*100:.2f}%
    '''

    with open("outcomes.qmd", "w") as f:
        f.write(main_text)
    
    return

def main():
    write_quarto()

if __name__ == "__main__":
    main()