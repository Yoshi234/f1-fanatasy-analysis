import pandas as pd
import fastf1
import fastf1.plotting
from matplotlib import pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
import os
import sys

def write_latex_table_to_pdf(
    table_shape=(2,2), 
    input_files=['hello.pdf'],
    output_tex='output.tex',
    folder='day1',
    output_name='output'
):
    x=1
    base_str = r'''
\documentclass{article}
\usepackage{graphicx}
\usepackage{array}
\usepackage[margin=0.2in]{geometry}
\begin{document}


\centering
'''
    table_size = r'   \begin{tabular}{|'
    for i in range(table_shape[1]):
        table_size += r'c|'
    table_size += r'}' + '\n'
    
    base_str += table_size + r'      \hline' + '\n'
    
    idx = 0
    w = 0.9/table_shape[0]
    
    for row in range(table_shape[0]):
        row_str = '      '
        for col in range(table_shape[1]):
            if (idx < len(input_files)) and (input_files[idx] != 'na') : 
                row_str += r'\includegraphics[width='+ f'{w}'  \
                        + r'\linewidth]' + r'{' \
                        + f'{input_files[idx]}' + r'}'
            if col < (table_shape[1] - 1):
                row_str += r'&'
            else:
                row_str += r'\\' + '\n'
            idx += 1
        
        base_str += row_str
    base_str += r'   \end{tabular}' + '\n'
    # base_str += r'\end{center}' + '\n'
    base_str += r'\end{document}'
            
    print(base_str)
    
    with open(f'{output_name}.tex', 'w') as f:
        f.write(base_str)
    
    os.system('make')
    os.system('mv {}.tex {}'.format(output_name, folder))
    os.system('mv {}.pdf {}'.format(output_name, folder))
    for file in os.listdir():
        if output_name in file: os.system('rm {}'.format(file))

# NOTE: MUST USE SEABORN 0.12.2
def plot_driver(
    laps_data, 
    driver='LEC',
    folder='',
    file_output='leclerc.pdf',
    session='day1'
):
    plt.clf()
    laps = laps_data.laps.pick_drivers(driver).pick_quicklaps()
    laps['LapTimeSec'] = laps['LapTime'].dt.total_seconds()
    fig, ax = plt.subplots(figsize=(8,8))
    # sns.set_theme(style='darkgrid')
    tire_palette = {'HARD':'#FFFFFF', 'MEDIUM':'#FFC906', 'SOFT':'#CC1E4A'}
    cmap_vals = LinearSegmentedColormap.from_list("green_red", ["green", "red"])

    x = sns.scatterplot(
        data=laps,
        x='LapNumber',
        y='LapTimeSec',
        palette=cmap_vals,
        ax=ax,
        hue="TyreLife",
        s=80,
        linewidth=1.5,
        legend='auto',
        edgecolor=laps['Compound'].map(tire_palette)
    )
    
    # for i, path in enumerate(x.collections[0].get_paths()):
    #     edge_color = tire_palette[laps.loc[i, 'Compound']]
    #     path.set_edgecolor(edge_color)
        
    # plt.colorbar(label='TyreLife')
    ax.set_facecolor("darkgray")
    ax.set_xlabel("Lap Number")
    ax.set_ylabel("Lap Time")

    for x in laps['Stint'].unique():
        start_lap = laps.loc[laps['Stint']==x,'LapNumber'].min()
        fin_lap = laps.loc[laps['Stint']==x,'LapNumber'].max()
        if start_lap == fin_lap:
            plt.axvline(x=fin_lap, color='purple', linestyle='--', linewidth=1)
        else:
            plt.axvline(x=start_lap, color='blue', linestyle='--', linewidth=1)
            plt.axvline(x=fin_lap, color='red', linestyle='--', linewidth=1)

    # plt.grid(color='w', which='major', axis='both')
    plt.ylim(88,99)
    plt.title("{} testing {} results".format(driver, session))
    sns.despine(left=True, bottom=True)
    plt.tight_layout()
    if folder not in os.listdir():
        os.mkdir(folder)
        
    # plt.show()
    print("[INFO]: file path = {}".format(f'{folder}/{file_output}'))
    plt.savefig("{}/{}".format(folder, file_output))
    plt.close()
    
def get_file_list(teams, session='day2'):
    '''
    Args:
    - teams --- dictionary of team compositions 
    - day ----- the string indicative of the day of testing or the 
                session
    '''
    files = []
    for t in teams:
        for d in teams[t]:
            if f'{d}.pdf' in os.listdir(session):
                files.append('{}/{}.pdf'.format(session, d))
            else:
                files.append('na')
    return files
    
def main():
    if len(sys.argv) < 4:
        print("[INFO]: Usage: python3 {} -round -session -folder".format(sys.argv[0]))
        exit()
    elif len(sys.argv) < 3:
        print("[INFO]: Usage: python3 {} {} -session -folder".format(sys.argv[0], sys.argv[1]))
        exit()
    elif len(sys.argv) < 2:
        print("[INFO]: Usage: python3 {} {} {} -folder".format(sys.argv[0], sys.argv[1], sys.argv[2]))
    else:
        # round and session
        round = int(sys.argv[1])
        session_num = int(sys.argv[2])
        folder = sys.argv[3]
        
        sess = fastf1.get_venet(2025, round, session_num)
        sess.load(lap=True, messages=False)
        
        if folder not in os.listdir():
            os.mkdir(folder)
        
        for driver in sess.laps['Driver'].unique():
            try:
                plot_driver(sess, driver, folder=folder, file_output=f'{driver}.pdf')
            except:
                print("[DEBUG]: driver {} caused an exception".format(driver))
        
        teams = {
            "williams": ['ALB', 'SAI'],
            "sauber": ['HUL', 'BOR'],
            "racing_bulls": ['TSU', 'HAD'],
            "haas": ['OCO', 'BEA'],
            "aston_martin": ['STR', 'ALO'],
            "alpine": ['GAS', 'DOO'],
            "mercedes": ['RUS', 'ANT'],
            "red_bull": ['VER', 'LAW'],
            "ferrari": ['LEC', 'HAM'],
            "mclaren": ['NOR', 'PIA']
        }
        
        files = get_file_list(folder)
        
        
        
