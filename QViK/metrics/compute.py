import pandas as pd

runs = pd.read_csv('logs/runs.csv')
seconds = runs['time'].sum()
hours, remainder = divmod(seconds, 3600)
minutes, seconds = divmod(remainder, 60)
print('Experiments consumed {:02}:{:02}:{:02} CPU time'.format(int(hours), int(minutes), int(seconds)))