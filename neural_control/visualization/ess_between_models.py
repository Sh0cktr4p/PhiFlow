import plotly.io as pio
from collections import defaultdict
import json
import plotly.graph_objects as go
import plotly.express as px

import numpy as np
import pandas as pd

# Root folder in which all models folders are
root = r"/home/trost/guided_research/PhiFlow/neural_control/storage/networks/"

# Test to be loaded
tests = [1]

# Models indexes that will be considered
models = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20]
iterations = [m * 25 for m in models]

# Folders of the models
folders = [
    'g_2',
    'g_3',
    'g_4',
    'l_1',
    'l_2',
    'l_3',
    'l_4',
    'skippy',
    'skippy_2',
    'skippy_3',
    #'love_fixed_lr_4',
    #'love_fixed_lr_5',
    #'love_lr_decay_long',
    #'love_lr_decay_5',
    #'love_lr_decay_6',
]

ss_i = 0.25

# Labels that will be used in the plot
# Key should be model folder and value should be the label
labels = dict(
    g_2="g 1",
    g_3="g 2",
    g_4="g 3",
    l_1="love 1",
    l_2="love 2",
    l_3="love 3",
    l_4="love 4",
    skippy="every 4th",
    skippy_2="every 8th 1",
    skippy_3="every 8th 2",
    #love_fixed_lr_4='fixed_2',
    #love_fixed_lr_5='fixed_3',
    #love_lr_decay_4='decay_1',
    #love_lr_decay_5='decay_2',
    #love_lr_decay_long='decay',
)
offsets = np.linspace(0, 1, len(folders) * len(models))
offsets -= 0.5
offsets *= 0.25
bar_width = offsets[1] - offsets[0]
offsets = iter(-offsets)

export_values = defaultdict(dict)

df = pd.DataFrame(columns=['test', 'model', 'folder', 'error_xy', 'std_xy', 'mean_xy'])

for folder_index, folder in enumerate(folders):
    for test in tests:
        for model in models:
            # Load metrics
            print(f'\n Loading {folder}/test{test}_{model}')
            try:
                with open(f'{root}/{folder}/tests/test{test}_{model}/metrics.json', 'r') as f:
                    data = json.load(f)
            except FileNotFoundError:
                print(f'Could not find metrics.json \n')
                continue
            # Calculate steady state error
            print(data.keys())
            ss_error_xy = np.array(data['ss_error_xy'])
            std_xy = np.array(data['ss_error_xy_stdd'])
            # Add to dataframe
            df = df.append({
                'test': test,
                # 'test_label': test_labels[test],
                'model': model,
                'folder': folder,
                'std_xy': std_xy,
                'ss_error_xy': ss_error_xy,
                # 'n_i': n_i[folder],
                'label': labels[folder],
                'training_type': 'online' if 'online' in folder else 'supervised'
            }, ignore_index=True)


# def get_colors(labels):


go.Figure()
# Bar plot with y axis being the mean error and x axis the entries in df. Add error bars as well
fig = px.bar(
    df,
    x="model",
    y="ss_error_xy",
    color="label",
    barmode="group",
    error_y='std_xy',
    # pattern_shape='training_type',
    # pattern_shape_sequence=['', '+'],
    # color_discrete_map=get_colors(df['label']),
    # color_discrete_map={
    #     'Diff0.5k_seed100': ' #cccccc',
    #     'Diff1k_seed100': '#b3b3b3',
    #     'Diff1k': '#707070',
    #     'Diff2k_seed100': '#000000',
    # }
)
fig.update_layout(bargap=0.2, bargroupgap=0.0)  # Make bars closer to each other

# Put legend inside fig
fig.update_layout(legend=dict(
    yanchor="top",
    y=1,
    xanchor="right",
    x=1
))

# Change grid pattern
fig.update_layout(xaxis=dict(showgrid=True),
                  yaxis=dict(showgrid=True)
                  )

# Set every xticks
print([m * 25 for m in models])
fig.update_xaxes(tickmode="array", tickvals=models, ticktext=iterations)
# Set titles
fig.update_layout(xaxis_title=r"number of iterations (x1000)", yaxis_title=r"$\overline{\|e_{xy}\|}_{ss}$")
# Hide legends title
fig.update_layout(legend_title_text="")


# Export to pdf
go.Figure().write_image(f'/home/trost/guided_research/PhiFlow/neural_control/storage/figures/skippy.pdf')
fig.show()
fig.write_image(f'/home/trost/guided_research/PhiFlow/neural_control/storage/figures/skippy.pdf')
print('Done')
