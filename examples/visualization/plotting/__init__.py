import pandas as pd
from plotly.offline import init_notebook_mode as init
from plotly.offline import iplot
import plotly.graph_objs as go


def init_notebook_mode():
    return init()

layout = dict(
    width=800,
    height=550,
    autosize=False,
    scene=dict(
        xaxis=dict(
            gridcolor='rgb(255, 255, 255)',
            zerolinecolor='rgb(255, 255, 255)',
            showbackground=True,
            backgroundcolor='rgb(230, 230,230)'
        ),
        yaxis=dict(
            gridcolor='rgb(255, 255, 255)',
            zerolinecolor='rgb(255, 255, 255)',
            showbackground=True,
            backgroundcolor='rgb(230, 230,230)'
        ),
        zaxis=dict(
            gridcolor='rgb(255, 255, 255)',
            zerolinecolor='rgb(255, 255, 255)',
            showbackground=True,
            backgroundcolor='rgb(230, 230,230)'
        ),
        aspectratio=dict(x=1, y=1, z=0.7),
        aspectmode='manual'
    ),
)

def plot(df, column_name):
    colors = ['rgb(228,26,28)','rgb(55,126,184)','rgb(77,175,74)']
    data = []
    for i in range(len(df[column_name].unique())):
        name = df[column_name].unique()[i]
        color = colors[i]
        x = df[df[column_name] == name]['SepalLength']
        y = df[df[column_name] == name]['SepalWidth']
        z = df[df[column_name] == name]['PetalLength']

        trace = dict(
            name = name,
            x = x, y = y, z = z,
            type = "scatter3d",
            mode = 'markers',
            marker = dict(size=3, color=color, line=dict(width=0)),
            text = \
                'Cluster0: ' + \
                df[df[column_name] == name]['Cluster0'].astype('str') + \
                '<br>Cluster1: ' + \
                df[df[column_name] == name]['Cluster1'].astype('str') + \
                '<br>Cluster2: ' + \
                df[df[column_name] == name]['Cluster2'].astype('str'),
            textposition = "top"
        )
        data.append(trace)

        cluster = dict(
            color = color,
            opacity = 0.3,
            type = "mesh3d",
            x = x, y = y, z = z
        )
        data.append(cluster)
    fig = dict(data=data, layout=layout)
    return iplot(fig)