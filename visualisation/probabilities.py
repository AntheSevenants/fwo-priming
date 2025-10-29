import matplotlib.pyplot as plt
import numpy as np

COLOURS = [ "blue", "orange", "green", "red", "purple" ]

def plot_ratio(model, attribute, ax=None, title=None, disable_title=False):
    df = model.datacollector.get_model_vars_dataframe()

    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 4))
    
    fig = ax.get_figure()
    
    if disable_title:
        fig.tight_layout()

    matrix = np.stack(df[attribute])
    for i in range(matrix.shape[1]):
        ax.plot(matrix[:,i], color=COLOURS[i])

    # if title is not None and not disable_title:
    #     ax.set_title(title)