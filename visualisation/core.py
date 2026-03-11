import model.model

import matplotlib.figure
import matplotlib.axes
import matplotlib.pyplot as plt
import numpy as np

from typing import List, Optional, Union, Any, Tuple

COLOURS = ["blue", "orange", "green", "red", "purple"]
LINE_STYLES = ["-", "--", ":", "-."]


def formatter(x, pos, scale=100):
    del pos
    return str(int(x * scale))


def check_ax(
        ax: Optional[matplotlib.axes.Axes] = None,
        disable_title: bool = False) -> Tuple[matplotlib.figure.Figure | matplotlib.figure.SubFigure | None, matplotlib.axes.Axes]:
    """Check if an Axis is defined. If not, create a new subfigure.

    Args:
        ax (Optional[matplotlib.axes.Axes], optional): The axis variable to be checked. Defaults to None.
        disable_title (bool, optional): Whether the title will be disabled for this figure. Defaults to False.

    Returns:
        Tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]: A tuple containing the figure and axis objects
    """

    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 4))

    fig = ax.get_figure()

    if disable_title:
        plt.tight_layout()

    return fig, ax


def filter_for_agent(
        matrix: np.ndarray,
        agent_filter: Optional[int] = None) -> np.ndarray:
    """Filter an input matrix for the data associated with a specific agent

    Args:
        matrix (np.ndarray): The input matrix (from the DataCollector)
        agent_filter (Optional[int], optional): The index of the specified agent. Defaults to None.

    Returns:
        np.ndarray: The filtered matrix
    """

    # If needed, index data for a specific agent
    if agent_filter is not None:
        # 3D matrix
        dimensionality = len(matrix.shape)

        if dimensionality == 3:
            matrix = matrix[:, agent_filter, :]
        else:
            matrix = matrix[:, agent_filter]

    return matrix


def get_value_lists(
        data: Union[model.model.PrimingModel, Union[List[float], List[List[float]]]],
        attributes: Union[str, List[str]],
        agent_filter: Optional[int] = None) -> List[np.ndarray]:
    """Return a list of values based on a model instance or a list of values

    Args:
        data (Union[model.model.PrimingModel, List[float]]): Either a model instance or a list of values
        attributes (Union[str, List[str]]]): The names of the series to plot. Always supply, even if input data is not a model, so dimensionality of the data can be assessed.
        agent_filter (Optional[int], optional): The index of the agent you want to filter for. If not supplied, no filtering is applied. Defaults to None.

    Raises:
        ValueError: Attribute must be specified when plotting data from a model instance.

    Returns:
        List[np.ndarray]: List of model result value matrices
    """

    # Convert single string to list for uniform processing
    if isinstance(attributes, str):
        attributes = [attributes]

    if attributes is not None:
        if len(attributes) > len(LINE_STYLES):
            raise ValueError(f"Number of attributes cannot exceed number of line styles (= {len (LINE_STYLES)})")
    
    value_lists = []

    # Model data comes from the model directly
    if isinstance(data, model.model.PrimingModel):
        # This can only work if the attribute is defined
        if attributes is None:
            raise ValueError("Attribute must be specified when plotting data from a model instance.")

        df = data.datacollector.get_model_vars_dataframe()    
    
        for attribute in attributes:
            if attribute is None:
                raise ValueError("Supplied attribute cannot be None")

            value_list = np.stack(df[attribute].tolist())
            # If needed, index data for a specific agent
            value_list = filter_for_agent(value_list, agent_filter)

            value_lists.append(value_list)
    else:
        if len(data) == 0:
            raise ValueError("Supplied value list cannot have zero length")
        
        # If just a single value list is supplied, wrap in an outer list
        if len(attributes) == 1:
            _data = [ data ]
        else:
            _data = data

        # Go over each inner list and conver to numpy array
        for value_list in _data:
            # Assume a valid list of data
            value_lists.append(np.array(value_list))

    return value_lists


def check_min_max_data(
        data: Union[model.model.PrimingModel, List[float]],
        min_data: Union[List[float], List[List[float]], None],
        max_data: Union[List[float], List[List[float]], None]
    ) -> Union[Tuple[np.ndarray, np.ndarray], Tuple[None, None]]:
    """Check the supplied minimal and maximal value lists and raise errors if the supplied data does not make sense.

    Args:
        data (Union[model.model.PrimingModel, List[float]]): Either a model instance or a list of values
        min_data (Union[List[float], List[List[float]], None]): List of minimal values. Needs to be defined together with max_data.
        max_data (Union[List[float], List[List[float]], None]): List of maximal values. Needs to be defined together with min_data.

    Raises:
        ValueError: Data cannot be a model instance if min_data and max_data are defined
        ValueError: max_data cannot be defined if min_data is undefined
        ValueError: min_data cannot be defined if max_data is undefined

    Returns:
        Union[Tuple[np.ndarray, np.ndarray], Tuple[None, None]]: If the check was successful, (min_data, max_data) as numpy arrays, else (None, None)
    """

    if isinstance(data, model.model.PrimingModel) \
        and min_data is not None \
        and max_data is not None:
        raise ValueError("Supplied data cannot be a model instance if min_data and max_data are defined")

    if min_data is not None and max_data is None:
        raise ValueError("max_data cannot be None if min_data is set")
    
    if max_data is not None and min_data is None:
        raise ValueError("min_data cannot be None if max_data is set")
    
    if max_data is not None and min_data is not None:
        return np.array(min_data), np.array(max_data)
    
    return None, None

def plot_value(
        data: Union[model.model.PrimingModel, List[float]],
        attribute: str,
        ylim: Optional[List[float]] = None,
        ax: Optional[matplotlib.axes.Axes] = None,
        agent_filter: Optional[int] = None,
        min_data: Optional[List[float]] = None,
        max_data: Optional[List[float]] = None,
        title: Optional[str] = None,
        disable_title: bool = False):
    """Plot a desired series of values from a model run

    Args:
        data (Union[model.model.PrimingModel, List[float]]): Either a model instance or a list of values
        attribute (Optional[str]): The name of the series to model.
        ylim (Optional[List[float]], optional): The expected range of values for y axis. Defaults to None.
        ax (Optional[matplotlib.axes.Axes], optional): A pre-existing axis. Pass if you are building a multi-plot. Defaults to None.
        agent_filter (Optional[int], optional): The index of the agent you want to filter values for. If not supplied, no filtering is applied. Defaults to None.
        min_data (Optional[List[float]], optional): List of minimal values. Needs to be defined together with max_data.
        title (Optional[str], optional): The title for the graph. Defaults to None.
        disable_title (bool, optional): Whether to show a title for this graph.. Defaults to False.
    """

    fig, ax = check_ax(ax, disable_title)

    # Get the right data based on the supplied arguments
    value_list = get_value_lists(data, attribute, agent_filter)[0]
    # Check if min and max data are supplied correctly
    _min_data, _max_data = check_min_max_data(data, min_data, max_data)

    ax.plot(value_list, color=COLOURS[0])

    # Plot the shaded area between min and max values
    if _min_data is not None and _max_data is not None:
        ax.fill_between(
            x=range(len(value_list)),
            y1=_min_data,
            y2=_max_data,
            color=COLOURS[0],
            alpha=0.2
        )

    if ylim is not None:
        ax.set_ylim(*ylim)

    if title is not None and not disable_title:
        ax.set_title(title)


def plot_ratio(
        data: Union[model.model.PrimingModel, List[float]],
        attributes: Union[str, List[str]],
        ylim: List[float] = [0, 1],
        ax: Optional[matplotlib.axes.Axes] = None,
        agent_filter: Optional[int] = None,
        min_data: Optional[List[float]] = None,
        max_data: Optional[List[float]] = None,
        title: Optional[str] = None,
        disable_title: bool = False):
    """Plot a desired series of ratio values from a model run

    Args:
        priming_model (model.model.PrimingModel): The model instance
        attributes (Union[str, List[str]]): The names of the series to model. Always supply, even if input data is not a model, so dimensionality of the data can be assessed.
        ylim (List[float], optional): The expected range of values, will be the y axis. Defaults to [0, 1].
        ax (Optional[matplotlib.axes.Axes], optional): A pre-existing axis. Pass if you are building a multi-plot. Defaults to None.
        agent_filter (Optional[int], optional): The index of the agent you want to filter values for. If not supplied, no filtering is applied. Defaults to None.
        min_data (Optional[List[float]], optional): List of minimal values. Needs to be defined together with max_data.
        max_data (Optional[List[float]], optional): List of maximal values. Needs to be defined together with min_data.
        title (Optional[str], optional): The title for the graph. Defaults to None.
        disable_title (bool, optional): Whether to show a title for this graph. Defaults to False.

    Raises:
        ValueError: If the number of attributes to plot is larger than the supported number of line styles
    """

    if isinstance(attributes, str):
        attributes = [attributes]  # Convert single string to list for uniform processing

    # Get the right data based on the supplied arguments
    value_lists = get_value_lists(data, attributes, agent_filter)
    # Check if min and max data are supplied correctly
    _min_data, _max_data = check_min_max_data(data, min_data, max_data)

    fig, ax = check_ax(ax, disable_title)

    for attribute_idx, matrix in enumerate(value_lists):
        for i in range(matrix.shape[1]):
            ax.plot(matrix[:, i], color=COLOURS[i], linestyle=LINE_STYLES[attribute_idx])

            # Plot the shaded area between min and max values
            if _min_data is not None and _max_data is not None:
                ax.fill_between(
                    x=range(matrix.shape[0]),
                    y1=_min_data[:, i],
                    y2=_max_data[:, i],
                    color=COLOURS[i],
                    alpha=0.2
        )

    if title is not None and not disable_title:
        ax.set_title(title)
    
    ax.set_ylim(*ylim)
    ax.set_yticks(np.arange(ylim[0], ylim[1] + 0.1, 0.1))


def plot_ratio_pass(
        data: Union[model.model.PrimingModel, List[float]],
        attribute: str,
        ylim: Optional[List[float]] = None,
        baseline: Optional[float] = None,
        ax: Optional[matplotlib.axes.Axes] = None,
        title: Optional[str] = None,
        disable_title: Optional[bool] = False):
    """Plot a desired series of ratio values for all agents at once for a given model run

    Args:
        data (Union[model.model.PrimingModel, List[float]]): Either a model instance or a list of values
        attribute (Optional[str], optional): The name of the series to model.
        ylim (Optional[List[float]], optional): The expected range of values for y axis. Defaults to None.
        baseline (Optional[float], optional): The baseline to show in each subplot. Can mark a default value. Defaults to None.
        ax (Optional[matplotlib.axes.Axes], optional): A pre-existing axis. Please do not pass any axes currently. Defaults to None.
        title (Optional[str], optional): The title for the graph. Defaults to None.
        disable_title (Optional[bool], optional): Whether to show a title for this graph.. Defaults to False.

    Raises:
        ValueError: Passing an Axis through ax is currently not supported
        ValueError: Input matrix dimensions can only be 2 or 3

    Returns:
        matplotlib.axes.Axis: The created graph
    """

    # Get the right data based on the supplied arguments
    matrix = get_value_lists(data, attribute)[0]

    if ax is not None:
        raise ValueError(
            "Cannot do mosaic plots for this graph type. Please do not pass an axis."
        )
    
    # num agents = size of list ite
    num_agents = matrix[0].shape[0]
    fig, axes = plt.subplots(
        nrows=1, ncols=num_agents, figsize=(15, 10), sharey=True
    )

    num_steps = matrix.shape[0]
    time_steps = np.arange(num_steps)

    num_dimensions = len(matrix.shape)

    baseline_to_plot = None
    if baseline is not None:
        # Vertical baseline which shows 0.5
        baseline_to_plot = np.full(num_steps, baseline)

    for i, _ax in enumerate(fig.axes):
        # Plot baselines first
        if baseline_to_plot is not None:
            _ax.plot(baseline_to_plot, time_steps, color="gray",
                    alpha=0.1, linestyle="dashed")

        if num_dimensions == 3:
            _ax.plot(matrix[:, i, 0], time_steps, color="blue")
        elif num_dimensions == 2:
            _ax.plot(matrix[:, i], time_steps, color="blue")
        else:
            raise ValueError("Invalid number of dimensions")

        if ylim is not None:
            _ax.set_xlim(*ylim)
        _ax.set_title(f"{i + 1}")
        _ax.set_xticks([])
        # ax.set_xlabel('Construction 0 usage')
        _ax.grid(True)

        # Disable ugly boxes
        for spine in _ax.spines.values():
            spine.set_visible(False)

    fig.axes[0].set_ylabel("Time steps in the simulation")
    fig.axes[0].invert_yaxis()

    return ax


def check_if_none(variable_name: str, value: Any):
    if value is None:
        raise ValueError(f"\"{variable_name}\" cannot be None")