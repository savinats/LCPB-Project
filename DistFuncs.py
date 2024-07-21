# -*- coding: utf-8 -*-
"""
Created on Fri Jul  5 15:22:42 2024

@author: lm-schulze
"""

# library imports
import transformers
import torch
import sklearn.manifold
import sklearn.metrics
import scipy
import numpy as np
import matplotlib.pyplot as plt
import warnings
from matplotlib.animation import FuncAnimation
from matplotlib.colors import LogNorm, Normalize
from IPython.display import HTML

# function to calculate the distances
def dist_matr(output, type='euclid'):
    """
    Calculate pairwise distances between elements in a 2D or 3D array using Euclidean distance or cosine similarity.

    Parameters:
    -----------
    output : numpy.ndarray
        Input array containing the model output data points. It must be either a 2D array of shape (b, c) or a 3D array of shape (a, b, c).
    
    type : str, optional (default='euclid')
        Type of distance metric to use. Must be one of:
        - 'euclid' : Use Euclidean distance.
        - 'cosine' : Use cosine similarity.
    
    Returns:
    --------
    numpy.ndarray
        Matrix containing the pairwise distances or similarities.
        - If the input is a 2D array of shape (b, c), the output will be a 2D array of shape (b, b).
        - If the input is a 3D array of shape (a, b, c), the output will be a 3D array of shape (a, b, b).
    
    """
    if type == 'euclid':
        if output.ndim == 2:
            # The input is a 2D array of shape (b, c)
            mat = np.linalg.norm(output[:, np.newaxis, :] - output[np.newaxis, :, :], axis=2)
        elif output.ndim == 3:
            # The input is a 3D array of shape (a, b, c)
            a, b, c = output.shape
            mat = np.zeros((a, b, b))
            for i in range(a):
                mat[i] = scipy.spatial.distance.squareform(scipy.spatial.distance.pdist(output[i], 'euclidean'))
        else:
            raise ValueError("Input array must be either 2D or 3D")
    elif type == 'cosine':
        if output.ndim == 2:
            # The input is a 2D array of shape (b, c)
            mat = sklearn.metrics.pairwise.cosine_distances(output)
        elif output.ndim == 3:
            # The input is a 3D array of shape (a, b, c)
            a, b, c = output.shape
            mat = np.zeros((a, b, b))
            for i in range(a):
                mat[i, :, :] = sklearn.metrics.pairwise.cosine_distances(output[i])
        else:
            raise ValueError("Input array must be either 2D or 3D")
    else:
        raise ValueError("Type must be 'euclid' or 'cosine'")
    return mat

# function to get the module outputs
def get_module_outputs(model, tokenizer, prompt, getInputs=False):
    """
    Extract the inputs and outputs of each module in a model given an input prompt by attaching forward
    hooks to all model modules and processing the input prompt. Returns separate lists for inputs and
    outputs if getInputs = True.

    Parameters:
    --------
    model : torch.nn.Module
        Model from which to get the module outputs.
    
    tokenizer : transformers.PreTrainedTokenizer 
        Tokenizer to encode the input prompt.
    
    prompt : str 
        Input text to be processed by the model.

    getInput : bool, optional (default=False)
        Determines if module inputs are also extracted and returned.

    Returns:
    --------
    Tupel
        List[np.ndarray]: List containing the output of each module as numpy arrays.
        List[str]: List of module names as str.

    or, if getInput=True:
        List[np.ndarray]: List containing the output of each module as numpy arrays.
        List[np.ndarray]: List containing the input of each module as numpy arrays.
        List[str]: List of module names as str.
        
    """
    
    # use a hook to get the output as numpy array
    outputs = []
    inputs = []
    modules = []

    def hook(module, input, output):
        outputs.append(output[0].detach().numpy().squeeze())
        modules.append(str(module))
        if getInputs:
            inputs.append(input[0].detach().numpy().squeeze())    
    
    # Attaching hook to all modules
    hook_handles = []
    for module in model.modules():
        handle=module.register_forward_hook(hook)
        hook_handles.append(handle)

    input_ids = tokenizer.encode(prompt, return_tensors="pt")

    # Pass the input through the model (this triggers the hooks)
    with torch.no_grad():
        model(input_ids)
    
    # Remove all hooks
    for handle in hook_handles:
        handle.remove()

    if getInputs:
        return(outputs, inputs, modules)
    else:
        return(outputs, modules)

def get_module_data(model, tokenizer, prompt, getInputs=False):
    """
    Extract the inputs and outputs of each module in a model given an input prompt by attaching forward
    hooks to all model modules and processing the input prompt. Returns list containing both inputs and
    outputs if getInputs = True.

    Parameters:
    --------
    model : torch.nn.Module
        Model from which to get the module outputs.
    
    tokenizer : transformers.PreTrainedTokenizer 
        Tokenizer to encode the input prompt.
    
    prompt : str 
        Input text to be processed by the model.

    getInput : bool, optional (default=False)
        Determines if module inputs are also extracted and returned.

    Returns:
    --------
    Tupel
        List[np.ndarray]: List containing the output of each module as numpy arrays.
        List[str]: List of module names as str.

    or, if getInput=True:
        List[np.ndarray]: List containing the output of each module as numpy arrays.
        List[np.ndarray]: List containing the input of each module as numpy arrays.
        List[str]: List of module names as str.
        
    """
    
    # use a hook to get the data as numpy array
    data = []
    modules = []

    def hook(module, input, output):
        if getInputs:
            data.append(input[0].detach().numpy().squeeze())    
            modules.append(str(module))
            
        data.append(output[0].detach().numpy().squeeze())
        modules.append(str(module))
        
    
    # Attaching hook to all modules
    hook_handles = []
    for module in model.modules():
        handle=module.register_forward_hook(hook)
        hook_handles.append(handle)

    input_ids = tokenizer.encode(prompt, return_tensors="pt")

    # Pass the input through the model (this triggers the hooks)
    with torch.no_grad():
        model(input_ids)
    
    # Remove all hooks
    for handle in hook_handles:
        handle.remove()
        
    return(data, modules)

def get_module_names(model, tokenizer):
    """
    Extract the names of the module modules in the order of input processing.

    Parameters:
    --------
    model : torch.nn.Module
        Model from which to get the module outputs.
    
    tokenizer : transformers.PreTrainedTokenizer 
        Tokenizer to encode the input prompt.

    Returns:
    --------
    List[np.ndarray]
        List containing the name of each module in the model as str.
        
    """
    
    # use a hook to get the module names as str
    modules = []
    def hook(module, input, output):
        modules.append(str(module))
    
    
    # Attaching hook to all modules
    hook_handles = []
    for module in model.modules():
        handle=module.register_forward_hook(hook)
        hook_handles.append(handle)

    input_ids = tokenizer.encode(" ", return_tensors="pt")

    # Pass the input through the model (this triggers the hooks)
    with torch.no_grad():
        model(input_ids)
    
    # Remove all hooks
    for handle in hook_handles:
        handle.remove()

    return(modules)


def filter_by_shape(outputs, module_names, ref_shape=None, returnIndex=False):
    """
    Filters the outputs and module names based on a reference shape.

    Parameters:
    --------
    outputs: list
        List of output tensors from different modules.

    module_names: list
        List of module names corresponding to the outputs.

    ref_shape: tuple, optional (default=None)
        The reference shape to filter by. If None, the shape of the first output is used.

    returnIndex: bool, optional (default=False)
        Whether to return list of original indices of filtered outputs.

    Returns:
    --------

    tuple: 
        A tuple containing:
        - filtered_outputs (np.array): Outputs that match the reference shape.
        - filtered_module_names (np.array): Module names corresponding to the filtered outputs.
        - idx (np.array, optional): Indices of the modules where outputs match the reference shape.

    """
    if ref_shape is None:
        ref_shape = outputs[0].shape

    # get indices of modules where outputs have the right shape
    idx = np.array([i for i, o in enumerate(outputs) if o.shape == ref_shape])

    # filter to get the outputs & module names at idx
    filtered_outputs = np.array([outputs[i] for i in idx])
    filtered_module_names = np.array([str(module_names[i]).split('(', 1)[0].strip() for i in idx])

    if returnIndex:
        return filtered_outputs, filtered_module_names, idx

    else:
        return filtered_outputs, filtered_module_names

def remove_module_type(outputs, modules, type="Dropout", idx=None):
    """
    Removes modules of a specific type from the outputs and modules.

    Parameters:
    --------
    outputs: list
        List of outputs from different modules.

    modules: list
        List of module names corresponding to outputs.

    type: str or list
        The type(s) of modules to remove. Can be a single string or a list of strings.

    idx: list, optional (default=None)
        Indices of the original modules. If provided, the indices of the remaining modules are returned.

    Returns:
    --------
    tuple: 
        A tuple containing:
        - filtered_outputs (np.array): Outputs with specified modules removed.
        - filtered_modules (np.array): Module names with specified modules removed.
        - filtered_idx (np.array, optional): Updated indices if `idx` is provided.
    """
    if isinstance(type, str):
        type = [type]
    
    # get indices of modules that do not match the specified types
    idr = np.array([i for i, m in enumerate(modules) if all(str(m)[:len(t)] != t for t in type)])
    
    # filter to get the outputs at idx
    filtered_outputs = np.array([outputs[i] for i in idr])
    filtered_modules = np.array([str(modules[i]).split('(', 1)[0].strip() for i in idr])

    if idx is not None:
        filtered_idx = np.array([idx[i] for i in idr])
        return filtered_outputs, filtered_modules, filtered_idx

    return filtered_outputs, filtered_modules


def filter_duplicates(data, mods, idx=None):
    """
    Filters out duplicate inputs/outputs from the data list, keeps the first instance of each.
    Removes corresponding elements from mods (and idx, if given).

    Parameters:
    --------
    data : list
        List of numpy arrays representing inputs/outputs of modules.

    mods: list
        List of module names, same length as data.

    idx: list, optional (default=None)
        List of module indices.

    Returns:
    --------
        tuple:
            A tuple containing:
            - filtered_data (list): data list without duplicates.
            - filtered_mods (list): Filtered module names corresponding to filtered data.
            - filtered_idx (list, optional): Filtered module indices.
    """
    seen = set()
    filtered_data = []
    filtered_mods = []
    filtered_idx = [] if idx is not None else None

    for i, (d, m) in enumerate(zip(data, mods)):
        # Create a hashable identifier for the array
        array_hash = (d.shape, d.tobytes())
        if array_hash not in seen:
            seen.add(array_hash)
            filtered_data.append(d)
            filtered_mods.append(m)
            if idx is not None:
                filtered_idx.append(idx[i])

    if idx is not None:
        return filtered_data, filtered_mods, filtered_idx

    return filtered_data, filtered_mods


def read_prompt(file_path):
    """
    Read text prompt from a file and remove newline characters.

    Parameters:
    --------
    file_path : str
        Path to the file containing the prompt.

    Returns:
    --------
    str 
        File content as a single string.
    """
    with open(file_path, 'r') as file:
        return file.read().replace('\n', '')

def get_last_token_distance(data, modules, ref_shape=None, returnIndex=False):
    """
    Calculate cosine distances between the last token and all other tokens in module outputs.
    Filters modules where the outputs have the expected shape (Nr. of tokens, embedding dimension).

    Parameters:
    --------
    data : List[np.ndarray]
        List of module inputs/outputs as numpy arrays.
    modules : List[str]
        List of module names as strings.
    ref_shape : Tuple, optional(default=None)
        Shape by which the outputs should be filtered prior to calculating the distances.
        If None, use unfiltered outputs.
    returnIndex: bool, optional (default=False)
        Whether to return list of original indices of filtered outputs.

    Returns:
    --------
    tuple:
        np.ndarray: Distances from the last token for each module.
        np.ndarray: Names of the modules corresponding to the indices.
        np.ndarray: Original indices of filtered outputs (optional)

    """
    # filter given data by shape
    if ref_shape is not None:
        filtered_data, filtered_mods, idx = filter_by_shape(data, modules, ref_shape, returnIndex=True)
    else: 
        filtered_data, filtered_mods = data, modules
        idx = np.arange(len(data))
        

    # calculate (cosine) distances
    distances_cos = dist_matr(filtered_data, type="cosine")
    # extract distances to the last token
    dist_from_token = distances_cos[:, -1, :]

    if returnIndex:
        return dist_from_token, filtered_mods, idx
    else:
        return dist_from_token, filtered_mods

def plot_kde(dist_from_token, module_descriptor, labels, filename="dist_kde", fix_lims=True):
    """
    Plot the Kernel Density Estimate (KDE) of token distance distributions.

    Plots the KDE of the cosine distance distribution of the last token
    for each module's output. Multiple KDE plots can be overlaid with different labels.

    Parameters:
    --------
    dist_from_token : np.ndarray
        Distances from the last token for each module.
        
    module_descriptor : np.ndarray
        Unique identifiers of the modules corresponding to the data.
        Used in figure titles and filenames.
        
    labels : List[str]
        Labels for the different distributions to plot.

    filename : str, optional (default="dist_kde")
        Base file name under which the plots are saved.

    fix_lims: bool, optional (default=True)
        Determines if x-axis limits are fixed to (0, 2).
        If False, axis limits are determined by dist_from_token.

    Returns:
    --------
    None
    """
    for n, mod in enumerate(module_descriptor):
        fig, ax = plt.subplots(figsize=(8, 3))

        for dist, label in zip(dist_from_token, labels):
            d = dist[n]
            kde = scipy.stats.gaussian_kde(d)
            if fix_lims:
                x_vals = np.linspace(0, 2, 1000)
            else: 
                x_vals = np.linspace(d.min(), d.max(), 1000)
                
            kde_vals = kde(x_vals)

            ax.plot(x_vals, kde_vals, lw=2, linestyle="-", label=label)
        
        ax.set_xlabel("Token distance (cosine)")
        ax.set_ylabel("Density")
        ax.set_title(f"KDE of Token distance distribution after mod. {mod}")
        ax.grid()
        ax.legend(loc='best')

        # save each figure as png under specified name + module identifier
        fig.tight_layout()
        plt.savefig(f"{filename}_{'_'.join(mod.split())}.png", bbox_inches = "tight")
        
        plt.show()


def plot_dist_heatmap(distances, labels, module_idx, scaling="SharedLin", filename="distances"): 
    """
    Plot the distances as heatmaps for each module output and save each as a separate file.

    Parameters:
    -----------
    distances : numpy.ndarray
        3D array of shape (num_modules, num_tokens, num_tokens) containing the pairwise distances 
        between tokens for each module.
    
    labels : list of str
        List of token labels for the heatmap axes.
    
    module_idx : list of str
        List of module indices or names to use as titles for each subplot.
    
    scaling : str, optional (default="SharedLin")
        The scaling method to use for the color maps. Must be one of:
        - "SharedLin" : All plots share the same linear color scale.
        - "SharedLog" : All plots share the same logarithmic color scale.
        - "IndividualLin" : Each plot has its own linear color scale.
        - "IndividualLog" : Each plot has its own logarithmic color scale.

    filename : str, optional (default="distances")
        Base file name under which the plots are saved.

    """

    num_plots = distances.shape[0] # one plot for each module

    if scaling == "SharedLog":  # all plots share the same logarithmic color scale
        # raise warning if log scale is used while negative values are present
        if np.min(distances) < 0:
            warnings.warn("Attempting to use log scale with negative values. All values <= 0 will be outside of color scale range.")
        # get min/max for color scale
        global_min = np.min(distances[distances > 0])
        global_max = np.max(distances)
    
    elif scaling == "SharedLin":  # all plots share the same linear color scale
        # get min/max for color scale
        global_min = np.min(distances)
        global_max = np.max(distances)
    
    for i in range(num_plots): 
        print(f"Creating plot {i+1}/{num_plots}. Module {module_idx[i]}")

        # create figure & axis
        fig, ax = plt.subplots(figsize=(8, 8))
        
        if scaling == "SharedLog":
            im = ax.imshow(distances[i], cmap='viridis', origin='lower', norm=LogNorm(vmin=global_min, vmax=global_max))
        elif scaling == "SharedLin":
            im = ax.imshow(distances[i], cmap='viridis', origin='lower', vmin=global_min, vmax=global_max)
        elif scaling == "IndividualLin":
            im = ax.imshow(distances[i], cmap='viridis', origin='lower')
        elif scaling == "IndividualLog":
            if np.min(distances[i]) < 0:
                # raise warning if log scale is used while negative values are present
                warnings.warn("Attempting to use log scale with negative values. All values <= 0 will be outside of color scale range.")
            # get min/max for log scaling
            local_min = np.min(distances[i][distances[i] > 0])
            local_max = np.max(distances[i])
            im = ax.imshow(distances[i], cmap='viridis', origin='lower', norm=LogNorm(vmin=local_min, vmax=local_max))
        else:
            raise ValueError("Scaling must be 'SharedLin', 'SharedLog', 'IndividualLin' or 'IndividualLog'")

        ax.set_title(f"Token distances after Module {module_idx[i]}")
        ax.set_xticks(np.arange(len(labels)))
        ax.set_yticks(np.arange(len(labels)))
        ax.set_xticklabels(labels, rotation=90)
        ax.set_yticklabels(labels)
        fig.colorbar(im, ax=ax, orientation='vertical', shrink=0.7)
        fig.tight_layout()

        # save each figure as png under specified name + module identifier
        plt.savefig(f"{filename}_mod_{'_'.join(module_idx[i].split())}.png", bbox_inches = "tight")
        plt.close(fig)

# we can also make a gif instead :D
def make_distances_gif(distances, labels, module_idx, filename="distances.gif", useLogScale=True):

    """
    Create an animated GIF of the distances heatmap for each module output.

    Parameters:
    -----------
    distances : numpy.ndarray
        A 3D array of shape (num_modules, num_tokens, num_tokens) containing the pairwise distances 
        between tokens for each module.
    
    labels : list of str
        A list of token labels for the heatmap axes.
    
    module_idx : list of str
        A list of module indices or names to use as titles for each frame in the GIF.
    
    filename : str, optional (default="distances.gif")
        The name of the output GIF file.
    
    useLogScale : bool, optional (default=True)
        Whether to use a logarithmic color scale for the heatmaps.

    Returns:
    --------
    matplotlib.animation.FuncAnimation
        The animation object representing the GIF.
    """

    # Find global maximum values 
    global_max = np.max(distances)
    
    # Create a figure and axis objects
    fig, ax = plt.subplots(figsize=(10, 8))

    if useLogScale:        
        # raise warning if log scale is used while negative values are present
        if np.min(distances)<0:
            warnings.warn("Attempting to use log scale with negative values. All values <= 0 will be outside of color scale range.")
            
        # Find global minimum (avoiding zeros for log scale)
        global_min = np.min(distances[distances > 0])
        # Initialize the image object
        im = ax.imshow(np.zeros(distances[0].shape), cmap='viridis', origin='lower', norm=LogNorm(vmin=global_min, vmax=global_max))
        
        # Create colorbar
        cbar = fig.colorbar(plt.cm.ScalarMappable(norm=LogNorm(vmin=global_min, vmax=global_max), cmap='viridis'),
                            ax=ax, orientation='vertical')
    else:
        # Find global minimum values
        global_min = np.min(distances)
         # Initialize the image object
        im = ax.imshow(np.zeros(distances[0].shape), cmap='viridis', origin='lower', vmin=global_min, vmax=global_max)
        
        # Create colorbar
        cbar = fig.colorbar(plt.cm.ScalarMappable(norm=Normalize(0, global_max), cmap='viridis'),
                            ax=ax, orientation='vertical')
    
    # Function to update the plot for each frame
    def update(frame):
        im.set_array(distances[frame])
        ax.set_title(f"Token distances after Module: {module_idx[frame]}")
        ax.set_xticks(np.arange(len(labels)))
        ax.set_yticks(np.arange(len(labels)))
        ax.set_xticklabels(labels, rotation=90)
        ax.set_yticklabels(labels)
        return [im]
    
    # Create animation
    ani = FuncAnimation(fig, update, frames=distances.shape[0], interval=400, blit=True, repeat=True, repeat_delay=500) 
    ani.save(filename=filename, writer="pillow")
    HTML(ani.to_jshtml())        
    return(ani)