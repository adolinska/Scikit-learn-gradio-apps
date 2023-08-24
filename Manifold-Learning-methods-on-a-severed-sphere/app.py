# Author: Jaques Grobler <jaques.grobler@inria.fr>
# License: BSD 3 clause

from time import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter
from sklearn import manifold
from sklearn.utils import check_random_state
import plotly.graph_objects as go
import gradio as gr
from matplotlib import style
plt.switch_backend("agg")
style.use('ggplot')

n_neighbors = 10
n_samples = 1000

font1 = {'family':'DejaVu Sans','size':10, 'color':'white'}

def sphere(n_neighbors, n_samples):

    # Create our sphere.
    random_state = check_random_state(0)
    p = random_state.rand(n_samples) * (2 * np.pi - 0.55)
    t = random_state.rand(n_samples) * np.pi

    # Sever the poles from the sphere.
    indices = (t < (np.pi - (np.pi / 8))) & (t > ((np.pi / 8)))
    colors = p[indices]
    x, y, z = (
        np.sin(t[indices]) * np.cos(p[indices]),
        np.sin(t[indices]) * np.sin(p[indices]),
        np.cos(t[indices]),
    )
    
    sphere_data = np.array([x, y, z]).T
    
    return x, y, z, colors, sphere_data

x, y, z, colors, sphere_data = sphere(n_neighbors, n_samples)

def create_3D_plot(n_neighbors = n_neighbors, n_samples = n_samples):

    x, y, z, colors = sphere(n_neighbors, n_samples)[:4]

    # Create the trace for the scatter plot
    scatter_trace = go.Scatter3d(
        x=x,
        y=y,
        z=z,
        mode='markers',
        marker=dict(
            size=5,
            color=colors,
            colorscale='rainbow',
            showscale=False
        )
    )

    # Create the figure and add the trace
    fig = go.Figure()
    fig.add_trace(scatter_trace)
    
    return fig

# Perform Locally Linear Embedding Manifold learning
methods = {"LLE":"standard", "LTSA":"ltsa" ,
           'Hessian LLE':'hessian', "Modified LLE":"modified"}

available = ["LLE", "LTSA",'Hessian LLE',"Modified LLE",
            "Isomap","MDS","Spectral Embedding", "t-SNE"]

def make_plot(method, methods = methods):
    
    # Plot our dataset.
    fig1 = plt.figure(figsize=(10, 6), facecolor = 'none', dpi = 200)
    plt.title(
        "Manifold Learning with %i points, %i neighbors" % (1000, n_neighbors), 
        pad = 20, bbox=dict(boxstyle="round,pad=0.3",color = "#6366F1"),
        fontdict = font1, size = 16
    )
    
    if method in methods.keys():
        t0 = time()
        trans_data = (
            manifold.LocallyLinearEmbedding(
                n_neighbors=n_neighbors, n_components=2, method=methods[method]
            )
            .fit_transform(sphere_data)
            .T
        )
        t1 = time()
        title = "%s: %.2g sec" % (method, t1 - t0)

    elif method == "Isomap":
        # Perform Isomap Manifold learning.
        t0 = time()
        trans_data = (
            manifold.Isomap(n_neighbors=n_neighbors, n_components=2)
            .fit_transform(sphere_data)
            .T
        )
        t1 = time()
        title = "%s: %.2g sec" % ("ISO", t1 - t0)

    elif method == "MDS":
        # Perform Multi-dimensional scaling.
        t0 = time()
        mds = manifold.MDS(2, max_iter=100, n_init=1, normalized_stress="auto")
        trans_data = mds.fit_transform(sphere_data).T
        t1 = time()
        title = "MDS: %.2g sec" % (t1 - t0)

    elif method == "Spectral Embedding":
        # Perform Spectral Embedding.
        t0 = time()
        se = manifold.SpectralEmbedding(n_components=2, n_neighbors=n_neighbors)
        trans_data = se.fit_transform(sphere_data).T
        t1 = time()
        title = "Spectral Embedding: %.2g sec" % (t1 - t0)

    elif method == "t-SNE":    
        # Perform t-distributed stochastic neighbor embedding.
        t0 = time()
        tsne = manifold.TSNE(n_components=2, random_state=0)
        trans_data = tsne.fit_transform(sphere_data).T
        t1 = time()
        title = "t-SNE: %.2g sec" % (t1 - t0)
    
    ax = fig1.add_subplot()
    ax.scatter(trans_data[0], trans_data[1], c=colors, cmap=plt.cm.rainbow)

    ax.tick_params(axis='x', which='both', bottom=False, labelbottom=False)  # Hide x-axis tick labels
    ax.tick_params(axis='y', which='both', left=False, labelleft=False)  # Hide y-axis tick labels
    
    return fig1, title

made ="""<div style="text-align: center;">
  <p>Made with ❤</p>"""

link = """<div style="text-align: center;">
<a href="https://scikit-learn.org/stable/auto_examples/manifold/plot_manifold_sphere.html#sphx-glr-auto-examples-manifold-plot-manifold-sphere-py">
Demo is based on this script from scikit-learn documentation</a>"""

intro = """<h1 style="text-align: center;">🤗 Manifold Learning methods on a severed sphere 🤗</h1>
"""
desc = """<h3 style="text-align: left;"> An application of the different <a href="https://scikit-learn.org/stable/modules/manifold.html#manifold">
Manifold Learning</a> techniques on a spherical data-set. Here one can see the use of dimensionality reduction in order to gain some intuition regarding the manifold learning methods. Regarding the dataset, the poles are cut from the sphere, as well as a thin slice down its side. This enables the manifold learning techniques to ‘spread it open’ whilst projecting it onto two dimensions.
<br><br>
For a similar example, where the methods are applied to the S-curve dataset, see <a href="https://scikit-learn.org/stable/auto_examples/manifold/plot_manifold_sphere.html#sphx-glr-auto-examples-manifold-plot-manifold-sphere-py">
Comparison of Manifold Learning methods</a>.
<br><br>
Note that the purpose of the <a href="https://scikit-learn.org/stable/modules/manifold.html#multidimensional-scaling">
MDS</a> is to find a low-dimensional representation of the data (here 2D) in which the distances respect well the distances in the original high-dimensional space, unlike other manifold-learning algorithms, it does not seeks an isotropic representation of the data in the low-dimensional space. Here the manifold problem matches fairly that of representing a flat map of the Earth, as with <a href="https://en.wikipedia.org/wiki/Map_projection">
map projection</a>.
</h3>
"""

with gr.Blocks(theme = gr.themes.Soft(
    primary_hue="amber",
    secondary_hue="orange",
    neutral_hue="teal",
    font=[gr.themes.GoogleFont('Inter'), 'ui-sans-serif', 'system-ui', 'sans-serif'],), title = "Manifold Learning methods on a severed sphere") as demo:
    with gr.Column():
        gr.HTML(intro)
        with gr.Accordion(label = "Description", open = True):
            gr.HTML(desc)
        with gr.Column():
            method = gr.Radio(available, label="Select method:", value= "LLE")
            title = gr.Textbox(label = 'Time for the method to perform:')
            with gr.Row():
                plot_3D = gr.Plot(label="3D projection of the sphere")
                plot = gr.Plot(label="Plot")
        
        method.change(fn=make_plot, inputs = method, outputs=[plot, title])
        
        demo.load(fn=make_plot, inputs = method, outputs=[plot, title])
        demo.load(fn=create_3D_plot, inputs = [], outputs=plot_3D)
        gr.HTML(made)
        gr.HTML(link)

demo.launch()

