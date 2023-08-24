# Authors: Jona Sassenhagen
# License: BSD 3 clause

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import FactorAnalysis, PCA
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_iris
import gradio as gr
from matplotlib import style
plt.switch_backend("agg")
style.use('ggplot')

font1 = {'family':'DejaVu Sans','color':'#2563EB','size': 14}

#load and transform the data
data = load_iris()
X = StandardScaler().fit_transform(data["data"])
feature_names = data["feature_names"]

methods = {
    "PCA": PCA(),
    "Unrotated FA": FactorAnalysis(),
    "Varimax FA": FactorAnalysis(rotation="varimax")
}

def factor_analysis(method):
    #figure1
    fig1, ax = plt.subplots(figsize=(10, 6), facecolor='none', dpi = 200)
    im = ax.imshow(np.corrcoef(X.T), cmap="Spectral", vmin=-1, vmax=1)

    ax.set_xticks([0, 1, 2, 3])
    ax.set_xticklabels(list(feature_names), 
                       rotation=90, fontdict = font1)
    ax.set_yticks([0, 1, 2, 3])
    ax.set_yticklabels(list(feature_names), fontdict = font1)
    plt.grid(False)
    plt.colorbar(im).ax.tick_params()
    ax.set_title("Iris feature correlation matrix", 
                 fontdict=font1, size = 18, 
                 color = "white", pad = 20,
                 bbox=dict(boxstyle="round,pad=0.3", 
                           color = "#2563EB"))
    plt.tight_layout()
    plt.close('all')

    n_comps = 2

    #figure2
    fig2, axs = plt.subplots(figsize=(8, 5), facecolor='none', dpi = 200)
    plt.grid(False)
    fa = methods[method]
    fa.set_params(n_components=n_comps)
    fa.fit(X)

    components = fa.components_

    vmax = np.abs(components).max()
    axs.imshow(components, cmap="Spectral", vmax=vmax, vmin=-vmax)
    axs.set_xticks(np.arange(len(feature_names)))
    axs.set_xticklabels(feature_names, fontdict=font1)
    axs.set_title(method, 
                  fontdict=font1, size = 18, 
                  color = "white", pad = 20,
                  bbox=dict(boxstyle="round,pad=0.3", 
                           color = "#2563EB"))
    axs.set_yticks([0, 1])
    axs.set_yticklabels(["Comp. 1", "Comp. 2"], fontdict=font1)
    
    plt.tight_layout()
    plt.close('all')
    
    return fig1, fig2, components

intro = """<h1 style="text-align: center;">🤗 <strong>Factor Analysis (with rotation) to visualize patterns</strong> 🤗</h1>
"""
desc = """<h3 style="text-align: left;"> Investigating the Iris dataset, we see that sepal length, petal length and petal width are highly correlated. 
Sepal width is less redundant. Matrix decomposition techniques can uncover these latent patterns. 
<br><br>Applying rotations to the resulting components does not inherently improve the predictive value of the derived latent space,
but can help visualise their structure; here, for example, the varimax rotation, 
which is found by maximizing the squared variances of the weights, 
finds a structure where the second component only loads positively on sepal width.
<br></h3>
"""

made ="""<div style="text-align: center;">
  <p>Made with ❤</p>"""

link = """<div style="text-align: center;">
<a href="https://scikit-learn.org/stable/auto_examples/decomposition/plot_varimax_fa.html#sphx-glr-auto-examples-decomposition-plot-varimax-fa-py" target="_blank" rel="noopener noreferrer">
Demo is based on this script from scikit-learn documentation</a>"""

with gr.Blocks(theme=gr.themes.Soft(primary_hue="blue",
                                    secondary_hue="sky",
                                    neutral_hue="neutral",
                                    font =  gr.themes.GoogleFont("Roboto")),
               title="Factor-Analysis-with-rotation") as demo:
    gr.HTML(intro)
    gr.HTML(desc)
    method = gr.Radio(["PCA", "Unrotated FA", "Varimax FA"], label = "Choose method to show on the plot:", value = "PCA")
    with gr.Box():
        with gr.Column():
            components = gr.Dataframe(headers= feature_names,label = "Loadings")
            with gr.Row():
                fig1 = gr.Plot(label="Plot covariance of Iris features")
                fig2 = gr.Plot(label="Factor analysis")
    
    method.change(fn=factor_analysis, inputs=method, outputs=[fig1, fig2, components])
    demo.load(fn=factor_analysis, inputs=method, outputs=[fig1, fig2, components])
    gr.HTML(made)
    gr.HTML(link)
    
demo.launch()