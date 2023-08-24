# Code source: Gaël Varoquaux
# License: BSD 3 clause

import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
import gradio as gr
from matplotlib.colors import ListedColormap
plt.switch_backend("agg")

font1 = {'family':'DejaVu Sans','size':20}

def create_data(random, size_num, x_min, x_max, y_min, y_max):
    #emulate some random data
    if random:
        size_num = int(size_num)
        x = np.random.uniform(x_min, x_max, size=(size_num, 1))
        y = np.random.uniform(y_min, y_max, size=(size_num, 1))
        
        X = np.hstack((x, y))
        Y = [0] * int(size_num/2) + [1] * int(size_num/2)
    else:
        X = np.c_[
            (0.4, -0.7),
            (-1.5, -1),
            (-1.4, -0.9),
            (-1.3, -1.2),
            (-1.5, 0.2),
            (-1.2, -0.4),
            (-0.5, 1.2),
            (-1.5, 2.1),
            (1, 1),
            # --
            (1.3, 0.8),
            (1.5, 0.5),
            (0.2, -2),
            (0.5, -2.4),
            (0.2, -2.3),
            (0, -2.7),
            (1.3, 2.8),
        ].T

        Y = [0] * 8 + [1] * 8
    return X, Y

# fit the model
def clf_kernel(color1, color2, dpi, size_num = None, x_min = None, 
                x_max = None, y_min = None,
                y_max = None, random = False):

    if size_num is not None or x_min is not None or x_max is not None or y_min is not None or y_max is not None:
        random = True

    X, Y = create_data(random, size_num, x_min, x_max, y_min, y_max)

    kernels = ["linear", "poly", "rbf"]
          
    # plot the line, the points, and the nearest vectors to the plane  
    fig, axs = plt.subplots(1,3, figsize = (16,8), facecolor='none', dpi = res[dpi])
    
    cmap = ListedColormap([color1, color2], N=2, name = 'braincell')
    for i, kernel in enumerate(kernels):
        clf = svm.SVC(kernel=kernel, gamma=2)
        clf.fit(X, Y)
        axs[i].scatter(
            clf.support_vectors_[:, 0],
            clf.support_vectors_[:, 1],
            s=80,
            facecolors="none",
            zorder=10,
            edgecolors="k",
        )
        axs[i].scatter(X[:, 0], X[:, 1], c=Y, zorder=10, cmap=cmap, edgecolors="k")

        axs[i].axis("tight")
        x_min = -3
        x_max = 3
        y_min = -3
        y_max = 3

        XX, YY = np.mgrid[x_min:x_max:200j, y_min:y_max:200j]
        Z = clf.decision_function(np.c_[XX.ravel(), YY.ravel()])

        # Put the result into a color plot
        Z = Z.reshape(XX.shape)
        axs[i].pcolormesh(XX, YY, Z > 0, cmap=cmap)
        axs[i].contour(
            XX,
            YY,
            Z,
            colors=["k", "k", "k"],
            linestyles=["--", "-", "--"],
            levels=[-0.5, 0, 0.5],
        )

        axs[i].set_xlim(x_min, x_max)
        axs[i].set_ylim(y_min, y_max)

        axs[i].set_xticks(())
        axs[i].set_yticks(())
        axs[i].set_title('Type of kernel: ' + kernel, 
                    color = "white", fontdict = font1, pad=20,  
                    bbox=dict(boxstyle="round,pad=0.3", 
                            color = "#6366F1"))
        
        plt.close()
    return fig, np.round(X, decimals=2)

intro = """<h1 style="text-align: center;">🤗 Introducing SVM-Kernels 🤗</h1>
"""
desc = """<h3 style="text-align: center;">Three different types of SVM-Kernels are displayed below. 
The polynomial and RBF are especially useful when the data-points are not linearly separable. </h3>
"""
notice = """<br><div style = "text-align: left;"> <em>Notice: Run the model on example data or use <strong>Randomize data</strong> 
button below to check out the model on randomized data-points. Any changes to visual parameters will reset the data!</em></div>"""

notice2 = """<br><div style = "text-align: left;"> <em>Notice:  The data points are categorized into two distinct classes, and they are evenly distributed on the plots to visually represent these classes.</em></div>"""

made ="""<div style="text-align: center;">
  <p>Made with ❤</p>"""

link = """<div style="text-align: center;">
<a href="https://scikit-learn.org/stable/auto_examples/svm/plot_svm_kernels.html#sphx-glr-auto-examples-svm-plot-svm-kernels-py" target="_blank" rel="noopener noreferrer">
Demo is based on this script from scikit-learn documentation</a>"""

res = {'Small': 50, 'Medium': 75, 'Large': 100}

with gr.Blocks(theme=gr.themes.Soft(primary_hue="indigo",
                                    secondary_hue="violet",
                                    neutral_hue="slate",
                                    font =  gr.themes.GoogleFont("Inter")),
               title="SVM-Kernels") as demo:
    
    gr.HTML(intro)
    gr.HTML(desc)
    
    with gr.Tab("Plotted results"):
        plot = gr.Plot(label="Kernel comparison:")
    with gr.Tab("Data coordinates"):
        gr.HTML(notice2)
        X = gr.Numpy(headers = ['x','y'], interactive=False)
    
    with gr.Column():
        
        with gr.Accordion(label = 'Randomize data'):
            gr.HTML(notice)
            samples = gr.Slider(4, 16, value = 8, step = 2, label = "Number of samples:")
            x_min = gr.Slider(-3, 0, value=-2, step=0.1, label="X Min:")
            x_max = gr.Slider(0, 3, value=2, step=0.1, label="X Max:")
            y_min = gr.Slider(-3, 0, value=-2, step=0.1, label="Y Min:")
            y_max = gr.Slider(0, 3, value=2, step=0.1, label="Y Max:")
            random = gr.Button("Randomize data")
        

        
        
        with gr.Accordion(label = "Visual parameters"):
            with gr.Row():
                color1 = gr.ColorPicker(label = 'Pick color one:', value = '#9abfd8')
                color2 = gr.ColorPicker(label = 'Pick color two:', value = '#371c4b')
            #dpi = gr.Slider(50, 100, value = 75, step = 1, label = "Set the resolution: ")
            dpi = gr.Radio(list(res.keys()), value = 'Medium', label = "Select the plot size:")
      
    params2 = [color1, color2, dpi]

    random.click(fn=clf_kernel, inputs=[color1, color2, dpi,samples, x_min, x_max, y_min, y_max], outputs=[plot,X]) 

    for i in params2:
        i.change(fn=clf_kernel, inputs=[color1, color2,dpi], outputs=[plot, X])
    
    demo.load(fn=clf_kernel, inputs=[color1, color2, dpi], outputs=[plot,X]) 
    gr.HTML(made)
    gr.HTML(link)
    
demo.launch()