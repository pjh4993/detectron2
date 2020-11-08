from multiprocessing import Pool
import numpy as np
import seaborn as sns
import pandas as pd

def draw_plot(df):
    g = sns.PairGrid(df, hue="score")
    g.map_diag(sns.histplot)
    g.map_offdiag(sns.scatterplot)
    g.add_legend()

    return g


data_path = "/media/pjh3974/demo/graph_data.npy"
graph_npy = np.load(data_path)
df = pd.DataFrame(data=graph_npy, columns=["iou","dtCtr","gtCtr","score","gtAsp"])

p = Process(target=draw_plot, args=(df))
p.start()
p.join()

g.savefig(os.path.join(self._output_dir, "sns_graph.png"))