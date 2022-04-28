# Manifold Learning and Graph Kernels

Third assignment solved in "Artificial Intelligence: Knowledge Representation and planning" course at Ca'Foscari University, check the <a href="">report</a> for a project explanation.

## Description of the project

Read <a href="https://www.dsi.unive.it/~atorsell/AI/graph/Unfolding.pdf">this article</a> presenting a way to improve the disciminative power of graph kernels.

Choose one <a href="https://www.dsi.unive.it/~atorsell/AI/graph/kernels.pdf">graph kernel</a> among

Shortest-path Kernel
Graphlet Kernel
Random Walk Kernel
Weisfeiler-Lehman Kernel
Choose one manifold learning technique among

Isomap
Diffusion Maps
Laplacian Eigenmaps
Local Linear Embedding
Compare the performance of an SVM trained on the given kernel, with or without the manifold learning step, on the following datasets:

<a href="https://www.dsi.unive.it/~atorsell/AI/graph/PPI.zip">PPI</a>
<a href="https://www.dsi.unive.it/~atorsell/AI/graph/SHOCK.zip">Shock</a>
The zip files contain csv files representing the adjacecy matrices of the graphs and of the lavels. the files graphxxx.csv contain the adjaccency matrices, one per file, while the file labels.csv contains all the labels
