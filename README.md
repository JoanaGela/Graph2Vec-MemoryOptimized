# Graph2Vec-MemoryOptimized

Variant of Graph2Vec with a specific focus on optimizing memory usage.

This repository contains a variant of the Graph2Vec algorithm specifically tailored for a dataset of graphs that share the same base shape but differ in node attributes. 

In our application scenario, the dataset comprises graphs of identical base shapes but with varying node attributes. The default attribute for nodes is set to 0. Additionally, all graphs are Directed Acyclic Graphs (DAGs).

## Changes to Graph2Vec

* Input Format: Unlike the original Graph2Vec implementation, where the input is a list of graphs, our variant takes a list of lists as input. Each inner list represents a graph and contains only the nodes with non-zero attributes.


The original work can be found in: https://arxiv.org/pdf/1707.05005.pdf and in the [Karate Club](https://github.com/benedekrozemberczki/karateclub) package.
