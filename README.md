# gognn_7412

We also uploaded the original dataset in the data folder. 

You can download all the intermediate files from the following link: https://github.com/joddiy/gognn_7412/tree/main/data

Also, You can directly clone the GitHub repository to get all codes and dataset.

# Environment setup

Most required packages are included in the `requirements.txt` file. You can install them by running the following
command:

    ```bash
    pip install -r requirements.txt
    ```

However, for the library of `torch_geometric`, you may need a GPU to run the code.

Please follow the installation instruction at https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html
to install the correct version of `torch_geometric` for your system.

# Instruction on how to run the code

1. Download the dataset from the link above, all just clone the repository.
2. Set up the root directory as the working directory for both R and Python Jupiter notebooks.
3. Run the [collect_go.R](r%2Fcollect_go.R) code first to preprocess the data and save the processed data.
4. Run the [graph_prune.ipynb](py%2Fgraph_prune.ipynb) to generate the pruned graph.
5. Run the [train_GNN_DNN.ipynb](py%2Ftrain_GNN_DNN.ipynb) to train the model.
