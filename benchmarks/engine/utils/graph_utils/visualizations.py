import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
import igraph as ig


from .constants import DatasetType, GraphVisualizationTool, network_repository_cora_url, cora_label_to_color_map
from .utils import convert_adj_to_edge_index


def plot_in_out_degree_distributions(edge_index, num_of_nodes, dataset_name):
    """
        Note: It would be easy to do various kinds of powerful network analysis using igraph/networkx, etc.
        I chose to explicitly calculate only the node degree statistics here, but you can go much further if needed and
        calculate the graph diameter, number of triangles and many other concepts from the network analysis field.
    """
    assert isinstance(edge_index, np.ndarray), f'Expected NumPy array got {type(edge_index)}.'
    if edge_index.shape[0] == edge_index.shape[1]:
        edge_index = convert_adj_to_edge_index(edge_index)

    # Store each node's input and output degree (they're the same for undirected graphs such as Cora)
    in_degrees = np.zeros(num_of_nodes, dtype=np.int)
    out_degrees = np.zeros(num_of_nodes, dtype=np.int)

    # Edge index shape = (2, E), the first row contains the source nodes, the second one target/sink nodes
    # Note on terminology: source nodes point to target/sink nodes
    num_of_edges = edge_index.shape[1]
    for cnt in range(num_of_edges):
        source_node_id = edge_index[0, cnt]
        target_node_id = edge_index[1, cnt]

        out_degrees[source_node_id] += 1  # source node points towards some other node -> increment it's out degree
        in_degrees[target_node_id] += 1  # similarly here

    hist = np.zeros(np.max(out_degrees) + 1)
    for out_degree in out_degrees:
        hist[out_degree] += 1

    fig = plt.figure()
    fig.subplots_adjust(hspace=0.6)

    plt.subplot(311)
    plt.plot(in_degrees, color='red')
    plt.xlabel('node id'); plt.ylabel('in-degree count'); plt.title('Input degree for different node ids')

    plt.subplot(312)
    plt.plot(out_degrees, color='green')
    plt.xlabel('node id'); plt.ylabel('out-degree count'); plt.title('Out degree for different node ids')

    plt.subplot(313)
    plt.plot(hist, color='blue')
    plt.xlabel('node degree'); plt.ylabel('# nodes for a given out-degree'); plt.title(f'Node out-degree distribution for {dataset_name} dataset')
    plt.xticks(np.arange(0, len(hist), 5.0))

    plt.grid(True)
    plt.show()


def visualize_graph(edge_index, node_labels, dataset_name, visualization_tool=GraphVisualizationTool.IGRAPH):
    """
    Check out this blog for available graph visualization tools:
        https://towardsdatascience.com/large-graph-visualization-tools-and-approaches-2b8758a1cd59
    Basically depending on how big your graph is there may be better drawing tools than igraph.
    Note:
    There are also some nice browser-based tools to visualize graphs like this one:
        http://networkrepository.com/graphvis.php?d=./data/gsm50/labeled/cora.edges
    Nonetheless tools like igraph can be useful for quick visualization directly from Python
    """
    assert isinstance(edge_index, np.ndarray), f'Expected NumPy array got {type(edge_index)}.'
    if edge_index.shape[0] == edge_index.shape[1]:
        edge_index = convert_adj_to_edge_index(edge_index)

    num_of_nodes = len(node_labels)
    edge_index_tuples = list(zip(edge_index[0, :], edge_index[1, :]))

    # Networkx package is primarily used for network analysis, graph visualization was an afterthought in the design
    # of the package - but nonetheless you'll see it used for graph drawing as well
    if visualization_tool == GraphVisualizationTool.NETWORKX:
        nx_graph = nx.Graph()
        nx_graph.add_edges_from(edge_index_tuples)
        nx.draw_networkx(nx_graph)
        plt.show()

    elif visualization_tool == GraphVisualizationTool.IGRAPH:
        # Construct the igraph graph
        ig_graph = ig.Graph()
        ig_graph.add_vertices(num_of_nodes)
        ig_graph.add_edges(edge_index_tuples)

        # Prepare the visualization settings dictionary
        visual_style = {}

        # Defines the size of the plot and margins
        visual_style["bbox"] = (3000, 3000)
        visual_style["margin"] = 35

        # I've chosen the edge thickness such that it's proportional to the number of shortest paths (geodesics)
        # that go through a certain edge in our graph (edge_betweenness function, a simple ad hoc heuristic)

        # line1: I use log otherwise some edges will be too thick and others not visible at all
        # edge_betweeness returns < 1.0 for certain edges that's why I use clip as log would be negative for those edges
        # line2: Normalize so that the thickest edge is 1 otherwise edges appear too thick on the chart
        # line3: The idea here is to make the strongest edge stay stronger than others, 6 just worked, don't dwell on it

        edge_weights_raw = np.clip(np.log(np.asarray(ig_graph.edge_betweenness()) + 1e-16), a_min=0, a_max=None)
        edge_weights_raw_normalized = edge_weights_raw / np.max(edge_weights_raw)
        edge_weights = [w**6 for w in edge_weights_raw_normalized]
        visual_style["edge_width"] = edge_weights

        # A simple heuristic for vertex size. Size ~ (degree / 2) (it gave nice results I tried log and sqrt as well)
        visual_style["vertex_size"] = [deg / 2 for deg in ig_graph.degree()]

        # This is the only part that's Cora specific as Cora has 7 labels
        if dataset_name.lower() == DatasetType.CORA.name.lower():
            visual_style["vertex_color"] = [cora_label_to_color_map[label] for label in node_labels]
        else:
            print('Feel free to add custom color scheme for your specific dataset. Using igraph default coloring.')

        # Set the layout - the way the graph is presented on a 2D chart. Graph drawing is a subfield for itself!
        # I used "Kamada Kawai" a force-directed method, this family of methods are based on physical system simulation.
        # (layout_drl also gave nice results for Cora)
        visual_style["layout"] = ig_graph.layout_kamada_kawai()

        print('Plotting results ... (it may take couple of seconds).')
        ig.plot(ig_graph, **visual_style)
    else:
        raise Exception(f'Visualization tool {visualization_tool.name} not supported.')


def draw_entropy_histogram(entropy_array, title, color='blue', uniform_distribution=False, num_bins=30):
    max_value = np.max(entropy_array)
    bar_width = (max_value / num_bins) * (1.0 if uniform_distribution else 0.75)
    histogram_values, histogram_bins = np.histogram(entropy_array, bins=num_bins, range=(0.0, max_value))

    plt.bar(histogram_bins[:num_bins], histogram_values[:num_bins], width=bar_width, color=color)
    plt.xlabel(f'entropy bins')
    plt.ylabel(f'# of node neighborhoods')
    plt.title(title)