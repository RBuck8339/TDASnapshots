import numpy as np
import sklearn
import kmapper as km
from sklearn.preprocessing import MinMaxScaler
import pickle
import os
import multiprocessing
import shutil
import networkx as nx
import pandas as pd 
import datetime as dt
from node2vec import Node2Vec  # Might use, uncertain as of now
from collections import defaultdict


class MyNetworkParser:
    # Directories
    timeseries_data_path = "data/"
    viz_dir = "Outputs/tda_viz/"
    tda_dir = "Outputs/tda_graph/"
    sequence_dir = "Outputs/Sequence/"
    
    timeWindow = [7]
    # Validation duration condition
    networkValidationDuration = 20
    finalDataDuration = 5
    labelTreshholdPercentage = 10
    
    
    def create_time_series_sequence(self, file):
        totalRnnSequenceData = list()
        totalRnnLabelData = list()
        print("Processing {}".format(file))
        windowSize = 7  # Day
        gap = 3
        lableWindowSize = 7  # Day
        maxDuration = 180  # Day
        indx = 0
        maxIndx = 2

        selectedNetwork = pd.read_csv((self.timeseries_data_path + file), sep=' ',
                                      names=["from", "to", "date", "value"])
        selectedNetwork['date'] = pd.to_datetime(selectedNetwork['date'], unit='s').dt.date
        selectedNetwork['value'] = selectedNetwork['value'].astype(float)
        selectedNetwork = selectedNetwork.sort_values(by='date')
        window_start_date = selectedNetwork['date'].min()
        data_last_date = selectedNetwork['date'].max()

        print(f"{file} -- {window_start_date} -- {data_last_date}")

        print("\n {} Days OF Data -> {} ".format(file, (data_last_date - window_start_date).days))
        # check if the network has more than 20 days of data
        if ((data_last_date - window_start_date).days < maxDuration):
            print(file + "Is not a valid network")
            shutil.move(self.file_path + file, self.file_path + "Invalid/" + file)
            return

        # normalize the edge weights for the graph network {0-9}
        max_transfer = float(selectedNetwork['value'].max())
        min_transfer = float(selectedNetwork['value'].min())

        selectedNetwork['value'] = selectedNetwork['value'].apply(
            lambda x: 1 + (9 * ((float(x) - min_transfer) / (max_transfer - min_transfer))))

        # Graph Generation Process and Labeling

        while (data_last_date - window_start_date).days > (windowSize + gap + lableWindowSize):
            print("\nRemaining Process  {} ".format(

                (data_last_date - window_start_date).days / (windowSize + gap + lableWindowSize)))
            indx += 1
            # if (indx == maxIndx):
            #     break
            transactionGraph = nx.MultiDiGraph()

            # select window data
            window_end_date = window_start_date + dt.timedelta(days=windowSize)
            selectedNetworkInGraphDataWindow = selectedNetwork[
                (selectedNetwork['date'] >= window_start_date) & (
                        selectedNetwork['date'] < window_end_date)]

            # select labeling data
            label_end_date = window_start_date + dt.timedelta(days=windowSize) + dt.timedelta(
                days=gap) + dt.timedelta(
                days=lableWindowSize)
            label_start_date = window_start_date + dt.timedelta(days=windowSize) + dt.timedelta(days=gap)
            selectedNetworkInLbelingWindow = selectedNetwork[
                (selectedNetwork['date'] >= label_start_date) & (selectedNetwork['date'] < label_end_date)]

            # generating the label for this window
            # 1 -> Increading Transactions 0 -> Decreasing Transactions
            label = 1 if (len(selectedNetworkInLbelingWindow) - len(
                selectedNetworkInGraphDataWindow)) > 0 else 0

            # group by for extracting node features
            outgoing_weight_sum = (selectedNetwork.groupby(by=['from'])['value'].sum())
            incoming_weight_sum = (selectedNetwork.groupby(by=['to'])['value'].sum())
            outgoing_count = (selectedNetwork.groupby(by=['from'])['value'].count())
            incoming_count = (selectedNetwork.groupby(by=['to'])['value'].count())

            # Node Features Dictionary for TDA mapper usage
            node_features = pd.DataFrame()

            # Populate graph with edges
            for item in selectedNetworkInGraphDataWindow.to_dict(orient="records"):
                from_node_features = {}
                to_node_features = {}
                # calculating node features for each edge
                # feature 1 -> sum of outgoing edge weights
                from_node_features["outgoing_edge_weight_sum"] = outgoing_weight_sum[item['from']]

                try:
                    to_node_features["outgoing_edge_weight_sum"] = outgoing_weight_sum[item['to']]
                except Exception as e:
                    to_node_features["outgoing_edge_weight_sum"] = 0

                # feature 2 -> sum of incoming edge weights
                to_node_features["incoming_edge_weight_sum"] = incoming_weight_sum[item['to']]
                try:
                    from_node_features["incoming_edge_weight_sum"] = incoming_weight_sum[item['from']]
                except Exception as e:
                    from_node_features["incoming_edge_weight_sum"] = 0
                # feature 3 -> number of outgoing edges
                from_node_features["outgoing_edge_count"] = outgoing_count[item['from']]
                try:
                    to_node_features["outgoing_edge_count"] = outgoing_count[item['to']]
                except Exception as e:
                    to_node_features["outgoing_edge_count"] = 0

                # feature 4 -> number of incoming edges
                to_node_features["incoming_edge_count"] = incoming_count[item['to']]
                try:
                    from_node_features["incoming_edge_count"] = incoming_count[item['from']]
                except Exception as e:
                    from_node_features["incoming_edge_count"] = 0

                # add temporal vector to all nodes, populated with -1

                from_node_features_with_daily_temporal_vector = dict(from_node_features)
                from_node_features_with_daily_temporal_vector["dailyClusterID"] = [-1] * windowSize
                from_node_features_with_daily_temporal_vector["dailyClusterSize"] = [-1] * windowSize

                to_node_features_with_daily_temporal_vector = dict(to_node_features)
                to_node_features_with_daily_temporal_vector["dailyClusterID"] = [-1] * windowSize
                to_node_features_with_daily_temporal_vector["dailyClusterSize"] = [-1] * windowSize

                # Temporal version
                transactionGraph.add_nodes_from(
                    [(item["from"], from_node_features_with_daily_temporal_vector)])
                transactionGraph.add_nodes_from([(item["to"], to_node_features_with_daily_temporal_vector)])
                transactionGraph.add_edge(item["from"], item["to"], value=item["value"])

                new_row = pd.DataFrame(({**{"nodeID": item["from"]}, **from_node_features}), index=[0])
                node_features = pd.concat([node_features, new_row], ignore_index=True)

                new_row = pd.DataFrame(({**{"nodeID": item["to"]}, **to_node_features}), index=[0])
                node_features = pd.concat([node_features, new_row], ignore_index=True)

                node_features = node_features.drop_duplicates(subset=['nodeID'])

            timeWindowSequence = self.process_TDA_extracted_rnn_sequence(selectedNetworkInGraphDataWindow, node_features)

            # timeWindowSequenceRaw = self.processRawExtractedRnnSequence(selectedNetworkInGraphDataWindow, node_features)
            # result_list = []
            # first_key_tda = next(iter(timeWindowSequence))
            # tda_value = timeWindowSequence[first_key_tda]
            #
            # first_key_raw = next(iter(timeWindowSequenceRaw))
            # raw_value = timeWindowSequenceRaw[first_key_raw]
            #
            # for sublist1, sublist2 in zip(tda_value, raw_value):
            #     merged_sublist = sublist1 + sublist2
            #     result_list.append(merged_sublist)
            #
            #
            # totalRnnSequenceData.append({first_key_tda + "_" + first_key_raw : result_list})

            totalRnnSequenceData.append(timeWindowSequence)
            totalRnnLabelData.append(label)
            window_start_date = window_start_date + dt.timedelta(days=1)

        total_merged_seq = self.merge_dicts(totalRnnSequenceData)
        finalDict = {"sequence": total_merged_seq, "label": totalRnnLabelData}
        print(finalDict)
        directory = 'Sequence/' + str(file)
        if not os.path.exists(directory):
            os.makedirs(directory)
        with open(directory + '/seq_tda_ablation.txt',
                  'wb') as file_in:
            pickle.dump(finalDict, file_in)
            file_in.close()

    def embed_node2vec(self, graph, dim, walk_len, num_walks, p, q, window):
        """
            Embed nodes using the Node2Vec class object. Implementing so that we have options later

            Args:
                graph (nx.MultiDiGraph()): The graph we are seeking to embed
                dim (int): The dimension of our embeddings
                walk_len (int): The length of each random walk
                num_walks (int): The number of random walks per node
                p (float): The likelihood of returning to the previous node
                q (float): The likelihood of visiting nodes far from starting node
                window (int): The number of neighboring nodes considered around target node during training

            Returns:
                embeddings (np.array()): An array of node embeddings for the given graph
        """
        node2vec = Node2Vec(graph, dimensions=dim, walk_length=walk_len, num_walks=num_walks, p=p, q=q)
        node2vec_model = node2vec.fit(window=window, min_count=1, batch_words=4)
            
        embeddings = np.array([node2vec_model.wv[str(node)] for node in graph.nodes()])

        return embeddings  # Need to edit to be a dataframe


    def embed_structure(self, selectedNetwork, selectedNetworkInGraphDataWindow, windowSize, transactionGraph):
        pass 
    
    
    # Not filling out for now, depends if we want to concatenate original graph information like they do in graphpulse
    def extract_graph_features(self, graph):
        pass
    
    
    # Ignoring for now, can do at a later date if necessary
    def extract_node_features(self, graph):
        pass
    

    def tda_to_networkx(self, tda_graph):
        """
            Turns a TDA graph object into a Networkx Graph object while preserving structure

            Args:
                tda_graph (dict): A TDA graph received from mapper 

            Returns:
                nx_graph (mx.Graph()): A networkx representation of the TDA mapper graph
        """
        nx_graph = nx.Graph()
    
        # Add nodes and cluster size attribute
        for node, cluster in tda_graph['nodes'].items():
            nx_graph.add_node(node, size=len(cluster))

        # Add edges to the graph
        for edge in tda_graph['links']:
            nx_graph.add_edge(edge[0], edge[1])
        
        return nx_graph
    

    def tda_process(self, mapper, lens, Xfilt, per_overlap, n_cubes, cls):
        """
            Creates a TDA graph and returns extracted features

            Args:
                mapper (km.KeplerMapper()): A TDA graph received from mapper 
                lens (array-like): Node embeddings with reduced dimensionality
                Xfilt (array-like): Original scaled node embeddings
                per_overlap (float): The percent overlap determining graph structure in TDA
                n_cubes (int): The number of cubes for mapper to map
                cls (int): The number of clusters for mapper to map

            Returns:
                features.values() (list): Our TDA feature vector
        """

        graph = mapper.map(
            lens,
            Xfilt,
            clusterer=sklearn.cluster.KMeans(n_clusters=cls, random_state=42),
            cover=km.Cover(n_cubes=n_cubes, perc_overlap=per_overlap),  
        )
        
        # If you want a visualization, uncomment this
        '''
        node_ids = list(nx_graph.nodes())
        custom_tooltips = [str(node_id) for node_id in node_ids]  # Use node IDs as tooltips
        custom_tooltips = np.array(custom_tooltips)  # Necessary or else mapper will crash
        
        mapper.visualize(
            graph,
            title="Ethereum Nodes Mapper",
            path_html=self.viz_dir + "" + str() + ".html",
            color_values=node_ids,  # Optional, color based on node IDs
            color_function_name="Node IDs",
            custom_tooltips=custom_tooltips,  # Must be a numpy array
        )
        '''

        features = {  
            'num_nodes': 0,
            'num_edges': 0,
            'density': 0,
            'max_cluster_size': 0,
            'avg_cluster_size': 0,
            'num_connected_components': 0,
            'largest_connected_component': 0,
            'average_edge_weight': 0
        }
        
        try:
            nx_graph = self.tda_to_networkx(graph)
            
            # Number of nodes
            features['num_nodes'] = len(graph['nodes'])
            # Number of edges
            features['num_edges'] = sum(len(edges) for edges in graph['links'].values())
            # Density
            features['density'] = nx.density(nx_graph) if nx_graph.number_of_edges() > 0 else 0

            
            # max cluster size
            features['max_cluster_size'] = len(
                graph["nodes"][
                    max(graph["nodes"], key=lambda k: len(graph["nodes"][k]))])

            # average clsuter size
            cluster_sizes = [len(nodes) for nodes in graph["nodes"].values()]
            features['avg_cluster_size'] = sum(cluster_sizes) / len(cluster_sizes)

            # Connected Components
            connected_components = list(nx.connected_components(nx_graph))
            largest_connected_component = max(connected_components, key=len) if connected_components else []
            features['num_connected_components'] = len(connected_components)
            features['largest_connected_component'] = len(largest_connected_component)
            
            # average edge weight
            edge_weights = defaultdict(dict)
            for source_node, target_nodes in graph['links'].items():
                for target_node in target_nodes:
                    common_indexes = len(
                        set(graph['nodes'][source_node]) & set(graph['nodes'][target_node]))
                    edge_weights[source_node][target_node] = common_indexes
                    total_edge_weights = sum(
                        weight for target_weights in edge_weights.values() for weight in target_weights.values())
                    total_edges = sum(len(target_weights) for target_weights in edge_weights.values())
                    features['average_edge_weight'] = total_edge_weights / total_edges


        except Exception as e:
            print("Caught exception creating features")
            print(e)
            
        key = "overlap{}-cube{}-cls{}".format(per_overlap, n_cubes, cls)
        daily_features = {key: features.values()}
        
        return daily_features
    
    
    def process_TDA_extracted_rnn_sequence(self, timeFrameData, nodeFeatures):

        # break the data to daily graphs
        timeWindowSequence = list()
        data_first_date = timeFrameData['date'].min()
        data_last_date = timeFrameData['date'].max()
        numberOfDays = (data_last_date - data_first_date).days
        start_date = data_first_date
        # initiate the graph
        processingDay = 0
        while (processingDay <= numberOfDays):
            # print("Processing TDA RNN sequential Extraction day {}".format(processingDay))
            daily_end_date = start_date + dt.timedelta(days=1)
            selectedDailyNetwork = timeFrameData[
                (timeFrameData['date'] >= start_date) & (timeFrameData['date'] < daily_end_date)]

            daily_node_features = pd.DataFrame()

            for item in selectedDailyNetwork.to_dict(orient="records"):
                new_row = pd.DataFrame(({**{"nodeID": item["from"]},
                                         **nodeFeatures[nodeFeatures["nodeID"] == item["to"]].drop("nodeID",
                                                                                                   axis=1).to_dict(
                                             orient='records')[0]}),
                                       index=[0])
                daily_node_features = pd.concat([daily_node_features, new_row], ignore_index=True)

                new_row = pd.DataFrame(({**{"nodeID": item["to"]},
                                         **nodeFeatures[nodeFeatures["nodeID"] == item["to"]].drop("nodeID",
                                                                                                   axis=1).to_dict(
                                             orient='records')[0]}),
                                       index=[0])
                daily_node_features = pd.concat([daily_node_features, new_row], ignore_index=True)

                daily_node_features = daily_node_features.drop_duplicates(subset=['nodeID'])

            # creat the TDA for each day
            try:

                Xfilt = daily_node_features
                Xfilt = Xfilt.drop(columns=['nodeID'])
                mapper = km.KeplerMapper()
                scaler = MinMaxScaler(feature_range=(0, 1))

                Xfilt = scaler.fit_transform(Xfilt)
                lens = mapper.fit_transform(Xfilt, projection=sklearn.manifold.TSNE())
                
                # UMAP
                '''
                umap_model = umap.UMAP(n_neighbors=10, min_dist=0.1, n_components=2)
                lens = umap_model.fit_transform(Xfilt)
                '''

                results = []

                per_overlap = 0.25
                n_cubes = 10
                cls = 1 
                for albation_index in [1]:
                    # per_overlap = round(per_overlap_indx * 0.05, 2)
                    result = (self.tda_process(mapper, lens, Xfilt, per_overlap, n_cubes, cls))
                    results.append(result)
                
                # Retrieve the results as they become available
                for result in results:
                    dailyFeatures = result
                    timeWindowSequence.append(dailyFeatures)
                

            except Exception as e:
                print(str(e))
            start_date = start_date + dt.timedelta(days=1)
            processingDay += 1
        
        # the graph has been repopulated with daily temporal features
        merged_dict = self.merge_dicts(timeWindowSequence)
        return merged_dict


    # Utility function, used to build the sequence/label pair
    def merge_dicts(self, list_of_dicts):
        merged_dict = {}
        for dictionary in list_of_dicts:
            for key, value in dictionary.items():
                if key in merged_dict:
                    merged_dict[key].append(value)
                else:
                    merged_dict[key] = [value]
        return merged_dict



if __name__ == "__main__":
    np = MyNetworkParser()
    file = 'networkadex.txt'
    np.create_time_series_sequence(file)
    sequence = 'Sequence/' + file


'''
Generate Snapshots
Get a dataframe of either strucutral encodings or node2vec embeddings where I can query based on the nodeid in question to build a daily graph
For Each snapshot, generate a TDA graph for each daily graph based on their roles in the graph for the entire snapshot
Generate graph-level features for the TDA graph (These will act as the labels for our RNN)
Still need to figure out what our input features are for the RNN (honestly the labels could be both)
Run data through the RNN
'''