conv_features = {
    'node_features_shape': (3, 32, 32),
    'edge_features_shape': (4, 32, 32),
    'global_features_shape': (1, 32, 32)
}
linear_features = {
    'node_features_shape': 100,
    'edge_features_shape': 7,
    'global_features_shape': 2
}
no_node_features = {
    'node_features_shape': None,
    'edge_features_shape': 7,
    'global_features_shape': 2
}
no_edge_features = {
    'node_features_shape': 100,
    'edge_features_shape': None,
    'global_features_shape': 2
}
no_global_features = {
    'node_features_shape': 100,
    'edge_features_shape': 7,
    'global_features_shape': None
}
all_features = {
    'conv_features': conv_features,
    'linear_features': linear_features,
    'no_node_features': no_node_features,
    'no_edge_features': no_edge_features,
    'no_global_features': no_global_features
}
