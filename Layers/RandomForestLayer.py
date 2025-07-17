import numpy as np
from .Base import BaseLayer

class RandomForestLayer(BaseLayer):
    """
    Random Forest Layer
    
    This layer implements Random Forest classification by building multiple
    decision trees and using majority voting for predictions.
    
    Attributes:
        n_trees: Number of trees in the forest
        max_depth: Maximum depth of each tree
        min_samples_split: Minimum samples required to split a node
        max_features: Number of features to consider for each split
        trees: List of trained decision trees
        training_samples: Stored training samples
        training_labels: Stored training labels
        num_classes: Number of output classes
    """
    
    def __init__(self, n_trees=10, max_depth=5, min_samples_split=10, max_features=None, num_classes=10, random_state=42):
        super().__init__()
        self.trainable = False  # Random Forest doesn't learn parameters in the NN sense
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.num_classes = num_classes
        self.random_state = random_state
        self.trees = []
        self.training_samples = None
        self.training_labels = None
        self.input_tensor = None
        self.is_fitted = False
        
        # Set random seed for reproducibility
        np.random.seed(random_state)
        
    class SimpleDecisionTree:
        """Simple decision tree implementation for Random Forest"""
        
        def __init__(self, max_depth=5, min_samples_split=10, max_features=None):
            self.max_depth = max_depth
            self.min_samples_split = min_samples_split
            self.max_features = max_features
            self.tree = None
            
        def _entropy(self, y):
            """Calculate entropy of labels"""
            if len(y) == 0:
                return 0
            _, counts = np.unique(y, return_counts=True)
            probabilities = counts / len(y)
            return -np.sum(probabilities * np.log2(probabilities + 1e-10))
        
        def _information_gain(self, X, y, feature_idx, threshold):
            """Calculate information gain for a split"""
            left_mask = X[:, feature_idx] <= threshold
            right_mask = ~left_mask
            
            if np.sum(left_mask) == 0 or np.sum(right_mask) == 0:
                return 0
            
            n = len(y)
            n_left, n_right = np.sum(left_mask), np.sum(right_mask)
            
            entropy_parent = self._entropy(y)
            entropy_left = self._entropy(y[left_mask])
            entropy_right = self._entropy(y[right_mask])
            
            weighted_entropy = (n_left/n) * entropy_left + (n_right/n) * entropy_right
            return entropy_parent - weighted_entropy
        
        def _find_best_split(self, X, y):
            """Find the best feature and threshold to split on"""
            n_features = X.shape[1]
            
            # Random feature selection
            if self.max_features is not None:
                feature_indices = np.random.choice(n_features, 
                                                 min(self.max_features, n_features), 
                                                 replace=False)
            else:
                feature_indices = np.arange(n_features)
            
            best_gain = 0
            best_feature = None
            best_threshold = None
            
            for feature_idx in feature_indices:
                thresholds = np.unique(X[:, feature_idx])
                for threshold in thresholds:
                    gain = self._information_gain(X, y, feature_idx, threshold)
                    if gain > best_gain:
                        best_gain = gain
                        best_feature = feature_idx
                        best_threshold = threshold
            
            return best_feature, best_threshold, best_gain
        
        def _build_tree(self, X, y, depth=0):
            """Recursively build the decision tree"""
            # Base cases
            if (depth >= self.max_depth or 
                len(y) < self.min_samples_split or 
                len(np.unique(y)) == 1):
                # Return leaf node with majority class
                unique, counts = np.unique(y, return_counts=True)
                return {'class': unique[np.argmax(counts)]}
            
            # Find best split
            feature, threshold, gain = self._find_best_split(X, y)
            
            if feature is None or gain == 0:
                # No good split found, return leaf
                unique, counts = np.unique(y, return_counts=True)
                return {'class': unique[np.argmax(counts)]}
            
            # Split data
            left_mask = X[:, feature] <= threshold
            right_mask = ~left_mask
            
            # Build subtrees
            left_subtree = self._build_tree(X[left_mask], y[left_mask], depth + 1)
            right_subtree = self._build_tree(X[right_mask], y[right_mask], depth + 1)
            
            return {
                'feature': feature,
                'threshold': threshold,
                'left': left_subtree,
                'right': right_subtree
            }
        
        def fit(self, X, y):
            """Train the decision tree"""
            self.tree = self._build_tree(X, y)
        
        def _predict_sample(self, x, tree):
            """Predict a single sample using the tree"""
            if 'class' in tree:
                return tree['class']
            
            if x[tree['feature']] <= tree['threshold']:
                return self._predict_sample(x, tree['left'])
            else:
                return self._predict_sample(x, tree['right'])
        
        def predict(self, X):
            """Predict multiple samples"""
            return np.array([self._predict_sample(x, self.tree) for x in X])
        
    def store_training_data(self, samples, labels):
        """Store training samples and labels, then train the forest"""
        self.training_samples = samples.copy()
        self.training_labels = labels.copy()
        print(f"Training Random Forest with {len(samples)} training samples...")
        self._train_forest()
        
    def _train_forest(self):
        """Train all trees in the forest"""
        self.trees = []
        n_samples = len(self.training_samples)
        
        # Set max_features default (sqrt of total features)
        if self.max_features is None:
            self.max_features = int(np.sqrt(self.training_samples.shape[1]))
        
        for i in range(self.n_trees):
            # Bootstrap sampling
            bootstrap_indices = np.random.choice(n_samples, n_samples, replace=True)
            bootstrap_X = self.training_samples[bootstrap_indices]
            bootstrap_y = self.training_labels[bootstrap_indices]
            
            # Create and train tree
            tree = self.SimpleDecisionTree(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                max_features=self.max_features
            )
            tree.fit(bootstrap_X, bootstrap_y)
            self.trees.append(tree)
        
        self.is_fitted = True
        print(f"Trained {len(self.trees)} trees successfully")
        
    def forward(self, input_tensor):
        """Forward pass: classify using Random Forest ensemble"""
        if not self.is_fitted:
            raise ValueError("Random Forest must be fitted before prediction")
            
        self.input_tensor = input_tensor
        batch_size = input_tensor.shape[0]
        
        # Flatten input if needed
        if len(input_tensor.shape) > 2:
            flattened_input = input_tensor.reshape(batch_size, -1)
        else:
            flattened_input = input_tensor
        
        # Get predictions from all trees
        tree_predictions = []
        for tree in self.trees:
            predictions = tree.predict(flattened_input)
            tree_predictions.append(predictions)
        
        tree_predictions = np.array(tree_predictions)  # shape: (n_trees, batch_size)
        
        # Ensemble voting - convert to probabilities
        output_tensor = np.zeros((batch_size, self.num_classes))
        
        for i in range(batch_size):
            # Count votes for each class from all trees
            sample_predictions = tree_predictions[:, i]
            unique_classes, counts = np.unique(sample_predictions, return_counts=True)
            
            for class_label, count in zip(unique_classes, counts):
                output_tensor[i, int(class_label)] = count / self.n_trees
        
        return output_tensor
    
    def backward(self, error_tensor):
        """Backward pass: Random Forest doesn't compute gradients"""
        return np.zeros_like(self.input_tensor)
    
    def initialize(self, weights_initializer, bias_initializer):
        """Random Forest doesn't need initialization"""
        pass
    
    def get_feature_importance(self):
        """Get feature importance (simplified version)"""
        if not self.is_fitted:
            return None
        
        # This is a simplified version - in practice, you'd track feature usage during tree building
        n_features = self.training_samples.shape[1]
        importance = np.zeros(n_features)
        
        # Count how often each feature is used across all trees (simplified)
        for tree in self.trees:
            used_features = self._extract_used_features(tree.tree)
            for feature_idx in used_features:
                importance[feature_idx] += 1
        
        # Normalize
        importance = importance / np.sum(importance) if np.sum(importance) > 0 else importance
        return importance
    
    def _extract_used_features(self, tree_node):
        """Extract features used in a tree (recursive)"""
        if 'class' in tree_node:
            return []
        
        features = [tree_node['feature']]
        features.extend(self._extract_used_features(tree_node['left']))
        features.extend(self._extract_used_features(tree_node['right']))
        return features
