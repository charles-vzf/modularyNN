import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from .BaseDataset import DataSet

class WaitPark(DataSet):
    """
    Optimized dataset class for the Park Waiting Time Prediction Hackathon.
    
    This dataset contains waiting times for different attractions at a theme park,
    along with optional weather data. The goal is to predict the waiting time
    in 2 hours (WAIT_TIME_IN_2H) based on current conditions.
    """
    def __init__(self, batch_size, include_weather=True, random=True, val_ratio=0.15):
        """
        Initialize the Park Waiting Time dataset.
        
        Args:
            batch_size: Number of samples per batch
            include_weather: Whether to include weather data as features
            random: Whether to shuffle the data for each epoch
            val_ratio: Proportion of data to use for validation (default: 0.15)
        """
        super().__init__(batch_size, random)
        self.include_weather = include_weather
        
        # Load and preprocess the data
        self._load_data()
        
        # Split the data
        train_ratio = 0.7  # 70% for training
        test_ratio = 0.15  # 15% for testing
        
        self.split_data(train_ratio=train_ratio, val_ratio=val_ratio, test_ratio=test_ratio)
    
    def _load_data(self):
        """Load and preprocess the Park Waiting Time dataset with optimizations."""
        # Load main waiting times data
        waiting_times_path = 'C:/Users/cmmcc/Desktop/RIKEN_AIP/code/modularyNN/Data/waitingTimes/waiting_times_train.csv'
        self.waiting_df = pd.read_csv(waiting_times_path)
        
        # Convert datetime to proper format
        self.waiting_df['DATETIME'] = pd.to_datetime(self.waiting_df['DATETIME'])
        
        # Extract temporal features
        self.waiting_df['HOUR'] = self.waiting_df['DATETIME'].dt.hour
        self.waiting_df['MINUTE'] = self.waiting_df['DATETIME'].dt.minute
        self.waiting_df['TIME_MINUTES'] = self.waiting_df['HOUR'] * 60 + self.waiting_df['MINUTE']  # Minutes since midnight
        self.waiting_df['DAY_OF_WEEK'] = self.waiting_df['DATETIME'].dt.dayofweek
        self.waiting_df['MONTH'] = self.waiting_df['DATETIME'].dt.month
        self.waiting_df['YEAR'] = self.waiting_df['DATETIME'].dt.year
        self.waiting_df['DAY'] = self.waiting_df['DATETIME'].dt.day
        
        # Optional: Load and merge weather data
        if self.include_weather:
            weather_path = 'C:/Users/cmmcc/Desktop/RIKEN_AIP/code/modularyNN/Data/waitingTimes/weather_data.csv'
            self.weather_df = pd.read_csv(weather_path)
            self.weather_df['DATETIME'] = pd.to_datetime(self.weather_df['DATETIME'])
            
            # Convertir les colonnes DATETIME en objets datetime si ce n'est pas déjà fait
            if not pd.api.types.is_datetime64_dtype(self.waiting_df['DATETIME']):
                self.waiting_df['DATETIME'] = pd.to_datetime(self.waiting_df['DATETIME'])
            
            if not pd.api.types.is_datetime64_dtype(self.weather_df['DATETIME']):
                self.weather_df['DATETIME'] = pd.to_datetime(self.weather_df['DATETIME'])
            
            # Créez des colonnes de date (sans l'heure) pour la contrainte de fusion par jour
            self.waiting_df['DATE'] = self.waiting_df['DATETIME'].dt.date
            self.weather_df['DATE'] = self.weather_df['DATETIME'].dt.date
            
            # Triez les deux DataFrames par DATETIME pour que merge_asof fonctionne correctement
            self.waiting_df = self.waiting_df.sort_values('DATETIME')
            self.weather_df = self.weather_df.sort_values('DATETIME')
            
            # Effectuez la fusion avec merge_asof qui trouve l'enregistrement météo le plus proche dans le temps
            # MAIS uniquement parmi les entrées du même jour grâce à 'by'
            self.merged_df = pd.merge_asof(
                self.waiting_df,          # Temps d'attente (gauche)
                self.weather_df,          # Données météo (droite)
                on='DATETIME',            # Fusionner sur l'horodatage complet
                by='DATE',                # Contrainte: uniquement le même jour
                direction='nearest'       # Stratégie: plus proche temporellement
            )
        else:
            self.merged_df = self.waiting_df.copy()
        
        # Drop rows with missing values in the target
        self.merged_df = self.merged_df.dropna(subset=['WAIT_TIME_IN_2H'])
        
        # Prepare features and target
        self._prepare_data()
    
    def _prepare_data(self):
        """Prepare features and target for the model."""
        # Features to use
        numerical_features = [
            'ADJUST_CAPACITY', 'DOWNTIME', 'CURRENT_WAIT_TIME',
            'HOUR', 'TIME_MINUTES', 'DAY', 'MONTH', 'YEAR'
        ]
        
        categorical_features = [
            'ENTITY_DESCRIPTION_SHORT', 'DAY_OF_WEEK'
        ]
        
        # Add weather features if included
        if self.include_weather:
            weather_features = [
                'temp', 'dew_point', 'feels_like', 'pressure',
                'humidity', 'wind_speed', 'clouds_all'
            ]
            numerical_features.extend(weather_features)
        
        # Handle time to parades and night show (may have NaNs)
        time_features = ['TIME_TO_PARADE_1', 'TIME_TO_PARADE_2', 'TIME_TO_NIGHT_SHOW']
        # Replace NaN with a large value to indicate no parade/show
        for feature in time_features:
            self.merged_df[feature] = self.merged_df[feature].fillna(999)
        
        numerical_features.extend(time_features)
        
        # Scale numerical features
        self.numerical_scaler = StandardScaler()
        numerical_data = self.numerical_scaler.fit_transform(
            self.merged_df[numerical_features].values
        )
        
        # Encode categorical features using one-hot encoding
        self.categorical_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        categorical_data = self.categorical_encoder.fit_transform(
            self.merged_df[categorical_features].values
        )
        
        # Combine features
        features = np.hstack([numerical_data, categorical_data])
        
        # Target: waiting time in 2 hours
        target = self.merged_df['WAIT_TIME_IN_2H'].values.reshape(-1, 1)
        
        # Save feature and target tensors
        self._input_tensor = features
        self._label_tensor = target
        
        # Save feature metadata for later use
        self.numerical_features = numerical_features
        self.categorical_features = categorical_features
    
    def show_sample(self, index, test=False):
        """
        Display information about a sample from the dataset as a clean pandas DataFrame.
        Shows both raw data and the encoded/normalized values.
        
        Args:
            index: Index of the sample to display
            test: Whether to use the test set (True) or training set (False)
        """
        # Get the data based on train/test set
        if test:
            input_features = self._input_tensor_test[index]
            target = self._label_tensor_test[index][0]
            dataset_slice = 'Test'
            # Get approximate index in original dataframe for test data
            data_index = index + int(len(self.merged_df) * 0.85)  # Assuming test is the last 15%
        else:
            input_features = self._input_tensor_train[index]
            target = self._label_tensor_train[index][0]
            dataset_slice = 'Train'
            # Get approximate index in original dataframe for train data
            data_index = index
        
        # Directly access the original dataframe row if possible
        try:
            if data_index < len(self.merged_df):
                # Get the actual row data from the merged dataframe
                original_data = self.merged_df.iloc[data_index]
                has_original_data = True
            else:
                has_original_data = False
        except:
            has_original_data = False
        
        # Retrieve the decoded numerical features
        num_numerical = len(self.numerical_features)
        numerical_values = self.numerical_scaler.inverse_transform([input_features[:num_numerical]])[0]
        
        # Create a feature dictionary for the processed data
        feature_dict = {}
        for i, feature in enumerate(self.numerical_features):
            feature_dict[feature] = numerical_values[i]
        
        # Decode categorical features 
        categorical_indices = input_features[num_numerical:]
        if hasattr(self.categorical_encoder, 'feature_names_in_'):
            categories = self.categorical_encoder.categories_
            for i, cat_feature in enumerate(self.categorical_features):
                cat_index = np.argmax(categorical_indices[
                    np.sum([len(c) for c in categories[:i]]):
                    np.sum([len(c) for c in categories[:i+1]])
                ])
                feature_dict[cat_feature] = categories[i][cat_index]
        
        # Get entity type for header
        if 'ENTITY_DESCRIPTION_SHORT' in feature_dict:
            entity_type = feature_dict['ENTITY_DESCRIPTION_SHORT']
        else:
            entity_type = "Unknown"
        
        # Print header with sample info
        print(f"\n{'-'*100}")
        print(f"{dataset_slice} Sample {index}: {entity_type} - Target Wait Time in 2 Hours: {target:.1f} minutes")
        print(f"{'-'*100}\n")
        
        # Create a unified DataFrame for display
        display_data = []
        
        # 1. Original Data (if available)
        if has_original_data:
            # Create a dictionary for the first row (original data)
            row = {'Type': 'Original Data'}
            
            # Add all original columns (format as needed)
            for col in original_data.index:
                if col == 'DATETIME':
                    row[col] = original_data[col].strftime('%Y-%m-%d %H:%M:%S')
                elif isinstance(original_data[col], (int, float, np.number)):
                    row[col] = f"{original_data[col]:.2f}"
                else:
                    row[col] = str(original_data[col])
            
            display_data.append(row)
        
        # 2. Processed Data
        processed_row = {'Type': 'Processed Data'}
        
        # Add processed features
        for feature in feature_dict:
            value = feature_dict[feature]
            if isinstance(value, (int, float, np.number)):
                processed_row[feature] = f"{value:.2f}"
            else:
                processed_row[feature] = str(value)
        
        # Add target explicitly
        processed_row['WAIT_TIME_IN_2H'] = f"{target:.2f}"
        display_data.append(processed_row)
        
        # 3. Encoded/Normalized Data
        encoded_row = {'Type': 'Encoded Data'}
        
        # Add normalized numerical features
        for i, feature in enumerate(self.numerical_features):
            encoded_row[feature] = f"{input_features[i]:.3f} (norm)"
        
        # Add one-hot categorical features
        cat_feature_offset = 0
        for i, cat_feature in enumerate(self.categorical_features):
            if hasattr(self.categorical_encoder, 'feature_names_in_'):
                categories = self.categorical_encoder.categories_[i]
                num_categories = len(categories)
                
                # Get one-hot values
                one_hot_values = input_features[num_numerical + cat_feature_offset:num_numerical + cat_feature_offset + num_categories]
                
                # Get the active category
                active_index = np.argmax(one_hot_values)
                active_category = categories[active_index] if active_index < len(categories) else "Unknown"
                
                # Create a readable representation for the one-hot encoding
                encoded_row[cat_feature] = f"{active_category} {np.array2string(one_hot_values, precision=1, separator=',')}"
                
                cat_feature_offset += num_categories
        
        display_data.append(encoded_row)
        
        # Create DataFrame and display it nicely
        display_df = pd.DataFrame(display_data)
        
        # Set the Type column as index for cleaner display
        display_df.set_index('Type', inplace=True)
        
        # Display the DataFrame
        # Use pandas styling options to make it look nicer
        print(display_df.to_string())
        print(f"\n{'-'*100}")
        
        return display_df  # Return the DataFrame for potential further use
