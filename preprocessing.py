import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
import warnings
warnings.filterwarnings('ignore')

class IoTDataPreprocessor:
    def __init__(self):
        # Categorical columns to encode (if any remain after feature selection)
        self.categorical_columns = [
            'http.request.method', 'dns.qry.qu', 'dns.qry.type',
            'mqtt.msg_decoded_as', 'mqtt.protoname'
        ]
        self.le_dict = {}

    def preprocess_data(self, path="selected_features_dataset.csv"):
        """
        Preprocess data that has already been through feature selection.
        This assumes the CSV contains only selected features + Attack_type column.
        """
        print(f"Loading feature-selected dataset: {path}\n{'='*70}")
        df = pd.read_csv(path)
        
        print("DataFrame Info:\n" + "-"*30)
        print(f"Shape: {df.shape}")
        print(f"Missing Values: {df.isna().sum().sum()} ({df.isna().sum().sum() / df.size * 100:.2f}%)")
        print(f"Duplicate Rows: {df.duplicated().sum()}")
        
        # Verify Attack_type column exists
        if 'Attack_type' not in df.columns:
            raise ValueError("Attack_type column not found in the dataset!")
        
        # Handle any remaining missing values
        print("\nHandling missing values...")
        for col in df.columns:
            if col in self.categorical_columns:
                df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else 'Unknown', inplace=True)
            else:
                df[col].fillna(0, inplace=True)
        
        # Encode any remaining categorical columns
        print("\nEncoding categorical columns...")
        for col in self.categorical_columns:
            if col in df.columns and df[col].dtype == 'object':
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col].astype(str))
                self.le_dict[col] = le
                print(f"Encoded {col} with {len(le.classes_)} unique values")
        
        # Encode Attack_type if it's not already encoded
        if df['Attack_type'].dtype == 'object':
            le = LabelEncoder()
            df['Attack_type'] = le.fit_transform(df['Attack_type'])
            print(f"Encoded Attack_type with {len(le.classes_)} unique values")
        
        # Separate features and target
        feature_columns = [col for col in df.columns if col != 'Attack_type']
        feature_df = df[feature_columns].copy()
        y_multiclass = df['Attack_type'].copy()
        
        # Ensure all feature columns are numeric
        print("\nEnsuring all features are numeric...")
        for col in feature_columns:
            if df[col].dtype == 'object' or df[col].dtype == 'string':
                print(f"Converting {col} to numeric...")
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        
        # Double-check: remove any columns that still can't be converted
        non_numeric_final = feature_df.select_dtypes(exclude=[np.number]).columns
        if len(non_numeric_final) > 0:
            print(f"Dropping final non-numeric columns: {non_numeric_final.tolist()}")
            feature_df.drop(columns=non_numeric_final, inplace=True)
        
        print(f"\nClass distribution:")
        unique, counts = np.unique(y_multiclass, return_counts=True)
        for cls, count in zip(unique, counts):
            print(f"Class {cls}: {count} samples")
        
        # Feature scaling
        print("\nApplying StandardScaler...")
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(feature_df)
        
        X_resampled, y_resampled = X_scaled, y_multiclass

        unique, counts = np.unique(y_resampled, return_counts=True)
        for cls, count in zip(unique, counts):
            print(f"Class {cls}: {count} samples")
        
        num_classes = len(np.unique(y_resampled))
        
        print(f"\n✅ Preprocessing complete!")
        print(f"Final shape: {X_resampled.shape}")
        print(f"Number of classes: {num_classes}")
        
        return X_resampled, y_resampled, num_classes

# Usage example
# if __name__ == "__main__":
#     preprocessor = IoTDataPreprocessor()
    
#     # Process the feature-selected dataset
#     X, y, num_classes, scaler, features = preprocessor.preprocess_data(
#         "selected_features_dataset.csv",
#         apply_smote=False  # Set to True if you want SMOTE
#     )
    
#     print(f"\nFinal Results:")
#     print(f"Features shape: {X.shape}")
#     print(f"Labels shape: {y.shape}")
#     print(f"Selected features count: {len(features)}")
