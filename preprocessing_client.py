# preprocessing_client_clean.py

import pandas as pd
import numpy as np
import joblib
import warnings
import os

warnings.filterwarnings('ignore')

class IoTDataPreprocessorClient:
    """
    Clean client preprocessor that uses server artifacts downloaded by websocket_service.py
    """
    
    def __init__(self, artifacts_path="artifacts/"):
        """
        Initialize with artifacts already downloaded by websocket service
        """
        self.artifacts_path = artifacts_path
        print(f"🔧 Loading server artifacts from: {artifacts_path}")
        
        # Load the NEW server artifacts format (ColumnTransformer)
        try:
            # Server's modern artifacts (what your server actually creates)
            self.preprocessor = joblib.load(os.path.join(artifacts_path, "preprocessor.pkl"))
            self.global_le = joblib.load(os.path.join(artifacts_path, "global_label_encoder.pkl"))
            self.feature_info = joblib.load(os.path.join(artifacts_path, "feature_info.pkl"))
            
            # Extract feature information
            self.numeric_features = self.feature_info['numeric_features']
            self.categorical_features = self.feature_info['categorical_features'] 
            self.feature_columns = self.numeric_features + self.categorical_features
            
            print(f"✅ Server artifacts loaded successfully!")
            print(f"   Numeric features: {len(self.numeric_features)}")
            print(f"   Categorical features: {len(self.categorical_features)}")
            print(f"   Total features: {len(self.feature_columns)}")
            
        except FileNotFoundError as e:
            print(f"❌ Server artifacts not found: {e}")
            print("💡 Make sure websocket_service.py downloaded the new artifacts!")
            raise
        except Exception as e:
            print(f"❌ Error loading server artifacts: {e}")
            raise

    def preprocess_data(self, path="client_local_dataset.csv", target_col='Attack_type'):
        """
        Transform client data using server's exact preprocessing pipeline
        """
        print(f"🔧 Processing client data: {path}")
        print("="*60)
        
        # Load client data
        df = pd.read_csv(path, low_memory=False)
        
        if target_col not in df.columns:
            raise ValueError(f"{target_col} column not found in client dataset!")
        
        print(f"Original client shape: {df.shape}")
        
        # Apply server's preprocessing logic exactly
        print("🔧 Applying server preprocessing pipeline...")
        
        # 1) Handle missing values (same as server)
        print("  → Handling missing values...")
        for col in self.numeric_features:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
                if df[col].isna().sum() > 0:
                    df[col].fillna(df[col].median(), inplace=True)
        
        for col in self.categorical_features:
            if col in df.columns:
                df[col] = df[col].astype(str)
                df[col] = df[col].replace(['nan', 'None', ''], 'Unknown')
        
        # 2) Ensure all server features exist
        print("  → Aligning features with server...")
        for col in self.feature_columns:
            if col not in df.columns:
                if col in self.numeric_features:
                    df[col] = 0.0  # Default numeric
                else:
                    df[col] = 'Unknown'  # Default categorical
                print(f"    Added missing feature: {col}")
        
        # 3) Apply server's ColumnTransformer
        try:
            feature_df = df[self.feature_columns].copy()
            X_processed = self.preprocessor.transform(feature_df)
            
            # Clean any numerical issues
            if np.any(np.isnan(X_processed)) or np.any(np.isinf(X_processed)):
                print("  → Cleaning NaN/Inf values...")
                X_processed = np.nan_to_num(X_processed, nan=0.0, posinf=0.0, neginf=0.0)
            
            print(f"  ✅ Features transformed: {feature_df.shape} → {X_processed.shape}")
            
        except Exception as e:
            print(f"  ❌ Preprocessing failed: {e}")
            raise
        
        # 4) Process target with server's label encoder
        print("  → Processing target variable...")
        try:
            # Server's global class names
            global_class_names = [
                'Backdoor', 'DDoS_HTTP', 'DDoS_ICMP', 'DDoS_TCP', 'DDoS_UDP',
                'Fingerprinting', 'MITM', 'Normal', 'Password', 'Port_Scanning',
                'Ransomware', 'SQL_injection', 'Uploading', 'Vulnerability_scanner', 'XSS'
            ]
            
            # Map unknown classes to 'Normal' (server's default)
            df[target_col] = df[target_col].apply(
                lambda x: x if x in global_class_names else 'Normal'
            )
            
            # Use server's label encoder
            y_encoded = self.global_le.transform(df[target_col].values)
            num_classes = len(np.unique(y_encoded))
            
            print(f"  ✅ Target processed: {num_classes} classes")
            
        except Exception as e:
            print(f"  ❌ Target processing failed: {e}")
            raise
        
        # Final summary
        print(f"\n✅ Client preprocessing complete!")
        print(f"   Final shape: {X_processed.shape}")
        print(f"   Feature range: [{X_processed.min():.4f}, {X_processed.max():.4f}]")
        print(f"   Classes found: {num_classes}")
        print(f"   Class distribution: {np.bincount(y_encoded)}")
        
        return X_processed, y_encoded, num_classes

    def get_feature_info(self):
        """Get server feature configuration"""
        return {
            'numeric_features': self.numeric_features,
            'categorical_features': self.categorical_features,
            'total_features': len(self.feature_columns)
        }
