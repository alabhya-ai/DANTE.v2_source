import pandas as pd
from sklearn.model_selection import train_test_split

# --- Configuration ---
# Input file names
X_INPUT_FILE = 'X.pkl'
Y_INPUT_FILE = 'y.pkl'

# Output file names
X_TRAIN_FILE = 'X_train.pkl'
Y_TRAIN_FILE = 'y_train.pkl'
X_TEST_FILE = 'X_test.pkl'
Y_TEST_FILE = 'y_test.pkl'

# Split ratio: 20% of data will be reserved for testing
TEST_SIZE = 0.2 
RANDOM_STATE = 42 # Set for reproducibility

def split_and_save_data():
    """
    Loads X and Y data, performs a stratified split, and saves the four resulting
    datasets to new pickle files.
    """
    print(f"Loading data from {X_INPUT_FILE} and {Y_INPUT_FILE}...")
    try:
        # Load X (action_id sequences - Series or DataFrame)
        X = pd.read_pickle(X_INPUT_FILE)
        # Load Y (malicious labels - Series or DataFrame)
        y = pd.read_pickle(Y_INPUT_FILE)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please ensure X.pkl and y.pkl are in the current directory.")
        return

    # --- Data Distribution Check (Before Split) ---
    malicious_count = y.sum()
    total_count = len(y)
    malicious_ratio = malicious_count / total_count
    
    print(f"Total samples: {total_count}")
    print(f"Malicious (1) samples: {malicious_count}")
    print(f"Malicious ratio: {malicious_ratio:.4f}")
    
    print(f"\nPerforming stratified split (Test Size: {TEST_SIZE * 100:.0f}%)...")

    # Use STRATIFY=y to ensure the malicious samples are split proportionally
    X_train, X_test, y_train, y_test = train_test_split(
        X, 
        y, 
        test_size=TEST_SIZE, 
        random_state=RANDOM_STATE, 
        stratify=y # CRITICAL: Maintains the class ratio in both train and test sets
    )

    # --- Save Split Data ---
    X_train.to_pickle(X_TRAIN_FILE)
    y_train.to_pickle(Y_TRAIN_FILE)
    X_test.to_pickle(X_TEST_FILE)
    y_test.to_pickle(Y_TEST_FILE)

    # --- Verification ---
    print("\nData Split Complete and Saved:")
    print(f" - {X_TRAIN_FILE}, {Y_TRAIN_FILE}: {len(X_train)} training samples")
    print(f" - {X_TEST_FILE}, {Y_TEST_FILE}: {len(X_test)} testing samples")
    
    # Verify stratification (Optional but recommended check)
    test_malicious_ratio = y_test.sum() / len(y_test)
    print(f" - Test set malicious ratio: {test_malicious_ratio:.4f} (Should match initial ratio)")

if __name__ == '__main__':
    split_and_save_data()