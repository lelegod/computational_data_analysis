import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# DTU Colors for plotting
DTU_RED = '#990000'
DTU_NAVY = '#00213E'

def info_leakage_audit():
    '''
    Simulating pure noise to catch Data Leakage.
    '''
    np.random.seed(42)
    N, M = 50, 1000
    # Create pure random noise
    X = np.random.randn(N, M)
    y = np.random.randn(N)

    print('--- Workflow A:Leakage ---')
    # TODO: Implement the 'Leaky' workflow
    # 1. Standardize the WHOLE dataset (X) using StandardScaler
    # 2. Calculate absolute correlation between each feature in X_scaled and y
    # 3. Select the indices of the top 10 features with highest correlation
    # 4. Create X_selected containing only these 10 features
    # 5. Split (X_selected, y) into 50/50 train and test sets
    # 6. Fit LinearRegression on training and print Test R^2
    
    # Placeholder for plot data (absolute correlations)
    corrs_a = np.zeros(10) 
    r2_a = 0.0 
    print(f'Workflow A (Leaky) Test R^2: {r2_a:.3f}')

    print('\n--- Workflow B: The Audit (No Leakage) ---')
    # TODO: Implement the 'Honest' workflow
    # 1. Split the original (X, y) into 50/50 train and test sets FIRST
    # 2. Fit a StandardScaler on X_train only and transform both X_train and X_test
    # 3. Calculate correlation between X_train features and y_train ONLY
    # 4. Select the top 10 features based on these training correlations
    # 5. Create subsets of X_train and X_test using these indices
    # 6. Fit LinearRegression on training subset and print Test R^2
    
    # Placeholder for plot data (absolute correlations)
    corrs_b = np.zeros(10)
    r2_b = 0.0
    print(f'Workflow B (non-leaky) Test R^2: {r2_b:.3f}')

    # --- VISUALIZATION OF THE EVIDENCE ---
    # TODO: Create a bar plot comparing the top 10 correlations for A and B
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.bar(range(10), corrs_a, color=DTU_RED)
    plt.title('Leaky Correlations', color=DTU_NAVY)
    plt.ylabel('Abs. Correlation with y')
    plt.ylim(0, 1)

    plt.subplot(1, 2, 2)
    plt.bar(range(10), corrs_b, color=DTU_NAVY)
    plt.title('Non-leaky Correlations (audited)', color=DTU_NAVY)
    plt.ylabel('Abs. Correlation with y')
    plt.ylim(0, 1)

    plt.tight_layout()
    plt.show()

    print('\nVERDICT:')
    print('If Workflow A gives high correlations on noise, why is it a scientific crime?')

if __name__ == '__main__':
    info_leakage_audit()