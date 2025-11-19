import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
# Mount venv: venv\Scripts\activate
# Freexze dependencies: pip freeze > requirements.txt
# Install dependencies: pip install -r requirements.txt  


df = pd.read_csv('emnist-letters-train.csv', header=None, dtype=np.float32)
print(f"Dataset shape: {df.shape}")


# First column (0) is the label, rest are pixels
# Add letter column
df['letter'] = df[0].apply(lambda x: chr(64 + int(x)))

# Rename first column for clarity
df = df.rename(columns={0: 'label'})

# Verify the mapping
print("\nLabel to Letter mapping (first 10):")
print(df[['label', 'letter']].drop_duplicates().head(10))


# Visualize some examples
# Visualize some examples
def show_letters(df, n=100):
    # Calculate grid size (10x10 for 100 images)
    rows = int(np.sqrt(n))
    cols = int(np.ceil(n / rows))
    
    fig, axes = plt.subplots(rows, cols, figsize=(20, 20))
    axes = axes.flatten()
    
    for i in range(n):
        # Get pixel columns (columns 1 to 784, excluding label and letter)
        # Make sure to convert to numeric and reshape
        pixel_values = df.iloc[i, 1:-1].values.astype(np.float32)
        pixels = pixel_values.reshape(28, 28)
        
        # EMNIST images may need rotation/flipping
        pixels = np.rot90(pixels, k=3)  # Rotate 270 degrees
        pixels = np.fliplr(pixels)       # Flip left-right
        
        axes[i].imshow(pixels, cmap='gray')
        axes[i].set_title(f"{df.iloc[i]['letter']}", 
                         fontsize=8, fontweight='bold')
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.show()

show_letters(df, 100)
