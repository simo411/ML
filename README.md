# Clustering Analysis on MNIST Dataset

This project demonstrates clustering analysis on the MNIST dataset using K-means, Gaussian Mixture Model (GMM), and DBSCAN algorithms. The code includes purity score evaluation and data visualization.

## Requirements
- Python 3.7 or higher
- pandas
- numpy
- matplotlib
- scikit-learn
- scipy

## Usage
1. Install the required Python packages using pip: `pip install pandas numpy matplotlib scikit-learn scipy`
2. Ensure you have the MNIST dataset files 'mnist-tsne-train.csv' and 'mnist-tsne-test.csv' in the same directory as the code.
3. Run the Python script in your terminal or IDE.
4. The code will perform K-means, GMM, and DBSCAN clustering on the training and testing data and display the results.

## Code Explanation
- The code reads the training and testing datasets.
- Implements K-means clustering with K=10 for the training data and computes purity score.
- Plots the data points and cluster centers for K-means on training data.
- Implements GMM clustering with K=10 for the training data and computes purity score.
- Plots the data points and cluster centers for GMM on training data.
- Implements DBSCAN clustering on the training data using different values of epsilon and min_samples, and computes purity score.
- Plots the data points for DBSCAN on training data.
- The code also includes a bonus section that investigates the optimal value of K for K-means and GMM algorithms using an elbow plot.

## Results
- The results for K-means and GMM clustering include data visualization and purity scores for both training and testing data.
- The results for DBSCAN clustering include data visualization and purity scores for different values of epsilon and min_samples.

## Bonus Section
- The bonus section explores the optimal value of K for K-means and GMM algorithms using an elbow plot.

## Conclusion
- The clustering analysis demonstrates the performance of K-means, GMM, and DBSCAN algorithms on the MNIST dataset.
- Discuss the findings and insights gained from the clustering analysis.
- Note any limitations or possible future improvements.
