from analyze_data import analyze_edf
from classifiers.parafac import process as parafac_classifier
from classifiers.flat import process as csp_classifier
from visualization.accuracy_over_bands import visualize_accuracy_over_bands


accuracy_data = analyze_edf(classifier=parafac_classifier, verbose='ERROR')
visualize_accuracy_over_bands(accuracy_data)
