from nltk.metrics import agreement, ConfusionMatrix
import itertools
import csv

def load_annotators(target_annotators=None):
    """Load annotator names from CSV header"""
    if target_annotators is None:
        target_annotators = ["annotator_1", "annotator_2"]
    with open('../Annotated_datasets/Winobias_annotated.csv', 'r') as f:
        reader = csv.reader(f)
        header = [name.strip() for name in next(reader)]
        annotators = [name for name in header if name in target_annotators]
    return annotators


def _get_annotator_indices(header, annotators):
    return [header.index(a) for a in annotators if a in header]

def load_data(annotators):
    """Load data from CSV file"""
    data = []
    with open('../Annotated_datasets/Winobias_annotated.csv', 'r') as f:
        reader = csv.reader(f)
        header = [name.strip() for name in next(reader)]
        annotator_indices = _get_annotator_indices(header, annotators)
        for row_idx, row in enumerate(reader, start=1):
            datapoint = f"datapoint_{row_idx}"  # Use row number as datapoint identifier
            for idx in annotator_indices:
                answer = row[idx].strip() if idx < len(row) else ""
                if answer:  # Only add if answer is not empty
                    data.append([header[idx], datapoint, answer])
    return data

def get_annotator_answers(annotator_name, datapoints):
    annotator_answers = []
    for i,answers in datapoints.items():
        for annotator, answer in answers.items():
            if annotator == annotator_name:
                annotator_answers.append(answer)
    return annotator_answers

def get_datapoints(data):
    datapoints = {}
    for i in data:
        if i[1] not in datapoints:
            datapoints[i[1]] = {}
        datapoints[i[1]][i[0]] = i[2]
    return datapoints

def get_disagreements(datapoints):
    disagreements = {}
    for i,answers in datapoints.items():
        if len(set(answers.values())) > 1:
            disagreements[i] = answers
    return disagreements

def calculate_fleiss_kappa(datapoints, annotators):
    """
    Calculate Fleiss' kappa manually to verify correctness.
    Fleiss' kappa formula: κ = (P̄ - P̄e) / (1 - P̄e)
    Where:
      P̄ = average proportion of agreement across all items
      P̄e = expected proportion of agreement by chance
    """
    # Count annotations per category per item
    n_items = len(datapoints)
    n_annotators = len(annotators)
    
    # Get all unique categories
    all_categories = set()
    for answers in datapoints.values():
        all_categories.update(answers.values())
    categories = sorted(list(all_categories))
    
    # Count how many annotators assigned each category to each item
    n_ij = {}  # n_ij[i][category] = number of annotators who assigned category to item i
    for item, answers in datapoints.items():
        n_ij[item] = {}
        for cat in categories:
            n_ij[item][cat] = sum(1 for ans in answers.values() if ans == cat)
    
    # Calculate P̄ (average proportion of agreement)
    sum_pi = 0.0
    for item in datapoints.keys():
        pi = 0.0
        for cat in categories:
            n_cat = n_ij[item][cat]
            pi += n_cat * (n_cat - 1)
        if n_annotators > 1:
            pi = pi / (n_annotators * (n_annotators - 1))
        sum_pi += pi
    P_bar = sum_pi / n_items
    
    # Calculate P̄e (expected proportion of agreement by chance)
    # Count total assignments per category across all items
    p_j = {}  # p_j[category] = proportion of all assignments that are this category
    total_assignments = 0
    category_counts = {cat: 0 for cat in categories}
    
    for answers in datapoints.values():
        for ans in answers.values():
            category_counts[ans] += 1
            total_assignments += 1
    
    for cat in categories:
        p_j[cat] = category_counts[cat] / total_assignments if total_assignments > 0 else 0
    
    P_bar_e = sum(p_j[cat] ** 2 for cat in categories)
    
    # Calculate Fleiss' kappa
    if P_bar_e == 1.0:
        kappa = 1.0  # Perfect agreement
    else:
        kappa = (P_bar - P_bar_e) / (1 - P_bar_e)
    
    return kappa

annotators = load_annotators()
data = load_data(annotators)
datapoints = get_datapoints(data)

# Create annotation task with data in format: [annotator, datapoint, annotation]
# This allows NLTK to compute agreement statistics
task = agreement.AnnotationTask(data=data)

# Open file for detailed results
output_file = 'Kappa_scores/kappa_results_Winobias.txt'
with open(output_file, 'w') as f:
    f.write("="*60 + "\n")
    f.write("INTER-ANNOTATOR AGREEMENT ANALYSIS\n")
    f.write("="*60 + "\n\n")
    
    # Calculate pairwise kappa for each pair of annotators
    # Kappa formula: κ = (Po - Pe) / (1 - Pe)
    # Where:
    #   Po = Observed agreement (proportion of items where both annotators agree)
    #   Pe = Expected agreement by chance (based on each annotator's label distribution)
    for pair in itertools.combinations(annotators, 2):
        f.write("\n\n*** " + pair[0] + " vs " + pair[1] + " ***\n")
        
        # Observed agreement: actual proportion of agreement
        observed = task.Ao(pair[0],pair[1])
        f.write(f"\nObserved agreement: {observed}\n")
        
        # Expected agreement: what we'd expect by chance
        expected = task.Ae_kappa(pair[0],pair[1])
        f.write(f"Expected agreement: {expected}\n")
        
        # Cohen's Kappa: (observed - expected) / (1 - expected)
        # Range: -1 to 1, where:
        #   1 = perfect agreement
        #   0 = agreement equal to chance
        #   <0 = agreement worse than chance
        kappa = task.kappa_pairwise(pair[0],pair[1])
        f.write(f"Pairwise kappa (Cohen's): {kappa}\n")
        
        # Show confusion matrix: how annotations align between the two annotators
        a1 = get_annotator_answers(pair[0], datapoints)
        a2 = get_annotator_answers(pair[1], datapoints)
        cm = ConfusionMatrix(a1,a2)
        f.write("\nConfusion Matrix:\n")
        f.write(str(cm) + "\n")
    
    f.write("\n" + "="*60 + "\n")
    f.write("Overall Inter-Annotator Agreement:\n")
    f.write("="*60 + "\n")
    
    # Average pairwise kappa (k-bar)
    k_bar = task.kappa()
    f.write(f"k (k-bar / Average Pairwise Kappa): {k_bar}\n")
    f.write(f"\nInterpretation:\n")
    if k_bar < 0:
        interpretation = "Poor agreement (worse than chance)"
    elif k_bar < 0.20:
        interpretation = "Slight agreement"
    elif k_bar < 0.40:
        interpretation = "Fair agreement"
    elif k_bar < 0.60:
        interpretation = "Moderate agreement"
    elif k_bar < 0.80:
        interpretation = "Substantial agreement"
    else:
        interpretation = "Almost perfect agreement"
    f.write(f"  {interpretation}\n")
    
    # Fleiss' Kappa - supports multiple annotators
    f.write("\n" + "-"*60 + "\n")
    f.write("Fleiss' Kappa (Multi-Annotator Metric):\n")
    f.write("-"*60 + "\n")
    fleiss_kappa = calculate_fleiss_kappa(datapoints, annotators)
    fleiss_kappa_nltk = task.multi_kappa()
    f.write(f"Fleiss' k (Standard Formula): {fleiss_kappa}\n")
    f.write(f"Fleiss' k (NLTK variant): {fleiss_kappa_nltk}\n")
    f.write(f"Difference: {abs(fleiss_kappa - fleiss_kappa_nltk)}\n")
    f.write(f"\nWhy the difference?\n")
    f.write(f"  - Standard Fleiss' k: Calculates agreement across ALL annotators\n")
    f.write(f"    simultaneously for each item, then averages across items.\n")
    f.write(f"    This is the true multi-annotator metric.\n")
    f.write(f"  - NLTK's multi_kappa: Averages pairwise observed agreements\n")
    f.write(f"    and pairwise expected agreements separately, then applies\n")
    f.write(f"    kappa formula. This is essentially averaging pairwise metrics.\n")
    f.write(f"\nInterpretation:\n")
    if fleiss_kappa < 0:
        fleiss_interpretation = "Poor agreement (worse than chance)"
    elif fleiss_kappa < 0.20:
        fleiss_interpretation = "Slight agreement"
    elif fleiss_kappa < 0.40:
        fleiss_interpretation = "Fair agreement"
    elif fleiss_kappa < 0.60:
        fleiss_interpretation = "Moderate agreement"
    elif fleiss_kappa < 0.80:
        fleiss_interpretation = "Substantial agreement"
    else:
        fleiss_interpretation = "Almost perfect agreement"
    f.write(f"  {fleiss_interpretation}\n")
    f.write("="*60 + "\n")
    
    f.write("\n\nDisagreements:\n")
    f.write("-"*60 + "\n")
    disagreements = get_disagreements(datapoints)
    for d,answers in disagreements.items():
        f.write(f"{d}: {answers}\n")

# Print only summary to CLI
k_bar = task.kappa()
fleiss_kappa = calculate_fleiss_kappa(datapoints, annotators)
fleiss_kappa_nltk = task.multi_kappa()
disagreements = get_disagreements(datapoints)

print("="*60)
print("INTER-ANNOTATOR AGREEMENT ANALYSIS")
print("="*60)
print(f"k̄ (k-bar / Average Pairwise Kappa): {k_bar}")
print(f"Fleiss' κ (Standard Formula): {fleiss_kappa}")
print(f"Fleiss' κ (NLTK variant): {fleiss_kappa_nltk}")
print(f"\nDifferences:")
print(f"  k-bar vs Fleiss' κ (Standard): {abs(k_bar - fleiss_kappa)}")
print(f"  Standard vs NLTK variant: {abs(fleiss_kappa - fleiss_kappa_nltk)}")
print(f"\nNote: NLTK's multi_kappa averages pairwise metrics,")
print(f"      while standard Fleiss' κ is a true multi-annotator metric.")

# Print pairwise kappas summary
print("\nPairwise Kappa Values (Cohen's):")
for pair in itertools.combinations(annotators, 2):
    kappa = task.kappa_pairwise(pair[0],pair[1])
    print(f"  {pair[0]} vs {pair[1]}: {kappa}")
print(f"\nTotal datapoints: {len(datapoints)}")
print(f"Disagreements: {len(disagreements)}")
print(f"Agreement rate: {(len(datapoints)-len(disagreements))/len(datapoints)*100}%")
print(f"\nDetailed results saved to: {output_file}")
print("="*60)
