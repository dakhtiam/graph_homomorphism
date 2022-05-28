
# Proteins:

Testgraphs:
[Vertex, trees up to size 5, c_3, c_4, c_5, K_4, K_5]

clf = make_pipeline(StandardScaler(), RandomForestClassifier(max_leaf_nodes= 55, random_state = 21))
cv = 5
Validation error = 0.76±0.02

image: proteins1_may20

