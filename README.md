# auto_filter
Implementation of the automated filter selection method

Important Note: There was a mistake on our publication regarding the number of examples in the Drosophila dataset, used as an external validation for our model. In the paper, we say that there were 300 instances in this validation dataset, but this number actually included several compounds for which there were no known interactors on STITCH and others that already appeared in the training dataset (C. Elegans data). After removing those, the resulting dataset (which was the one used for the experiments reported in the paper) has 45 instances, referring to different compounds for which there is data on Drosophila Melanogaster but not for Caenorhabidtis Elegans. We apologize for the oversight.
