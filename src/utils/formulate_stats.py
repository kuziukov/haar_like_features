

def _formulate_stats(n_pos, n_neg, pos, neg):

	#=====[ Calculate False/True Pos/Neg ]=====
	FN= n_pos*(1-pos)
	TP = n_pos - FN
	accuracy = pos*n_pos/(n_pos+n_neg)

	FP = n_neg*(1-neg)
	TN = n_neg - FP
	accuracy += neg*n_neg/(n_pos + n_neg)

	#=====[ Calculate precision, recall, and f1 ]=====
	precision = float(TP)/(TP+FP)
	recall = float(TP)/(TP+FN)

	f1 = 2*precision*recall/(precision+recall)

	return (accuracy, f1)