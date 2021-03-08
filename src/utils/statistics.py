from modules.Evaluator import Evaluator


def statistics(output_file_prefix='', num_images=100):
		""" Reports number of people, matches, misses, and false positives per experiment from generated output files """

		#=====[ Instantiate our evaluator ]=====
		evaluator = Evaluator('dataset/INRIAPerson/Test')

		num_people, num_hits, num_misses, num_FP, num_processed = evaluator.aggregate_stats(output_file_prefix, num_images)

		#=====[ Print statistics ]=====
		print('-----> Stats for ' + output_file_prefix + ' ( ' + str(num_processed) + '/' + str(num_images) +' processed) :\n\n')
		print('Miss Rate: ' + str(float(num_misses)/num_people))
		print('False Positives: ' + str(num_FP))
		print('FPPI: ' + str(float(num_FP)/num_images))
		print('Hits: ' + str(num_hits))
		print('Misses: ' + str(num_misses))
		print('Total People: ' + str(num_people))