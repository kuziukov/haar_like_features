import pickle
from modules.Classifier import Classifier

file_name = 'resources/features.p'
save_to_file = 'resources/model'

x = pickle.load(open(file_name,'rb'))


classifier = Classifier(n_estimators=200, max_depth=2)

print ('-----> Training')
classifier.train(x['input'], x['labels'])
print ('-----> Training Complete')


pickle.dump(classifier, open(save_to_file, 'wb'))

