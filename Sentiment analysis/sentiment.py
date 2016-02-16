#label:	will be either pos if the text is determined to be positive, neg if the text is negative, or neutral if the text is neither pos nor neg.
#probability: an object that contains the probability for each label. neg and pos will add up to 1, while neutral is standalone. If neutral is greater than 0.5 then the label will be neutral. Otherwise, the label will be pos or neg, whichever has the greater probability.

import json
import requests

url = 'http://text-processing.com/api/sentiment/'

string = raw_input()

analysis_string = {'text': string}

result = requests.post(url, data=analysis_string)

negative_val = result.json()['probability']['neg']
print negative_val

positive_val = result.json()['probability']['pos']
print positive_val

neutral_val = result.json()['probability']['neutral']
print neutral_val

label = result.json()['label']
print label