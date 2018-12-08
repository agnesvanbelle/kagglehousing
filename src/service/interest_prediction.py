class InterestPrediction(object):
    
  def __init__(self, prob_list, index_to_class):
    self.prob_list = list(prob_list)
    self.index_to_class = index_to_class
    
  def serialize(self):
    d = {self.index_to_class[i] : float(self.prob_list[i]) for i in range(len(self.prob_list))}
    most_likely_class = self.index_to_class[self.prob_list.index(max(self.prob_list))]
    
    print("most likely class: ", most_likely_class)
    print("d:", d)
    
    return {
         "interest": {
          "most_likely_level" : most_likely_class,
          "probabilities_per_level": d,
         }
      }
