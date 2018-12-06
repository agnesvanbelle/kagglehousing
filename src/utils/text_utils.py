from html.parser import HTMLParser
import string
import re
import unicodedata

whitespace_regexp = re.compile(r'\s+')

# strips tags
class MLStripper(HTMLParser):
  def __init__(self):
    super().__init__()
    self.reset()
    self.fed = []
  def handle_data(self, d):
    self.fed.append(d)
  def get_data(self):
    return self.fed

def strip_tags(html):
  s = MLStripper()
  s.feed(html)
  data_list = s.get_data()
  data_string = ''.join(data_list)
  data_string = whitespace_regexp.sub(' ', data_string)
  return data_string

def remove_punct_except_commas(txt):
  return remove_punct(txt, keep_commas = True)

def remove_punct(txt):
  s = string.punctuation + "—"
  s2 = re.escape(s)
  s2 = '[' + s2 +  ']'
  txt = re.sub(s2, " ", txt)
  return txt

def remove_numbers(txt):
  return re.sub('\d+', '', txt)

def remove_accents(s):
  if type(s) != str:
    s=str(s, 'utf-8')
  new_s = ''.join(c for c in unicodedata.normalize('NFD', s) if unicodedata.category(c) != 'Mn')
  if type(new_s) != str:
    new_s = str(new_s, 'utf-8')
  return new_s

def remove_excess_spaces(txt): 
  return  ''.join(re.sub(r"\s+", " ", txt).splitlines()).strip()

def clean_text(txt):
  return (remove_excess_spaces(remove_numbers(remove_accents(remove_punct(strip_tags(txt.lower())))))
          if txt != None else '')

if __name__ == "__main__":
  print(string.punctuation.replace(",", ""))
  
  