import numpy as np
word_dict = {0:'A',1:'B',2:'C',3:'D',4:'E',5:'F',6:'G',7:'H',8:'I',9:'J',10:'K',11:'L',12:'M',
             13:'N',14:'O',15:'P',16:'Q',17:'R',18:'S',19:'T',20:'U',21:'V',22:'W',23:'X', 24:'Y',25:'Z'}
word_dict2 = inv_map = {v: k for k, v in word_dict.items()}

gt_dataset0 = [
'K', 'R', 'G', 'D', 'Y', 'C', 'V', 'N', 'S', 'L', 'Q', 'R', 'N', 'M', 'L', 'L', 'G', 'U', 'Z', 'Z',
'B', 'B', 'T', 'S', 'T', 'W', 'U', 'O', 'G', 'V', 'J', 'V', 'V', 'R', 'G', 'F', 'R', 'A', 'W', 'W',
'Y', 'B', 'Y', 'H', 'L', 'N', 'K', 'S', 'S', 'C', 'O', 'V', 'H', 'S', 'T', 'B', 'G', 'I', 'R', 'N',
'T', 'X', 'B', 'C', 'E', 'I', 'I', 'R', 'W', 'M', 'I', 'S', 'A', 'M', 'M', 'O', 'T', 'H', 'X', 'A',
'A', 'Q', 'Y', 'L', 'U', 'A', 'E', 'E', 'R', 'X', 'K', 'P', 'I', 'E', 'H', 'V', 'F', 'F', 'G', 'T',
'K', 'K', 'H', 'W', 'M', 'V', 'S', 'M', 'P', 'R', 'V', 'C', 'Y', 'W', 'Y', 'N', 'R', 'G', 'V', 'H',
'H', 'G', 'U', 'P', 'A', 'B', 'A', 'E', 'F', 'G', 'Y', 'M', 'B', 'Q', 'E', 'A', 'H', 'Z', 'E', 'Y',
'I', 'O', 'J', 'F', 'S', 'Z', 'M', 'L', 'Z', 'R', 'F', 'O', 'C', 'D', 'T', 'U', 'V', 'G', 'V', 'E',
'Q', 'F', 'U', 'U', 'A', 'A', 'P', 'N', 'L', 'U', 'M', 'T', 'L', 'U', 'P', 'J', 'C', 'T', 'O', 'Z',
'F', 'X', 'J', 'F', 'V', 'C', 'O', 'T', 'L', 'E', 'V', 'Z', 'T', 'X', 'R', 'R', 'R', 'N', 'J', 'D',
'S', 'C', 'A', 'N', 'D', 'E', 'L', 'A', 'L', 'T', 'T', 'S', 'E', 'T', 'P', 'R', 'W', 'O', 'A', 'G',
'X', 'R', 'H', 'X', 'T', 'S', 'Z', 'I', 'Y', 'O', 'I', 'N', 'S', 'K', 'Z', 'W', 'K', 'A', 'Q', 'Q',
'J', 'E', 'M', 'Y', 'E', 'G', 'C', 'Y', 'E', 'T', 'L', 'H', 'O', 'A', 'M', 'W', 'M', 'N', 'H', 'A',
'R', 'K', 'C', 'Z', 'X', 'L', 'J', 'K', 'A', 'S', 'F', 'J', 'W', 'M', 'K', 'E', 'Z', 'T', 'X', 'N',
'X', 'B', 'S', 'J', 'J', 'J', 'D', 'B', 'P', 'X', 'K', 'G', 'Q', 'I', 'Y', 'I', 'F', 'O', 'P', 'I',
'T', 'O', 'Q', 'E', 'L', 'M', 'X', 'U', 'M', 'Z', 'S', 'U', 'T', 'N', 'G', 'P', 'R', 'N', 'E', 'B',
'V', 'Q', 'U', 'Y', 'Q', 'K', 'Z', 'P', 'T', 'Z', 'E', 'T', 'K', 'F', 'Z', 'P', 'Y', 'I', 'H', 'E',
'S', 'D', 'V', 'M', 'S', 'F', 'U', 'V', 'O', 'K', 'O', 'S', 'R', 'I', 'R', 'U', 'B', 'O', 'O', 'Z',
'A', 'A', 'T', 'O', 'H', 'A', 'M', 'E', 'W', 'T', 'Y', 'I', 'M', 'L', 'O', 'W', 'T', 'L', 'B', 'G',
'B', 'U', 'G', 'V', 'U', 'H', 'J', 'Z', 'H', 'E', 'Z', 'H', 'R', 'A', 'T', 'T', 'P', 'I', 'N', 'Q'
]

count = np.zeros(26, dtype='int')
for i in gt_dataset0:
    count[word_dict2[i]] += 1
print(count)


