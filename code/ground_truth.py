import numpy as np
word_dict = {0:'A',1:'B',2:'C',3:'D',4:'E',5:'F',6:'G',7:'H',8:'I',9:'J',10:'K',11:'L',12:'M',
             13:'N',14:'O',15:'P',16:'Q',17:'R',18:'S',19:'T',20:'U',21:'V',22:'W',23:'X', 24:'Y',25:'Z'}
word_dict2 = {v: k for k, v in word_dict.items()}

#Matrius ground truth

#example2
example2_gt = [
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


#image6
image6_gt = [
 'R', 'A', 'M', 'U', 'F', 'L', 'A', 'L', 'E', 'V', 'A', 'N', 'T', 'A', 'R', 'A', 'O', 'Z', 'H', 'M',
 'E', 'T', 'R', 'O', 'N', 'A', 'P', 'A', 'L', 'O', 'S', 'M', 'O', 'D', 'O', 'R', 'R', 'A', 'E', 'O',
 'G', 'U', 'U', 'A', 'S', 'N', 'E', 'R', 'E', 'N', 'A', 'G', 'I', 'D', 'R', 'A', 'C', 'N', 'M', 'D',
 'O', 'N', 'G', 'D', 'R', 'T', 'N', 'B', 'T', 'R', 'P', 'N', 'N', 'O', 'R', 'O', 'R', 'E', 'A', 'A',
 'C', 'U', 'O', 'A', 'O', 'E', 'A', 'E', 'C', 'I', 'O', 'O', 'R', 'E', 'M', 'O', 'C', 'T', 'T', 'C',
 'I', 'M', 'S', 'M', 'D', 'O', 'S', 'C', 'A', 'A', 'I', 'I', 'P', 'E', 'L', 'S', 'A', 'S', 'O', 'I',
 'G', 'E', 'O', 'I', 'I', 'J', 'D', 'D', 'E', 'S', 'U', 'U', 'R', 'L', 'I', 'P', 'I', 'C', 'I', 'H',
 'A', 'R', 'R', 'A', 'T', 'O', 'O', 'A', 'N', 'R', 'T', 'S', 'A', 'A', 'R', 'T', 'U', 'R', 'O', 'C',
 'L', 'A', 'E', 'T', 'R', 'R', 'L', 'E', 'N', 'I', 'A', 'R', 'A', 'E', 'S', 'L', 'R', 'R', 'I', 'A',
 'A', 'L', 'T', 'M', 'U', 'E', 'S', 'A', 'V', 'I', 'T', 'L', 'M', 'N', 'E', 'I', 'A', 'S', 'C', 'L',
 'N', 'R', 'N', 'E', 'S', 'I', 'F', 'C', 'C', 'P', 'M', 'I', 'G', 'P', 'T', 'R', 'M', 'T', 'I', 'A',
 'I', 'A', 'E', 'N', 'D', 'O', 'B', 'L', 'I', 'S', 'A', 'U', 'D', 'E', 'I', 'E', 'I', 'E', 'F', 'H',
 'M', 'C', 'V', 'H', 'A', 'M', 'B', 'R', 'E', 'R', 'I', 'S', 'L', 'O', 'B', 'V', 'R', 'N', 'O', 'C',
 'A', 'O', 'T', 'I', 'F', 'A', 'R', 'G', 'N', 'J', 'I', 'R', 'O', 'I', 'O', 'R', 'R', 'E', 'B', 'U',
 'Z', 'T', 'U', 'R', 'N', 'O', 'V', 'I', 'L', 'L', 'O', 'O', 'B', 'O', 'R', 'G', 'A', 'N', 'O', 'D'
]


#image4
image4_gt = [
 'O', 'I', 'R', 'A', 'V', 'I', 'U', 'Q', 'S', 'E', 'M', 'A', 'C', 'U', 'L', 'A', 'R', 'G', 'A', 'R',
 'A', 'R', 'U', 'T', 'U', 'S', 'A', 'R', 'E', 'N', 'I', 'L', 'O', 'M', 'E', 'B', 'A', 'D', 'M', 'N',
 'P', 'A', 'T', 'O', 'A', 'A', 'C', 'I', 'P', 'A', 'C', 'I', 'P', 'N', 'E', 'J', 'E', 'D', 'U', 'O',
 'U', 'R', 'O', 'N', 'C', 'R', 'R', 'M', 'C', 'O', 'P', 'E', 'V', 'R', 'A', 'F', 'R', 'O', 'L', 'D',
 'R', 'E', 'R', 'I', 'H', 'R', 'I', 'A', 'N', 'U', 'N', 'O', 'L', 'A', 'E', 'C', 'B', 'S', 'T', 'I',
 'E', 'E', 'E', 'A', 'E', 'O', 'A', 'F', 'E', 'C', 'C', 'U', 'L', 'C', 'N', 'A', 'E', 'I', 'I', 'P',
 'P', 'S', 'C', 'R', 'P', 'N', 'O', 'Z', 'A', 'T', 'M', 'A', 'C', 'I', 'L', 'I', 'L', 'S', 'P', 'T',
 'E', 'O', 'I', 'E', 'A', 'R', 'O', 'J', 'E', 'I', 'R', 'I', 'A', 'E', 'S', 'A', 'D', 'C', 'L', 'I',
 'R', 'P', 'P', 'V', 'T', 'O', 'A', 'L', 'N', 'I', 'O', 'O', 'U', 'I', 'I', 'E', 'I', 'O', 'E', 'C',
 'I', 'T', 'A', 'O', 'I', 'R', 'I', 'A', 'O', 'N', 'P', 'Z', 'S', 'N', 'N', 'L', 'M', 'N', 'S', 'O',
 'T', 'R', 'L', 'C', 'C', 'I', 'R', 'C', 'O', 'I', 'R', 'S', 'H', 'E', 'A', 'A', 'I', 'I', 'E', 'O',
 'O', 'A', 'O', 'A', 'O', 'I', 'D', 'O', 'R', 'A', 'V', 'A', 'E', 'C', 'R', 'R', 'Z', 'B', 'A', 'V',
 'G', 'Z', 'N', 'R', 'A', 'G', 'U', 'F', 'Z', 'E', 'L', 'B', 'A', 'D', 'M', 'I', 'R', 'A', 'R', 'R',
 'A', 'A', 'O', 'D', 'A', 'I', 'P', 'O', 'C', 'A', 'H', 'Y', 'O', 'E', 'N', 'E', 'G', 'E', 'G', 'A',
 'F', 'R', 'M', 'O', 'R', 'E', 'C', 'U', 'R', 'C', 'O', 'L', 'O', 'R', 'E', 'N', 'R', 'O', 'H', 'P'
]


#image3
image3_gt = [
 'R', 'A', 'L', 'I', 'N', 'E', 'A', 'R', 'C', 'U', 'O', 'T', 'A', 'O', 'S', 'O', 'P', 'A', 'R', 'O',
 'E', 'E', 'A', 'C', 'O', 'R', 'T', 'A', 'F', 'R', 'I', 'O', 'A', 'J', 'E', 'O', 'D', 'I', 'A', 'C',
 'G', 'R', 'N', 'A', 'T', 'S', 'E', 'L', 'L', 'A', 'B', 'D', 'B', 'T', 'O', 'E', 'T', 'I', 'U', 'T',
 'A', 'C', 'A', 'A', 'P', 'D', 'A', 'N', 'E', 'Y', 'A', 'C', 'N', 'A', 'I', 'N', 'O', 'L', 'A', 'V',
 'Z', 'O', 'C', 'T', 'A', 'L', 'A', 'P', 'E', 'G', 'O', 'E', 'A', 'N', 'B', 'V', 'J', 'P', 'A', 'R',
 'O', 'I', 'A', 'N', 'S', 'G', 'E', 'R', 'N', 'O', 'N', 'A', 'S', 'R', 'E', 'U', 'O', 'O', 'A', 'S',
 'R', 'D', 'B', 'E', 'E', 'T', 'M', 'E', 'S', 'I', 'O', 'A', 'I', 'N', 'O', 'N', 'I', 'T', 'L', 'S',
 'A', 'E', 'A', 'I', 'A', 'S', 'R', 'O', 'M', 'E', 'L', 'T', 'G', 'M', 'P', 'F', 'N', 'N', 'I', 'I',
 'D', 'M', 'R', 'T', 'R', 'E', 'S', 'E', 'R', 'I', 'N', 'A', 'A', 'E', 'A', 'E', 'A', 'D', 'O', 'D',
 'O', 'F', 'E', 'E', 'M', 'A', 'L', 'U', 'V', 'T', 'N', 'A', 'N', 'C', 'I', 'N', 'E', 'N', 'O', 'E',
 'O', 'N', 'A', 'I', 'N', 'E', 'K', 'A', 'O', 'Z', 'E', 'D', 'D', 'B', 'A', 'C', 'U', 'T', 'A', 'M',
 'M', 'A', 'N', 'A', 'G', 'E', 'R', 'N', 'A', 'M', 'U', 'C', 'M', 'L', 'A', 'S', 'A', 'S', 'R', 'O',
 'A', 'A', 'R', 'E', 'C', 'A', 'B', 'A', 'U', 'L', 'Q', 'A', 'I', 'R', 'E', 'R', 'E', 'C', 'T', 'N',
 'R', 'S', 'A', 'N', 'D', 'E', 'Z', 'R', 'A', 'P', 'I', 'F', 'A', 'N', 'E', 'C', 'A', 'D', 'O', 'I',
 'T', 'R', 'U', 'J', 'A', 'L', 'O', 'R', 'P', 'U', 'L', 'S', 'O', 'S', 'O', 'J', 'U', 'L', 'P', 'O'
]


#image2
image2_gt = [
 'B', 'N', 'I', 'L', 'A', 'S', 'A', 'S', 'N', 'H', 'O', 'B', 'P', 'C', 'O', 'E', 'L', 'O', 'V', 'E',
 'E', 'D', 'O', 'M', 'I', 'N', 'O', 'I', 'I', 'A', 'L', 'B', 'E', 'I', 'O', 'R', 'D', 'A', 'L', 'A',
 'R', 'D', 'E', 'L', 'A', 'S', 'T', 'G', 'Z', 'O', 'M', 'L', 'O', 'R', 'A', 'I', 'S', 'E', 'U', 'P',
 'I', 'A', 'I', 'U', 'E', 'E', 'O', 'E', 'N', 'L', 'O', 'I', 'E', 'P', 'R', 'N', 'C', 'T', 'A', 'R',
 'L', 'N', 'G', 'O', 'L', 'G', 'D', 'D', 'A', 'S', 'R', 'N', 'A', 'R', 'E', 'T', 'O', 'D', 'A', 'E',
 'I', 'I', 'D', 'O', 'R', 'U', 'A', 'R', 'O', 'O', 'E', 'S', 'A', 'F', 'O', 'R', 'A', 'L', 'P', 'C',
 'O', 'S', 'B', 'E', 'R', 'E', 'E', 'L', 'S', 'V', 'T', 'B', 'O', 'R', 'I', 'C', 'L', 'N', 'A', 'I',
 'C', 'A', 'C', 'E', 'S', 'N', 'T', 'E', 'F', 'I', 'O', 'G', 'A', 'D', 'E', 'I', 'O', 'E', 'L', 'S',
 'A', 'K', 'A', 'O', 'E', 'O', 'C', 'S', 'L', 'F', 'N', 'L', 'A', 'D', 'T', 'L', 'I', 'P', 'A', 'A',
 'R', 'I', 'N', 'G', 'R', 'C', 'R', 'L', 'A', 'I', 'P', 'D', 'N', 'N', 'I', 'N', 'M', 'O', 'D', 'R',
 'A', 'S', 'T', 'S', 'A', 'I', 'A', 'D', 'R', 'N', 'I', 'E', 'U', 'B', 'O', 'S', 'O', 'T', 'A', 'L',
 'C', 'T', 'O', 'U', 'L', 'R', 'E', 'G', 'E', 'O', 'V', 'P', 'A', 'P', 'A', 'Q', 'U', 'E', 'T', 'E',
 'O', 'E', 'R', 'M', 'R', 'A', 'P', 'U', 'A', 'N', 'O', 'B', 'O', 'T', 'E', 'L', 'L', 'A', 'N', 'I',
 'L', 'S', 'A', 'N', 'U', 'D', 'I', 'A', 'M', 'E', 'T', 'R', 'O', 'I', 'C', 'R', 'E', 'T', 'U', 'F',
 'A', 'T', 'L', 'T', 'B', 'V', 'A', 'R', 'A', 'D', 'A', 'A', 'L', 'L', 'I', 'N', 'A', 'L', 'J', 'O'
]

#image8
image8_gt = [
 'M', 'E', 'Z', 'C', 'L', 'A', 'R', 'O', 'S', 'T', 'N', 'E', 'M', 'A', 'L', 'C', 'I', 'C', 'I', 'B',
 'E', 'S', 'A', 'H', 'I', 'N', 'D', 'U', 'R', 'O', 'E', 'T', 'I', 'F', 'N', 'O', 'C', 'R', 'O', 'M',
 'L', 'P', 'D', 'C', 'A', 'R', 'T', 'E', 'R', 'A', 'L', 'O', 'G', 'N', 'E', 'O', 'D', 'U', 'E', 'D',
 'O', 'A', 'U', 'N', 'A', 'A', 'M', 'S', 'U', 'B', 'L', 'E', 'V', 'A', 'R', 'E', 'A', 'M', 'A', 'R',
 'N', 'T', 'A', 'C', 'M', 'O', 'S', 'E', 'D', 'O', 'S', 'O', 'R', 'I', 'E', 'N', 'T', 'A', 'L', 'P',
 'A', 'O', 'I', 'T', 'L', 'N', 'J', 'U', 'N', 'T', 'A', 'R', 'D', 'U', 'T', 'I', 'T', 'C', 'E', 'R',
 'R', 'R', 'A', 'I', 'N', 'I', 'O', 'C', 'O', 'R', 'D', 'E', 'R', 'O', 'E', 'U', 'Q', 'R', 'A', 'F',
 'A', 'N', 'T', 'F', 'V', 'E', 'N', 'I', 'A', 'A', 'N', 'I', 'R', 'R', 'A', 'T', 'A', 'P', 'Z', 'I',
 'T', 'A', 'C', 'A', 'E', 'M', 'M', 'S', 'S', 'T', 'E', 'R', 'R', 'A', 'P', 'L', 'E', 'N', 'E', 'N',
 'N', 'E', 'R', 'C', 'C', 'T', 'O', 'A', 'E', 'N', 'A', 'C', 'A', 'L', 'A', 'M', 'A', 'R', 'R', 'C',
 'U', 'T', 'E', 'U', 'I', 'A', 'U', 'N', 'S', 'G', 'E', 'N', 'R', 'E', 'H', 'C', 'I', 'P', 'I', 'A',
 'G', 'S', 'I', 'L', 'N', 'S', 'S', 'B', 'A', 'O', 'U', 'M', 'T', 'R', 'A', 'S', 'N', 'O', 'M', 'U',
 'E', 'E', 'D', 'T', 'O', 'R', 'A', 'G', 'O', 'R', 'I', 'R', 'I', 'B', 'A', 'S', 'T', 'O', 'L', 'T',
 'R', 'U', 'O', 'A', 'O', 'R', 'U', 'B', 'A', 'L', 'C', 'D', 'O', 'D', 'D', 'I', 'A', 'N', 'A', 'A',
 'P', 'T', 'C', 'D', 'P', 'A', 'N', 'I', 'Z', 'O', 'T', 'A', 'H', 'U', 'R', 'A', 'C', 'A', 'N', 'R'
]

#image7
image7_gt = [
 'O', 'R', 'I', 'P', 'A', 'P', 'O', 'R', 'T', 'A', 'L', 'C', 'A', 'P', 'I', 'S', 'C', 'A', 'R', 'F', 
 'G', 'O', 'D', 'A', 'G', 'A', 'P', 'V', 'D', 'E', 'S', 'P', 'E', 'C', 'H', 'O', 'H', 'I', 'A', 'P', 
 'E', 'O', 'T', 'R', 'E', 'P', 'X', 'E', 'I', 'R', 'E', 'C', 'R', 'E', 'J', 'E', 'U', 'R', 'L', 'R', 
 'N', 'R', 'A', 'E', 'F', 'L', 'O', 'G', 'A', 'T', 'R', 'E', 'K', 'O', 'J', 'S', 'S', 'A', 'E', 'E', 
 'E', 'E', 'G', 'L', 'O', 'G', 'A', 'A', 'C', 'H', 'I', 'N', 'C', 'H', 'E', 'A', 'M', 'R', 'J', 'N', 
 'T', 'R', 'I', 'A', 'L', 'O', 'A', 'S', 'E', 'R', 'B', 'R', 'S', 'I', 'R', 'R', 'E', 'R', 'A', 'D', 
 'I', 'E', 'R', 'C', 'O', 'T', 'S', 'E', 'R', 'P', 'A', 'A', 'E', 'U', 'F', 'F', 'A', 'E', 'D', 'A', 
 'S', 'M', 'R', 'O', 'D', 'A', 'I', 'F', 'O', 'Z', 'R', 'M', 'T', 'P', 'P', 'I', 'R', 'B', 'O', 'R', 
 'T', 'E', 'T', 'E', 'H', 'C', 'R', 'O', 'C', 'T', 'I', 'A', 'R', 'I', 'A', 'R', 'T', 'A', 'R', 'C',
 'A', 'T', 'O', 'R', 'T', 'N', 'E', 'D', 'N', 'O', 'L', 'R', 'M', 'E', 'A', 'F', 'T', 'N', 'A', 'Z', 
 'P', 'A', 'S', 'M', 'A', 'R', 'A', 'E', 'O', 'E', 'E', 'E', 'T', 'E', 'C', 'L', 'E', 'R', 'O', 'A', 
 'E', 'L', 'O', 'G', 'I', 'O', 'I', 'S', 'R', 'C', 'N', 'N', 'L', 'A', 'U', 'U', 'E', 'T', 'C', 'P', 
 'A', 'L', 'L', 'I', 'M', 'M', 'S', 'P', 'A', 'T', 'U', 'O', 'E', 'C', 'C', 'T', 'R', 'T', 'A', 'A', 
 'R', 'A', 'I', 'C', 'I', 'V', 'N', 'E', 'O', 'L', 'S', 'P', 'O', 'M', 'A', 'I', 'I', 'S', 'S', 'L', 
 'C', 'A', 'M', 'P', 'A', 'R', 'O', 'N', 'B', 'O', 'T', 'C', 'E', 'R', 'I', 'D', 'C', 'L', 'O', 'E'
]

"""print([word_dict2[key] for key in image8_gt])
print([word_dict2[key] for key in image7_gt])"""