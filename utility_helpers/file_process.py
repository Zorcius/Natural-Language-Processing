import numpy as np

prediction1 = np.array(open("/home/hy/jq/prediction_htt.txt",encoding='utf-8').readlines())
label = np.array(open("/home/hy/jq/prediction1.txt",encoding='utf-8').readlines())

print(len(prediction1) - len(prediction1==label))