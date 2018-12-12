# Copyright (c) 2019 American Express Travel Related Services Company, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License. You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software distributed under the License
# is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
# or implied. See the License for the specific language governing permissions and limitations under
# the License.

import numpy as numpy
import pandas as pd
import random

max_features = 4000
num_examples = 32000
min_length = 150
max_length = 450
chunks = 16

df = pd.DataFrame(columns = ['x1','y'])

for i in range(num_examples):
	exampleLength = random.sample(range(min_length,max_length),1)[0]
	example = [str(num) for num in random.sample(range(1,max_features+1),exampleLength)]
	example =' '.join(example)
	label = str(random.sample(range(0,2),1)[0])
	df.loc[len(df)]=[example,label]

chunkSize = int(num_examples/chunks)
for i in range(chunks):
	start = i*chunkSize
	end = (i+1)*chunkSize
	print('Chunk',i)
	print(start,end)
	df[start:end].to_csv('chunk'+str(i+1)+'.csv',index=False)


