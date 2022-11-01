import woodwork as ww
import cudf 
import pandas as pd 

df = pd.DataFrame()
df['test'] = ['this is a natural language', 'please be a column yes please ', 'discomfort endlessly has pulled itself on me', 'distracting reacting against my will i stand gainst my own reflect its haunting i cant seem'] 
df.ww.init(name='e') 
print(df.ww.logical_types) 


df = cudf.DataFrame()
df['key'] = [0, 1, 2, 3]
df['test'] = ['this is a natural language', 'please be a column yes please ', 'discomfort endlessly has pulled itself on me', 'distracting reacting against my will i stand gainst my own reflect its haunting i cant seem'] 
df['user'] = ['user@gmail.com', 'user@gmail.com', 'user@gamil.com', 'ef@gma.com']

for k, v in df.iterrows(): 
    print(k, v) 

df.ww.init(name='e')
print(df.ww.logical_types)
#print(df['test'].str.match('(^[a-zA-Z0-9_.+\-]+@[a-zA-Z0-9\-]+\.[a-zA-Z0-9\-.]+$)'))
